"""
Flow of Demonstration

The code is the same as data parallel but includes MLflow loggers. Just highlight the loggers and show the logs in tracking server

"""


import torch
from datetime import datetime
import torch.nn as nn
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import config as cfg
import os
import mlflow


def train(model, train_loader, optimizer, epoch, rank, do_data_parallel=False):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, leave=False):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(rank), attention_mask.to(rank), labels.to(rank)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # A Mean of loss is required to be computed as calculated by each device
        if do_data_parallel and torch.cuda.device_count() > 1:
            loss = outputs.loss.mean()
        else:
            loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}')
    return avg_train_loss


def test(model, test_loader, rank):
    model.eval()
    total_accuracy = 0.0

    all_true_labels_in_epoch = []
    all_predictions_in_epoch = []

    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(rank), attention_mask.to(rank), labels.to(rank)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predicted_labels = torch.argmax(logits, dim=1)
            correct_predictions = torch.sum(predicted_labels == labels).item()
            total_accuracy += correct_predictions

            all_predictions_in_epoch.extend(predicted_labels.tolist())
            all_true_labels_in_epoch.extend(labels.tolist())

        accuracy = total_accuracy / len(test_loader.dataset)
        print(f'Accuracy on Test Set: {accuracy:.4f}')
        return all_true_labels_in_epoch, all_predictions_in_epoch, accuracy


def data_parallel_main(args):
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)

    do_data_parallel = args['do_data_parallel']

    batch_size = args['batch_size']
    learning_rate = args['learning_rate']
    epochs = args['epochs']

    train_data_size = args['train_data_size']
    test_data_size = args['test_data_size']
    max_length = args['max_length']

    parent_run = args['mlflow_parent_run']

    dataset = load_dataset('imdb')
    dataset = dataset.shuffle(seed=32)

    train_texts, train_labels, test_texts, test_labels = (
        dataset['train']['text'][0:train_data_size], dataset['train']['label'][0:train_data_size],
        dataset['test']['text'][0:test_data_size], dataset['test']['label'][0:test_data_size]
    )

    # Initialize BERT tokenizer
    model_name = args["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize and encode the text data
    print("Tokenizing Train Dataset")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    print("Tokenizing Test Dataset")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    # Create PyTorch datasets
    print("Train and Test Assignment")
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'],
                                  torch.tensor(train_labels))
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'],
                                 torch.tensor(test_labels))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the BERT-based text classifier model
    print("Loading Model")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification

    # Enable or disable data parallel
    if do_data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=cfg.visible_devices)

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * epochs)

    # Training loop
    rank = torch.device(args['device'])
    torch.cuda.set_per_process_memory_fraction(cfg.memory_limit)
    model.to(rank)
    print("Training on " + str(rank))
    start_time = datetime.now()

    with mlflow.start_run(experiment_id = cfg.MLFLOW_EXPERIMENT_ID, nested=True):

        mlflow.set_tag("mlflow.parentRunId", parent_run.info.run_id)

        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('max_length', max_length)

        for epoch in tqdm(range(epochs)):
            train_loss = train(model, train_loader, optimizer, epoch, rank, do_data_parallel)
            labels, predicted_labels, test_accuracy = test(model, test_loader, rank)

            prediction_dict = {"inputs": test_texts, "outputs": predicted_labels, "true": labels}

            mlflow.log_table(data=prediction_dict, artifact_file="classification_eval_results.json")

            mlflow.log_metric('Train Loss', train_loss, step=epoch)
            mlflow.log_metric('Test Accuracy', test_accuracy, step=epoch)

            scheduler.step()

        model_components = {"model": model, "tokenizer": tokenizer}

        mlflow.transformers.log_model(transformers_model=model_components, artifact_path="sequence_classifier",
                                  task="text-classification")

    end_time = datetime.now()

    print('Time taken per epoch (seconds): ' + str(((end_time - start_time).seconds) / epochs))

    return {'loss': train_loss}


if __name__ == '__main__':
    total_devices = len(cfg.visible_devices) if cfg.do_data_parallel else 1

    print(f"Training on {total_devices} devices")

    batch_size = cfg.per_device_batch_size * total_devices

    print("Per Device Batch Size = ", cfg.per_device_batch_size)
    print("Total Effective Batch Size = ", batch_size)

    args = {'do_data_parallel': cfg.do_data_parallel, 'batch_size': batch_size, 'learning_rate': cfg.learning_rate,
            'epochs': cfg.epochs,
            'train_data_size': cfg.train_data_size, 'test_data_size': cfg.test_data_size, 'max_length': cfg.max_length,
            'model_name': cfg.model_name, 'device': cfg.device}
    data_parallel_main(args)
    max_memory_consumed = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(f"Max Memory Consumed Per Device = {max_memory_consumed} GB")