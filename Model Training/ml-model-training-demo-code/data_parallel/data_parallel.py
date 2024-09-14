# Import necessary libraries and modules
import torch  # Import the PyTorch library
from datetime import datetime  # Import the datetime module for time tracking
import torch.nn as nn  # Import PyTorch's neural network module
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup  # Import specific modules from the Hugging Face Transformers library
from datasets import load_dataset  # Import a function to load datasets
from torch.utils.data import DataLoader, TensorDataset  # Import PyTorch data loading utilities
from tqdm import tqdm  # Import tqdm for progress tracking
import config as cfg  # Import a custom configuration module (assumed to exist)
import os  # Import the os module for operating system-related functions

# Function to train the model
def train(model, train_loader, optimizer, epoch, rank, do_data_parallel=False):
    model.train()  # Set the model in training mode
    total_loss = 0.0  # Initialize a variable to store the total loss
    for batch in tqdm(train_loader, leave=False):  # Iterate over training data batches and display a progress bar
        input_ids, attention_mask, labels = batch  # Unpack batch into input IDs, attention masks, and labels
        input_ids, attention_mask, labels = input_ids.to(rank), attention_mask.to(rank), labels.to(rank)  # Move data to the specified device (e.g., GPU)

        optimizer.zero_grad()  # Zero out the gradients in the optimizer

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  # Perform a forward pass

        # A Mean of loss is required to be computed as calculated by each device
        if do_data_parallel and torch.cuda.device_count() > 1:  # Check if data parallelism is enabled and multiple GPUs are available
            loss = outputs.loss.mean()  # Compute the mean loss across GPUs
        else:
            loss = outputs.loss  # Use the loss as-is if not using data parallelism

        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update the model's parameters based on the gradients

        total_loss += loss.item()  # Accumulate the loss

    avg_train_loss = total_loss / len(train_loader)  # Calculate the average training loss
    print(f'Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}')  # Print the average training loss for the epoch
    return avg_train_loss  # Return the average training loss

# Function to test the model
def test(model, test_loader, rank):
    model.eval()  # Set the model in evaluation mode
    total_accuracy = 0.0  # Initialize a variable to store the total accuracy

    all_true_labels_in_epoch = []  # Initialize a list to store true labels for the epoch
    all_predictions_in_epoch = []  # Initialize a list to store predicted labels for the epoch

    with torch.no_grad():  # Disable gradient calculation for testing
        for batch in tqdm(test_loader, leave=False):  # Iterate over test data batches and display a progress bar
            input_ids, attention_mask, labels = batch  # Unpack batch into input IDs, attention masks, and labels
            input_ids, attention_mask, labels = input_ids.to(rank), attention_mask.to(rank), labels.to(rank)  # Move data to the specified device

            outputs = model(input_ids, attention_mask=attention_mask)  # Perform a forward pass without labels
            logits = outputs.logits  # Get the model's output logits

            predicted_labels = torch.argmax(logits, dim=1)  # Get the predicted labels
            correct_predictions = torch.sum(predicted_labels == labels).item()  # Count correct predictions
            total_accuracy += correct_predictions  # Accumulate correct predictions

            all_predictions_in_epoch.extend(predicted_labels.tolist())  # Append predicted labels to the list
            all_true_labels_in_epoch.extend(labels.tolist())  # Append true labels to the list

        accuracy = total_accuracy / len(test_loader.dataset)  # Calculate the accuracy on the test set
        print(f'Accuracy on Test Set: {accuracy:.4f}')  # Print the accuracy
        return all_true_labels_in_epoch, all_predictions_in_epoch, accuracy  # Return true labels, predicted labels, and accuracy

# Main function for data parallel training
def data_parallel_main(args):
    
    do_data_parallel = args['do_data_parallel']  # Get a flag indicating whether to use data parallelism

    batch_size = args['batch_size']  # Get the batch size from the arguments
    learning_rate = args['learning_rate']  # Get the learning rate from the arguments
    epochs = args['epochs']  # Get the number of training epochs from the arguments

    train_data_size = args['train_data_size']  # Get the size of the training data from the arguments
    test_data_size = args['test_data_size']  # Get the size of the test data from the arguments
    max_length = args['max_length']  # Get the maximum sequence length from the arguments

    # Load the IMDb dataset and shuffle it
    dataset = load_dataset('imdb')
    dataset = dataset.shuffle(seed=32)

    # Get a subset of the dataset for training and testing
    train_texts, train_labels, test_texts, test_labels = (
        dataset['train']['text'][0:train_data_size], dataset['train']['label'][0:train_data_size],
        dataset['test']['text'][0:test_data_size], dataset['test']['label'][0:test_data_size]
    )

    # Initialize BERT tokenizer
    model_name = args["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize and encode the text data
    print("Tokenizing Train Dataset")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')  # Tokenize training data
    print("Tokenizing Test Dataset")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')  # Tokenize test data

    # Create PyTorch datasets
    print("Train and Test Assignment")
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'],
                                    torch.tensor(train_labels))  # Create a training dataset
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'],
                                    torch.tensor(test_labels))  # Create a test dataset

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Create a data loader for training data
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Create a data loader for test data

    # Define the BERT-based text classifier model
    print("Loading Model")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification

    # Enable or disable data parallel if multiple GPUs are available
    if do_data_parallel and torch.cuda.device_count() > 1:  # Check if data parallelism is enabled
        model = nn.DataParallel(model, device_ids=cfg.visible_devices)  # Enable data parallelism across multiple GPUs

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)  # Define the optimizer
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * epochs)  # Define the learning rate scheduler

    # Training loop
    rank = torch.device(args['device'])  # Get the device for training
    torch.cuda.set_per_process_memory_fraction(cfg.memory_limit)  # Set GPU memory allocation limit
    model.to(rank)  # Move the model to the specified device
    print("Training on " + str(rank))  # Print the device being used for training
    start_time = datetime.now()  # Record the start time for training

    for epoch in tqdm(range(epochs)):  # Iterate through training epochs
        train_loss = train(model, train_loader, optimizer, epoch, rank, do_data_parallel)  # Perform training and get the average loss
        labels, predicted_labels, test_accuracy = test(model, test_loader, rank)  # Perform testing and get accuracy

        scheduler.step()  # Update the learning rate scheduler

    end_time = datetime.now()  # Record the end time for training

    print('Time taken per epoch (seconds): ' + str(((end_time - start_time).seconds) / epochs))  # Calculate and print the time taken per epoch

    return {'loss': train_loss}  # Return the final training loss

if __name__ == '__main__':
    total_devices = len(cfg.visible_devices) if cfg.do_data_parallel else 1  # Determine the total number of devices used for training

    print(f"Training on {total_devices} devices")  # Print the number of devices being used for training

    batch_size = cfg.per_device_batch_size * total_devices  # Calculate the total effective batch size

    print("Per Device Batch Size = ", cfg.per_device_batch_size)  # Print the batch size per device
    print("Total Effective Batch Size = ", batch_size)  # Print the total effective batch size

    args = {'do_data_parallel': cfg.do_data_parallel, 'batch_size': batch_size, 'learning_rate': cfg.learning_rate,
            'epochs': cfg.epochs,
            'train_data_size': cfg.train_data_size, 'test_data_size': cfg.test_data_size, 'max_length': cfg.max_length,
            'model_name': cfg.model_name, 'device': cfg.device}  # Create a dictionary of training arguments
    data_parallel_main(args)  # Start the training process
    max_memory_consumed = round(torch.cuda.max_memory_allocated() / 1e9, 2)  # Get and round the maximum GPU memory consumption
    print(f"Max Memory Consumed Per Device = {max_memory_consumed} GB")  # Print the maximum memory consumed per device
