# Import necessary libraries and modules
import os  # Import the os module for operating system-related functions
import functools  # Import the functools module for higher-order functions

import torch.distributed as dist  # Import PyTorch's distributed module
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy  # Import FullyShardedDataParallel and ShardingStrategy from FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy  # Import size_based_auto_wrap_policy from FSDP

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload  # Import CPUOffload from FSDP

from tqdm import tqdm  # Import tqdm for progress tracking

import torch  # Import PyTorch
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertModel  # Import various components from Hugging Face Transformers
from datasets import load_dataset  # Import a function to load datasets
from torch.utils.data import DataLoader, TensorDataset  # Import PyTorch data loading utilities
from transformers.models.bert.modeling_bert import BertLayer  # Import BertLayer from Transformers
import torch.multiprocessing as mp  # Import the multiprocessing module

from torch.utils.data.distributed import DistributedSampler  # Import DistributedSampler for distributed data loading
import config as cfg  # Import a custom configuration module (assumed to exist)

# Initialize the distributed process group
def setup():
    dist.init_process_group("nccl")

# Cleanup the distributed process group
def cleanup():
    dist.destroy_process_group()

# Function for training the model
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()  # Set the model in training mode
    total_loss = torch.zeros(2).to(rank)  # Initialize a tensor to store the total loss

    if sampler:
        sampler.set_epoch(epoch)  # Set the epoch for the sampler if provided

    for batch in tqdm(train_loader, leave=False):  # Iterate over training data batches and display a progress bar
        input_ids, attention_mask, labels = batch  # Unpack batch into input IDs, attention masks, and labels
        input_ids, attention_mask, labels = input_ids.to(rank), attention_mask.to(rank), labels.to(rank)  # Move data to the specified device (e.g., GPU)

        optimizer.zero_grad()  # Zero out the gradients in the optimizer

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  # Perform a forward pass

        loss = outputs.loss  # Get the loss

        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update the model's parameters based on the gradients

        total_loss[0] += loss.item()  # Accumulate the loss
        total_loss[1] += len(batch)  # Accumulate the number of items in the batch

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)  # Sum the total loss across all distributed processes
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss[0] / total_loss[1]))  # Print the average loss

# Function for testing the model
def test(model, rank, world_size, test_loader):
    model.eval()  # Set the model in evaluation mode

    total_accuracy = torch.zeros(1).to(rank)  # Initialize a tensor to store the total accuracy

    all_true_labels_in_epoch = []
    all_predictions_in_epoch = []

    with torch.no_grad():  # Disable gradient calculation for testing
        for batch in tqdm(test_loader, leave=False):  # Iterate over test data batches and display a progress bar
            input_ids, attention_mask, labels = batch  # Unpack batch into input IDs, attention masks, and labels
            input_ids, attention_mask, labels = input_ids.to(rank), attention_mask.to(rank), labels.to(rank)  # Move data to the specified device

            outputs = model(input_ids, attention_mask=attention_mask)  # Perform a forward pass without labels
            logits = outputs.logits  # Get the model's output logits

            predicted_labels = torch.argmax(logits, dim=1)  # Get the predicted labels
            correct_predictions = torch.sum(predicted_labels == labels).item()  # Count correct predictions
            total_accuracy[0] += correct_predictions  # Accumulate correct predictions

            all_predictions_in_epoch.extend(predicted_labels.tolist())  # Append predicted labels to the list
            all_true_labels_in_epoch.extend(labels.tolist())  # Append true labels to the list

        dist.all_reduce(total_accuracy, op=dist.ReduceOp.SUM)  # Sum the total accuracy across all distributed processes

        if rank == 0:
            print(f'Accuracy on Test Set: {total_accuracy[0]/len(test_loader.dataset):.4f}')  # Print the accuracy

# Main function for Fully Sharded Data Parallel training
def fsdp_main(args):

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    batch_size = args['per_device_batch_size']
    learning_rate = args['learning_rate']
    epochs = args['epochs']

    train_data_size = args['train_data_size']
    test_data_size = args['test_data_size']
    max_length = args['max_length']

    # Load and preprocess the IMDb dataset
    dataset = load_dataset('imdb')
    train_texts, train_labels, test_texts, test_labels = (
        dataset['train']['text'][0:train_data_size], dataset['train']['label'][0:train_data_size],
        dataset['test']['text'][0:test_data_size], dataset['test']['label'][0:test_data_size]
    )

    # Initialize BERT tokenizer
    model_name = args['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize and encode the text data
    if local_rank == 0:
        print("Tokenizing Train Dataset")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    
    if local_rank == 0:
        print("Tokenizing Test Dataset")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    # Create PyTorch datasets
    if local_rank == 0:
        print("Train and Test Assignment")
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'],
                                  torch.tensor(train_labels))
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'],
                                 torch.tensor(test_labels))

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)

    setup()

    train_kwargs = {'batch_size': batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': batch_size, 'sampler': sampler2}

        # Create data loaders
    train_loader = DataLoader(train_dataset, **train_kwargs)  # Create a data loader for training data
    test_loader = DataLoader(test_dataset, **test_kwargs)  # Create a data loader for test data

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )

    # Define the BERT-based text classifier model
    if local_rank == 0:
        print("Loading Model")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification

    torch.cuda.set_device(local_rank)  # Set the GPU device

    torch.cuda.set_per_process_memory_fraction(cfg.memory_limit)  # Set GPU memory allocation limit

    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=torch.cuda.current_device())  # Wrap the model with Fully Sharded Data Parallel

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)  # Define the optimizer
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * epochs)  # Define the learning rate scheduler

    for epoch in range(1, epochs + 1):  # Iterate through training epochs
        train(args, model, rank, world_size, train_loader, optimizer, epoch)  # Perform training
        test(model, rank, world_size, test_loader)  # Perform testing
        
        scheduler.step()  # Update the learning rate scheduler
    
    dist.barrier()  # Synchronize all distributed processes
    cleanup()  # Clean up the distributed process group

if __name__ == '__main__':

    local_rank = int(os.environ['LOCAL_RANK'])  # Get the local rank

    total_devices = int(os.environ['WORLD_SIZE'])  # Get the total number of devices

    if local_rank == 0:
        print(f"Training on {total_devices} devices")  # Print the number of devices being used for training

    batch_size = cfg.per_device_batch_size * total_devices  # Calculate the total effective batch size

    if local_rank == 0:
        print("Per Device Batch Size = ", cfg.per_device_batch_size)  # Print the batch size per device
        print("Total Effective Batch Size = ", batch_size)  # Print the total effective batch size

    args = {'per_device_batch_size': cfg.per_device_batch_size, 'learning_rate': cfg.learning_rate,
            'epochs': cfg.epochs,
            'train_data_size': cfg.train_data_size, 'test_data_size': cfg.test_data_size, 'max_length': cfg.max_length,
            'model_name': cfg.model_name}  # Create a dictionary of training arguments
    
    fsdp_main(args)  # Start the Fully Sharded Data Parallel training
    max_memory_consumed = round(torch.cuda.max_memory_allocated() / 1e9, 2)  # Get and round the maximum GPU memory consumption in GB
    
    if local_rank == 0:
        print(f"Max Memory Consumed Per Device = {max_memory_consumed} GB")  # Print the maximum memory consumed per device
