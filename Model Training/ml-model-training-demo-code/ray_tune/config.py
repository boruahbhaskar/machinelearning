# Define hyperparameters
do_data_parallel = True

batch_size = [8, 16]
learning_rate = 2e-5
epochs = 2

train_data_size = 1000
test_data_size = 100
max_length = 512
model_name = 'prajjwal1/bert-mini'

device = 'cuda'
num_gpu = 1

num_samples = 1

# Flag used to simulate limited memory. Set to 1.0 if you wish to use 100% memory on each device
memory_limit = 1.0