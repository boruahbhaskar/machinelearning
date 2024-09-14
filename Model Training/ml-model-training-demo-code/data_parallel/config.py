# Define hyperparameters
do_data_parallel = True

per_device_batch_size = 1
learning_rate = 2e-5
epochs = 1

train_data_size = 200
test_data_size = 20
max_length = 512
model_name = 'bert-large-uncased'


device = 'cuda'

# Flag used to simulate limited memory. Set to 1.0 if you wish to use 100% memory on each device
memory_limit = 1.0

# Only use the specified devices
visible_devices = [0,1,2,3]