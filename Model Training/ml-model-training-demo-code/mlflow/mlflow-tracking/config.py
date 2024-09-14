# Define hyperparameters
do_data_parallel = False

batch_size = [8, 16]
learning_rate = 2e-5
epochs = 1

train_data_size = 100
test_data_size = 100
max_length = 512
model_name = 'prajjwal1/bert-mini'

device = 'cuda'

memory_limit = 1.0

num_gpu = 1

num_samples = 1


MLFLOW_TRACKING_URI = 'http://localhost:5000'
MLFLOW_EXPERIMENT_ID = '1'
