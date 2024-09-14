from ray import tune
from ray.tune.schedulers import HyperBandScheduler
import config as cfg

import sys

from data_parallel import data_parallel_main

config_space = {
    'do_data_parallel': cfg.do_data_parallel,
    'batch_size': tune.grid_search(cfg.batch_size),  # Specify the batch sizes you want to tune
    'learning_rate': cfg.learning_rate,
    'epochs': cfg.epochs,
    'train_data_size': cfg.train_data_size,
    'test_data_size': cfg.test_data_size,
    'max_length': cfg.max_length,
    'model_name': cfg.model_name,
    'device': cfg.device
}

scheduler = HyperBandScheduler(
    metric="loss",  # Specify the metric to optimize
    mode="min"
)

trainable_with_resources = tune.with_resources(data_parallel_main, {"gpu": cfg.num_gpu})

tuner = tune.Tuner(trainable=trainable_with_resources, 
                    param_space=config_space, 
                    tune_config=tune.TuneConfig(num_samples=cfg.num_samples, 
                                                scheduler=scheduler))

results = tuner.fit()

best_trial = results.get_best_result("loss", "min", "last")

best_trial_config = best_trial.config
print("Best trial config {batch_size, learning_rate, epochs}: ", 
            {best_trial_config['batch_size'], best_trial_config['learning_rate'], best_trial_config['epochs']})

print("Best trial final training loss:", best_trial.metrics["loss"])