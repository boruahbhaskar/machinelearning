from ray import tune
from ray.tune.schedulers import HyperBandScheduler
import config as cfg
import mlflow

from data_parallel_raytune_mlflow_tracking import data_parallel_main


# Starts a parent run to group all the tuning runs together
# When run using MLflow Projects, this will return the run object created my the CLI 'mlflow run ....'
mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
print('hey')
parent_run = mlflow.start_run(experiment_id=cfg.MLFLOW_EXPERIMENT_ID)

# Important to finish the parent run, otherwise a bug in Mlflow Projects logs everything to parent run instead
# of child runs
mlflow.end_run(status="FINISHED")


config_space = {
    'do_data_parallel': cfg.do_data_parallel,
    'batch_size': tune.grid_search(cfg.batch_size),  # Specify the batch sizes you want to tune
    'learning_rate': cfg.learning_rate,
    'epochs': cfg.epochs,
    'train_data_size': cfg.train_data_size,
    'test_data_size': cfg.test_data_size,
    'max_length': cfg.max_length,
    'model_name': cfg.model_name,
    'device': cfg.device,
    'mlflow_parent_run': parent_run
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