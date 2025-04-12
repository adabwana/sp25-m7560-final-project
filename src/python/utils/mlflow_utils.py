import mlflow

def setup_mlflow_experiments(experiment_base: str, model_names: list):
    """Setup MLflow experiments hierarchy."""
    if mlflow.get_experiment_by_name(experiment_base) is None:
        mlflow.create_experiment(experiment_base)
    
    for model_name in model_names:
        experiment_name = f"{experiment_base}/{model_name}"
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)

def log_training_results(run_name: str, metrics: dict, params: dict, model, error=None):
    """Log results to MLflow."""
    with mlflow.start_run(run_name=run_name) as run:
        if error:
            mlflow.log_param("error", str(error))
            return
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=run_name
        ) 