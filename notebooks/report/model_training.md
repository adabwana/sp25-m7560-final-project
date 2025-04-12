# Model Training

This chapter details our systematic approach to model training. Our training framework serves three key objectives:

1. **Model Development**: Train multiple model architectures with different configurations
2. **Hyperparameter Optimization**: Find optimal parameters through systematic grid search
3. **Performance Tracking**: Monitor and log training metrics for model selection

The core training code is shared between two scripts: [`src/python/test_train/duration/TRAIN_duration.py`](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/python/test_train/duration/TRAIN_duration.py) and [`src/python/test_train/occupancy/TRAIN_occupancy.py`](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/python/test_train/occupancy/TRAIN_occupancy.py). Like our testing scripts, these share approximately 95% of their code, differing primarily in:

1. Target variable selection (`Duration_In_Min` vs `Occupancy`)
2. Model definitions import (`algorithms_duration.py` vs `algorithms_occupancy.py`)
3. Output paths (`results/duration/` vs `results/occupancy/`)
4. MLflow experiment naming (`Duration_Pred` vs `Occupancy_Pred`)

We use MLflow to track experiments and maintain model versioning, enabling systematic comparison of different training configurations and reproducible results.

The complete implementation can be found in our [GitHub repository](https://github.com/adabwana/f24-m7550-final-project/).

## Data Preparation and Loading

Our first step involves loading the engineered features and preparing them for training:

```python
def load_and_prepare_data(project_root: str):
    """Load and prepare the dataset."""
    df = pd.read_csv(f'{project_root}/data/processed/train_engineered.csv')
    
    target = 'Duration_In_Min'  # Changes to 'Occupancy' for occupancy training
    features_to_drop = ['Student_IDs', 'Semester', 'Class_Standing', 'Major', 
                       'Expected_Graduation', ...]
    
    X, y = prepare_data(df, target, features_to_drop)
    return train_test_split(X, y, test_size=0.2, shuffle=False)
```

We maintain chronological order by setting `shuffle=False`, as both duration and occupancy predictions exhibit temporal patterns. The features we drop are either identifiers or categorical variables already encoded during feature engineering.

## Core Training Architecture

The main training function orchestrates the entire process, setting up experiments and managing model iterations:

```python
def train_models(X_train, y_train, X_test, y_test):
    """Main training function."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_base = "Duration_Pred"
    
    # Get definitions
    models = get_model_definitions()
    pipelines = get_pipeline_definitions()
    cv_methods = get_cv_methods(len(X_train))
    scalers = [RobustScaler(), StandardScaler(), MinMaxScaler()]
```

This setup enables systematic experimentation across different model types, scaling approaches, and cross-validation strategies. The MLflow integration ensures reproducibility and experiment tracking.

## Single Model Training Implementation

The core training logic for individual models incorporates GridSearchCV with RMSE scoring:

```python
def train_single_model(name, scale_type, cv_name, pipeline, params, cv, X_train, y_train):
    """Train a single model configuration."""
    rmse_scorer = make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=False
    )
    
    search = GridSearchCV(
        pipeline, 
        params,
        scoring=rmse_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        error_score='raise'
    )
```

The implementation leverages parallel processing through joblib's 'loky' backend and includes comprehensive error handling. Each training iteration is tracked through MLflow:

```python
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_metric("rmse", rmse_score)
        mlflow.log_metric("rmse_std", cv_std)
        
        # Log parameters
        for param_name, param_value in search.best_params_.items():
            mlflow.log_param(param_name, param_value)
```

## Model Evaluation Framework

The evaluation system provides systematic assessment of trained models:

```python
def evaluate_final_models(results, X_test, y_test):
    """Simple evaluation of models on test set without visualization."""
    final_results = []
    
    for result in results:
        model_name = f"{result['model']}_{result['pipeline_type']}_{result['cv_method']}"
        try:
            model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
            y_pred = model.predict(X_test)
            
            final_results.append({
                'Model': result['model'],
                'Pipeline': result['pipeline_type'],
                'CV_Method': result['cv_method'],
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred)
            })
```

This framework enables consistent evaluation across different model configurations while maintaining detailed performance metrics.

## Execution Flow

The main execution flow ties all components together:

```python
def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(project_root)
    
    # Train models
    results = train_models(X_train, y_train, X_test, y_test)
    
    # Save basic results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{project_root}/results/duration/training_results.csv', index=False)
    
    # Basic evaluation on test set
    final_results = evaluate_final_models(results, X_test, y_test)
    final_df = pd.DataFrame(final_results)
    final_df.to_csv(f'{project_root}/results/duration/test_evaluation.csv', index=False)
```

This structured approach ensures reproducible training runs while maintaining comprehensive result logging and evaluation.

## Resource Management

The implementation includes several key resource management features:

1. **Parallel Processing**: Utilizes joblib's parallel backend for efficient computation
2. **Memory Management**: Implements garbage collection after model training
3. **Error Recovery**: Includes comprehensive exception handling throughout the pipeline

These components work together to provide robust and efficient model training capabilities while maintaining consistent evaluation and deployment readiness.