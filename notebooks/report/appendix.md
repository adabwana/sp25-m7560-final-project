# Appendix

## Technical Details

Below are the technical details of our implementation from:

1) Training
2) Testing
3) Evaluation

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

# Model Testing

This chapter details our systematic approach to model evaluation. Our testing framework serves three key objectives:

1. **Performance Validation**: Assess how well our models generalize to unseen data
2. **Model Comparison**: Evaluate different architectures and strategies against each other
3. **Production Readiness**: Ensure models are reliable and stable for deployment

The core testing code is shared between two scripts: [`src/python/test_train/duration/TEST_duration.py`](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/python/test_train/duration/TEST_duration.py) and [`src/python/test_train/occupancy/TEST_occupancy.py`](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/python/test_train/occupancy/TEST_occupancy.py). Like our training scripts, these share approximately 95% of their code, differing primarily in:

1. Target variable selection (`Duration_In_Min` vs `Occupancy`)
2. Output paths (`results/duration/` vs `results/occupancy/`)
3. MLflow experiment base name (`Duration_Pred` vs `Occupancy_Pred`)

We use MLflow, an open-source platform for machine learning lifecycle management, to track our experiments and maintain model versioning. This enables reproducible testing and systematic comparison of different model configurations.

The complete implementation can be found in our [GitHub repository](https://github.com/adabwana/f24-m7550-final-project/).

## Data Preparation

Our testing framework uses the same data split that was established during training to ensure consistent evaluation. We maintain the original 80/20 split of our dataset, where the 20% holdout set was never seen during model training or validation:

```python
df = pd.read_csv(f'{project_root}/data/processed/train_engineered.csv')

target = 'Duration_In_Min'  # Changes to 'Occupancy' for occupancy testing
features_to_drop = ['Student_IDs', 'Semester', 'Class_Standing', 'Major', 
                   'Expected_Graduation', ...]

X, y = prepare_data(df, target, features_to_drop)

# Using same holdout set from training
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False,  # Maintains chronological order
    random_state=3  # Same seed used in training
)
```

By using the same holdout set and random seed as training, we ensure:

1. No data leakage between training and testing
2. Fair comparison across all model variants
3. Consistent evaluation of temporal patterns
4. Reliable assessment of model generalization

The framework tests models trained with three distinct cross-validation strategies:

1. **K-Fold** (`kfold`): Provides baseline performance through random splits, particularly suitable for duration prediction
2. **Rolling Window** (`rolling`): Uses fixed-size moving windows to capture recent temporal dependencies, optimal for occupancy prediction
3. **Expanding Window** (`expanding`): Implements growing windows that accumulate historical data, useful for long-term trends

These strategies are systematically evaluated in our testing loop:

```python
pipeline_types = ['vanilla', 'interact_select', 'pca_lda']
cv_methods = ['kfold', 'rolling', 'expanding']

for pipeline_type in pipeline_types:
    for cv_name in cv_methods:
        full_model_name = f"{model_name}_{pipeline_type}_{cv_name}"
```

This comprehensive approach allows us to assess how different cross-validation strategies affect model performance across our various pipeline configurations.

## MLflow Configuration

The testing framework relies heavily on MLflow for model management and experiment tracking:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def check_mlflow_connection():
    try:
        client = mlflow.tracking.MlflowClient()
        client.search_experiments()
        return True
    except Exception as e:
        print(f"Error connecting to MLflow server: {e}")
        print("Please ensure MLflow server is running...")
        return False
```

This setup includes robust error handling for MLflow connectivity, ensuring the testing pipeline fails gracefully if the MLflow server is unavailable.

## Model Testing Implementation

The core testing loop systematically evaluates each trained model variant:

```python
for experiment in experiments:
    model_name = experiment.name.split('/')[-1]
    print(f"\nTesting models for {model_name}...")
    
    # Pipeline types we used in training
    pipeline_types = ['vanilla', 'interact_select', 'pca_lda']
    cv_methods = ['kfold', 'rolling', 'expanding']
    
    for pipeline_type in pipeline_types:
        for cv_name in cv_methods:
            full_model_name = f"{model_name}_{pipeline_type}_{cv_name}"
            try:
                model = mlflow.sklearn.load_model(f"models:/{full_model_name}/latest")
                y_pred = model.predict(X_test)
                metrics = calculate_metrics(y_test, y_pred)
```

The implementation tests each combination of model type, pipeline configuration, and cross-validation method used during training.

## Performance Visualization

For each model variant, we generate comprehensive visualization artifacts:

```python
# Create prediction analysis plots
fig = plot_prediction_analysis(y_test, y_pred, full_model_name)
plt.savefig(f'{project_root}/results/duration/prediction_analysis_{full_model_name}.png')
plt.close()

# Create feature importance biplot for the best model
if metrics['RMSE'] < best_model_rmse:
    best_model_rmse = metrics['RMSE']
    best_model_predictions = y_pred
    plot_feature_importance_biplot(
        X_test, y_test, y_pred, 
        X_test.columns,
        f'{project_root}/results/duration'  # Changes for occupancy
    )
```

These visualizations include prediction analysis plots and feature importance biplots, particularly for the best-performing models.

## Results Management

The framework maintains systematic records of test results:

```python
# If we have results for this experiment, keep only top 3 based on RMSE
if experiment_results:
    top_3_results = sorted(experiment_results, key=lambda x: x['RMSE'])[:3]
    test_results.extend(top_3_results)

# Create results DataFrame
results_df = pd.DataFrame(test_results)
print("\nTop 3 Models per Experiment:")
print(results_df.sort_values('RMSE'))

# Save results and create visualization artifacts
save_visualization_results(results_df, project_root)
```

This approach ensures we maintain records of the best-performing models while generating comprehensive visualization artifacts for analysis.

## Resource Management

The testing framework includes several key features for robust execution:

1. **Error Handling**: Comprehensive try-except blocks around model loading and prediction
2. **Resource Cleanup**: Systematic closure of visualization artifacts
3. **Directory Management**: Automatic creation of results directories
4. **Progress Logging**: Detailed console output during testing

These components work together to provide reliable model evaluation while maintaining clear performance records and visualization artifacts.

# Evaluation Framework

This chapter details our systematic approach to model evaluation and comparison. Our evaluation framework serves three key objectives:

1. **Performance Analysis**: Aggregate and compare metrics across different model configurations
2. **Model Selection**: Identify and extract the best performing model configurations
3. **Result Documentation**: Generate comprehensive performance reports and visualizations

The core evaluation code is shared between two scripts: [`src/python/evaluation/model_evals.py`](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/python/evaluation/model_evals.py) and [`src/python/evaluation/best_model_params.py`](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/python/evaluation/best_model_params.py). These scripts work together to provide a complete evaluation pipeline for both duration and occupancy predictions.

## Performance Analysis Pipeline

Our framework begins by aggregating performance metrics across different model configurations:

```python
def load_and_analyze(filepath, dataset_name):
    """Hierarchical model performance analysis."""
    df = pd.read_csv(filepath)
    
    # Cross-validation strategy analysis
    cv_groups = df.groupby('CV_Method')[['RMSE', 'R2']].agg(['mean', 'std'])
    cv_groups = cv_groups.sort_values(('RMSE', 'mean'))
    
    # Pipeline architecture analysis
    pipeline_groups = df.groupby('Pipeline')[['RMSE', 'R2']].agg(['mean', 'std'])
    pipeline_groups = pipeline_groups.sort_values(('RMSE', 'mean'))
    
    # Model type analysis
    model_groups = df.groupby('Model')[['RMSE', 'R2']].agg(['mean', 'std'])
    model_groups = model_groups.sort_values(('RMSE', 'mean'))
```

This hierarchical analysis enables us to understand performance patterns across different aspects of our modeling approach.

## Best Model Identification

The framework systematically identifies and extracts the optimal model configurations:

```python
def get_best_model_params(eval_path, experiment_base):
    """Extract optimal model configuration."""
    # Identify best performer
    df = pd.read_csv(eval_path)
    best_row = df.loc[df['RMSE'].idxmin()]
    
    # Retrieve detailed configuration
    model_name = f"{best_row['Model']}_{best_row['Pipeline']}_{best_row['CV_Method']}"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(f"{experiment_base}/{best_row['Model']}")
```

This process ensures we capture not just the best performance metrics, but the complete configuration that achieved them.

## Results Documentation

The framework implements systematic result formatting and storage:

```python
def save_and_print_results(results, dataset_type):
    """Format and persist evaluation results."""
    print(f"\n====== Best Model for {dataset_type.title()} Prediction ======")
    print(f"Model: {results['model']}")
    print(f"Pipeline: {results['pipeline']}")
    print(f"CV Method: {results['cv_method']}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"R2: {results['r2']:.4f}")
    
    # Save configuration
    output_file = os.path.join(output_dir, f"{dataset_type}_best_model.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
```

This approach ensures consistent documentation of results across both prediction tasks.

## Evaluation Pipeline

The main evaluation pipeline processes both prediction tasks:

```python
def main():
    """Execute comprehensive evaluation pipeline."""
    # Process occupancy results
    occupancy_eval = "results/occupancy/test_evaluation.csv"
    if os.path.exists(occupancy_eval):
        occ_results = get_best_model_params(occupancy_eval, "Occupancy_Pred")
        save_and_print_results(occ_results, "occupancy")
    
    # Process duration results
    duration_eval = "results/duration/test_evaluation.csv"
    if os.path.exists(duration_eval):
        dur_results = get_best_model_params(duration_eval, "Duration_Pred")
        save_and_print_results(dur_results, "duration")
```

## Framework Components

The implementation includes several key features for comprehensive evaluation:

1. **Analysis Systems**
   - Performance metric aggregation
   - Cross-model comparison
   - Statistical analysis

2. **Result Processing**
   - Parameter extraction
   - Configuration logging
   - Performance ranking

3. **Output Generation**
   - Result formatting
   - Metric visualization
   - Configuration persistence

These components work together to provide systematic performance analysis and model selection capabilities, ensuring we can confidently identify and document our best performing models.