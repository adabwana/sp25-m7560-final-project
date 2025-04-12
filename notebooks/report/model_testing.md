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