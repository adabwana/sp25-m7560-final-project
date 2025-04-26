# Predictions

This chapter details our approach to deploying the final production models. Our prediction framework serves three key objectives:

1. **Model Deployment**: Retrain best configurations on complete training dataset
2. **Feature Consistency**: Ensure alignment between training and prediction features
3. **Output Generation**: Produce and validate final predictions

The core prediction code resides in [`src/python/predictions/final_models.py`](https://github.com/adabwana/f24-m7550-final-project/blob/master/src/python/predictions/final_models.py), implementing separate prediction pipelines for duration and occupancy tasks.

## Configuration Management

Our framework begins by loading the optimal model configurations identified during evaluation:

```python
def load_model_config(model_type):
    """Load the best model configuration from JSON."""
    json_path = f"results/best_models/{model_type}_best_model.json"
    with open(json_path, 'r') as f:
        return json.load(f)
```

These configurations capture the winning combinations of model architecture, pipeline variant, and cross-validation strategy from our evaluation phase.

## Production Model Training

We implement the final training phase using the complete training dataset:

```python
def train_model(model_type):
    """Train and save a model based on type (duration or occupancy)."""
    config = load_model_config(model_type)
    train_df = pd.read_csv(f'{project_root}/data/processed/train_engineered.csv')
    X_train, y_train = prepare_data(train_df, target_var, features_to_drop)
    
    pipeline = create_penalized_splines_pipeline(config['parameters'], model_type)
    pipeline.fit(X_train, y_train)
```

This approach maximizes model performance by utilizing all available training data while maintaining the validated configurations.

## Pipeline Implementation

We reconstruct the optimal pipeline configurations for each prediction task:

```python
def create_penalized_splines_pipeline(params, model_type='duration'):
    """Create pipeline with parameters from best model."""
    spline = SplineTransformer(degree=degree, n_knots=n_knots)
    ridge = Ridge(alpha=alpha)
    scaler = RobustScaler()
    
    base_pipeline = Pipeline([
        ('scaler', scaler),
        ('spline', spline),
        ('ridge', ridge)
    ])
```

The implementation maintains task-specific requirements:
- Duration models use vanilla pipelines with optimized splines
- Occupancy models incorporate RoundedRegressor for count constraints

## Feature Alignment

We ensure consistent feature processing between training and prediction:

```python
def align_features_with_model(X_test, model):
    """Align test features with model's expected features."""
    model_features = model.named_steps['scaler'].feature_names_in_
    
    missing_features = set(model_features) - set(X_test.columns)
    extra_features = set(X_test.columns) - set(model_features)
    return X_test.reindex(columns=model_features)
```

This alignment prevents feature mismatch issues during prediction on `LC_test`.

## Prediction Generation

We implement task-specific prediction protocols:

```python
def main():
    """Generate predictions on test data."""
    X_test, _ = prepare_data(test_df, target, features_to_drop)
    final_model = mlflow.sklearn.load_model(model_path)
    X_test = align_features_with_model(X_test, final_model)
    
    # Task-specific prediction constraints
    if model_type == 'duration':
        predictions = np.maximum(final_model.predict(X_test), 1)
    else:  # occupancy
        predictions = np.round(np.maximum(
            final_model.predict(X_test), 1
        )).astype(int)
```

Each task maintains its specific constraints:
- Duration predictions enforce positive values
- Occupancy predictions implement integer rounding

## Results Management

We maintain systematic organization of prediction outputs:

```python
def save_predictions(predictions, model_type):
    """Save predictions with appropriate formatting."""
    output_path = f'results/predictions/{model_type}_predictions.csv'
    test_df[f'Predicted_{model_type.title()}'] = predictions
    test_df.to_csv(output_path, index=False)
    
    # Generate summary statistics
    summary_stats = {
        'mean': predictions.mean(),
        'std': predictions.std(),
        'min': predictions.min(),
        'max': predictions.max()
    }
    return summary_stats
```

This approach ensures:
- Consistent output formatting
- Validation through summary statistics
- Clear organization of prediction files

These components work together to provide a robust prediction pipeline that maintains the methodological rigor established during model selection while maximizing prediction accuracy through full training data utilization.
