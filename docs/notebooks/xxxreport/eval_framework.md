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