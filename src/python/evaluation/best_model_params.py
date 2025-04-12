import pandas as pd
import os
import mlflow
import json
from mlflow.tracking import MlflowClient
import sys

def get_best_model_params(eval_path, experiment_base):
    """Get the best model and its parameters using MLflow."""
    # Read evaluation results
    df = pd.read_csv(eval_path)
    
    # Get best model combination
    best_row = df.loc[df['RMSE'].idxmin()]
    model = best_row['Model']
    pipeline = best_row['Pipeline']
    cv_method = best_row['CV_Method']
    
    # Connect to MLflow
    client = MlflowClient()
    
    # Get experiment ID
    experiment = client.get_experiment_by_name(f"{experiment_base}/{model}")
    if experiment is None:
        return f"No experiment found with name: {experiment_base}/{model}"
    
    # Construct model name as done in training
    model_name = f"{model}_{pipeline}_{cv_method}"
    
    try:
        # Search for runs with matching run_name
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attributes.run_name = '{model_name}'"
        )
        
        if not runs:
            return f"No matching runs found for {model_name}"
        
        # Get the best run (most recent)
        best_run = runs[0]
        
        return {
            'model': model,
            'pipeline': pipeline,
            'cv_method': cv_method,
            'rmse': float(best_row['RMSE']),
            'r2': float(best_row['R2']),
            'parameters': best_run.data.params,
            'run_id': best_run.info.run_id,
            'model_name': model_name
        }
    except Exception as e:
        return f"Error retrieving model {model_name}: {str(e)}"

def save_and_print_results(results, dataset_type, output_dir="results/best_models"):
    """Save results to JSON and print them."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{dataset_type}_best_model.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n====== Best Model for {dataset_type.title()} Prediction ======")
    if isinstance(results, dict):
        print(f"\nModel: {results['model']}")
        print(f"Pipeline: {results['pipeline']}")
        print(f"CV Method: {results['cv_method']}")
        print(f"RMSE: {results['rmse']:.4f}")
        print(f"R2: {results['r2']:.4f}")
        print(f"Run ID: {results['run_id']}")
        print("\nParameters:")
        for param, value in results['parameters'].items():
            print(f"{param}: {value}")
        print(f"\nResults saved to: {output_file}")
    else:
        print(results)

def main():
    # Add project root to Python path
    project_root = '/workspace'
    sys.path.append(project_root)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
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

if __name__ == "__main__":
    main() 