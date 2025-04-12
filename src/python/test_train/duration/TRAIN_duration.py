import sys
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import mlflow
import os

# Local imports
project_root = '/workspace'
sys.path.append(project_root)

from src.python.utils.logging_config import setup_logger
from src.python.utils.mlflow_utils import setup_mlflow_experiments
from src.python.models.algorithms_duration import get_model_definitions
from src.python.models.pipelines import get_pipeline_definitions
from src.python.models.cross_validation import get_cv_methods
from src.python.utils.preprocess import prepare_data

logger = setup_logger()

def load_and_prepare_data(project_root: str):
    """Load and prepare the dataset."""
    df = pd.read_csv(f'{project_root}/data/processed/train_engineered.csv')
    
    target = 'Duration_In_Min'
    target_2 = 'Occupancy'
    features_to_drop = ['Student_IDs', 'Semester', 'Class_Standing', 'Major', 'Expected_Graduation',
                       'Course_Name', 'Course_Number', 'Course_Type', 'Course_Code_by_Thousands',
                       'Check_Out_Time', 'Session_Length_Category', target, target_2]
    
    X, y = prepare_data(df, target, features_to_drop)
    return train_test_split(X, y, test_size=0.2, shuffle=False)

def train_models(X_train, y_train, X_test, y_test):
    """Main training function."""
    # Setup MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_base = "Duration_Pred"
    
    # Get definitions
    models = get_model_definitions()
    pipelines = get_pipeline_definitions()
    cv_methods = get_cv_methods(len(X_train))
    scalers = [RobustScaler(), StandardScaler(), MinMaxScaler()]
    
    # Setup experiments
    setup_mlflow_experiments(experiment_base, models.keys())
    mlflow.sklearn.autolog()
    
    results = []
    for name, (model, params) in models.items():
        logger.info(f"\nStarting training for model: {name}")
        
        # Set the experiment for this model type
        experiment_name = f"{experiment_base}/{name}"
        mlflow.set_experiment(experiment_name)
        
        for scale_type, pipeline_func in pipelines.items():
            pipeline = pipeline_func(model)
            
            # Create parameter grid
            pipeline_params = {
                'scaler': scalers,
                **{f'model__{key.split("model__")[1]}': value 
                   for key, value in params.items() 
                   if key.startswith('model__')},
                **{f'select_features__{key.split("select_features__")[1]}': value 
                   for key, value in params.items() 
                   if key.startswith('select_features__')}
            }
            
            # Filter valid parameters
            valid_params = {k: v for k, v in pipeline_params.items() 
                          if k in pipeline.get_params()}
            
            for cv_name, cv in cv_methods.items():
                try:
                    results.extend(
                        train_single_model(
                            name, scale_type, cv_name, pipeline, 
                            valid_params, cv, X_train, y_train
                        )
                    )
                except Exception as e:
                    logger.error(f"Error during {name} fitting: {str(e)}")
                finally:
                    gc.collect()
    
    return results

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
    
    with joblib.parallel_backend('loky'):
        search.fit(X_train, y_train)
    
    rmse_score = -search.best_score_
    best_index = search.best_index_
    cv_std = search.cv_results_['std_test_score'][best_index]
    
    # Create consistent model name
    model_name = f"{name}_{scale_type}_{cv_name}"
    
    # Register the model with MLflow
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_metric("rmse", rmse_score)
        mlflow.log_metric("rmse_std", cv_std)
        
        # Log parameters
        for param_name, param_value in search.best_params_.items():
            mlflow.log_param(param_name, param_value)
        
        # Save and register the model with consistent naming
        mlflow.sklearn.log_model(
            search.best_estimator_,
            "model",
            registered_model_name=model_name
        )
    
    return [{
        'model': name,
        'pipeline_type': scale_type,
        'cv_method': cv_name,
        'rmse': rmse_score,
        'rmse_std': cv_std,
        'best_params': search.best_params_
    }]

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
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            continue
    
    return final_results

def main():
    # Create necessary directories
    os.makedirs(f'{project_root}/results/duration', exist_ok=True)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(project_root)
    
    # Train models
    results = train_models(X_train, y_train, X_test, y_test)
    
    # Save basic results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{project_root}/results/duration/training_results.csv', index=False)
    print("\nTraining Results:")
    print(results_df.sort_values(['rmse']))
    
    # Basic evaluation on test set
    final_results = evaluate_final_models(results, X_test, y_test)
    final_df = pd.DataFrame(final_results)
    final_df.to_csv(f'{project_root}/results/duration/test_evaluation.csv', index=False)
    
    logger.info("\nResults saved to results/duration/")

if __name__ == "__main__":
    main()
