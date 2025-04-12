import pandas as pd
import numpy as np
import os
import mlflow
import json
import sys
import shutil
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit

# Add project root to path
project_root = '/workspaces/sp25-m7560-final-project'
sys.path.append(project_root)

from src.python.utils.preprocess import prepare_data, features_to_drop, target, target_2
from src.python.models.algorithms_occupancy import RoundedRegressor

def load_model_config(model_type):
    """Load the best model configuration from JSON."""
    json_path = f"results/best_models/{model_type}_best_model.json"
    with open(json_path, 'r') as f:
        return json.load(f)

def create_penalized_splines_pipeline(params, model_type='duration'):
    """Create pipeline with parameters from best model."""
    # Handle different parameter names for duration vs occupancy
    if model_type == 'duration':
        alpha = float(params['model__ridge__alpha'])
        degree = int(params['model__spline__degree'])
        n_knots = int(params['model__spline__n_knots'])
    else:  # occupancy
        alpha = float(params['model__estimator__ridge__alpha'])
        degree = int(params['model__estimator__spline__degree'])
        n_knots = int(params['model__estimator__spline__n_knots'])
    
    # Create pipeline components
    spline = SplineTransformer(degree=degree, n_knots=n_knots)
    ridge = Ridge(alpha=alpha)
    scaler = RobustScaler()
    
    # Create base pipeline
    base_pipeline = Pipeline([
        ('scaler', scaler),
        ('spline', spline),
        ('ridge', ridge)
    ])
    
    # For occupancy, wrap with RoundedRegressor
    if model_type == 'occupancy':
        return RoundedRegressor(base_pipeline)
    
    return base_pipeline

def get_cv_splitter(cv_method, n_samples, n_splits=10):
    """Get the appropriate cross-validation splitter."""
    default_test_size = n_samples // (n_splits + 1)
    
    if cv_method == 'kfold':
        return KFold(n_splits=n_splits, shuffle=True, random_state=3)
    elif cv_method == 'rolling':
        return TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=default_test_size * 5,
            test_size=default_test_size
        )
    else:
        raise ValueError(f"Unsupported CV method: {cv_method}")

def train_model(model_type):
    """Train and save a model based on type (duration or occupancy)."""
    # Load configuration
    config = load_model_config(model_type)
    
    # Load and prepare full training data
    train_df = pd.read_csv(f'{project_root}/data/processed/train_engineered.csv')
    target_var = target if model_type == 'duration' else target_2
    X_train, y_train = prepare_data(train_df, target_var, features_to_drop)
    
    # Get CV splitter based on config and data size
    cv = get_cv_splitter(config['cv_method'], X_train.shape[0])
    
    # Create and train pipeline with cross-validation
    pipeline = create_penalized_splines_pipeline(config['parameters'], model_type)
    
    # Perform cross-validation for model evaluation
    scores = []
    for train_idx, val_idx in cv.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        pipeline.fit(X_fold_train, y_fold_train)
        score = pipeline.score(X_fold_val, y_fold_val)
        scores.append(score)
    
    print(f"\nCross-validation R² scores ({config['cv_method']}):")
    print(f"Mean: {np.mean(scores):.4f}")
    print(f"Std: {np.std(scores):.4f}")
    
    # Train final model on full dataset
    pipeline.fit(X_train, y_train)
    
    # Save the model (handle existing directory)
    model_path = f"models/final/{model_type}_model"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    
    os.makedirs('models/final', exist_ok=True)
    mlflow.sklearn.save_model(
        pipeline,
        path=model_path,
        serialization_format="cloudpickle"
    )
    
    print(f"\nFinal {model_type.title()} Model Training Complete")
    print(f"Model saved to: {model_path}")
    print(f"Model type: {config['model']}")
    print(f"Pipeline type: {config['pipeline']}")
    print("\nModel Parameters:")
    for param, value in config['parameters'].items():
        print(f"{param}: {value}")

def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Train duration model
    train_model('duration')
    
    # Train occupancy model
    train_model('occupancy')

if __name__ == "__main__":
    main()
