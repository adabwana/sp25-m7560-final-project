import pandas as pd
import os
import sys
import mlflow
import numpy as np

# Add project root to path
project_root = '/workspaces/sp25-m7560-final-project'
sys.path.append(project_root)

from src.python.utils.preprocess import prepare_data, features_to_drop, target_2

def align_features_with_model(X_test, model):
    """Align test features with model's expected features."""
    # Handle RoundedRegressor wrapper
    if hasattr(model, 'estimator_'):
        base_model = model.estimator_
    else:
        base_model = model
    
    # Get feature names from the model (first step of pipeline)
    model_features = base_model.named_steps['scaler'].feature_names_in_
    
    # Find missing and extra features
    missing_features = set(model_features) - set(X_test.columns)
    extra_features = set(X_test.columns) - set(model_features)
    
    if missing_features:
        print("\nWarning: Features missing in test data (will be filled with zeros):")
        for feature in sorted(missing_features):
            print(f"- {feature}")
        # Add missing columns with zeros
        for col in missing_features:
            X_test[col] = 0
    
    if extra_features:
        print("\nWarning: Extra features in test data (will be dropped):")
        for feature in sorted(extra_features):
            print(f"- {feature}")
        # Drop extra columns
        X_test = X_test.drop(columns=list(extra_features))
    
    # Ensure column order matches model's expected order
    X_test = X_test.reindex(columns=model_features)
    
    return X_test

def main():
    """Make predictions using the final trained model."""
    # Load test data
    test_df = pd.read_csv(f'{project_root}/data/processed/test_engineered.csv')
    
    # Prepare test data using preprocessing pipeline
    X_test, _ = prepare_data(test_df, target_2, features_to_drop)
    
    # Load the final model
    model_path = "models/final/occupancy_model"
    final_model = mlflow.sklearn.load_model(model_path)
    
    # Align features with model's expectations
    X_test = align_features_with_model(X_test, final_model)
    
    # Make predictions, ensure non-negative values, and round to integers
    raw_predictions = final_model.predict(X_test)
    predictions = np.round(np.maximum(raw_predictions, 1)).astype(int)
    
    # Add predictions to test dataframe
    test_df['Predicted_Occupancy'] = predictions
    
    # Save predictions
    os.makedirs('results/predictions', exist_ok=True)
    test_df.to_csv('results/predictions/occupancy_predictions.csv', index=False)
    
    print("\nPredictions Summary:")
    print(f"Mean Predicted Occupancy: {predictions.mean():.1f} students")
    print(f"Min Predicted Occupancy: {predictions.min()} students")
    print(f"Max Predicted Occupancy: {predictions.max()} students")
    print(f"\nPredictions saved to: results/predictions/occupancy_predictions.csv")

if __name__ == "__main__":
    main()
