import pandas as pd
import os
import sys
import mlflow
import numpy as np

# Add project root to path
project_root = '/workspace'
sys.path.append(project_root)

from src.python.utils.preprocess import prepare_data, features_to_drop, target

def align_features_with_model(X_test, model):
    """Align test features with model's expected features."""
    # Get feature names from the model (first step of pipeline)
    model_features = model.named_steps['scaler'].feature_names_in_
    
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
    X_test, _ = prepare_data(test_df, target, features_to_drop)
    
    # Load the final model
    model_path = "models/final/duration_model"
    final_model = mlflow.sklearn.load_model(model_path)
    
    # Align features with model's expectations
    X_test = align_features_with_model(X_test, final_model)
    
    # Make predictions and ensure non-negative values
    predictions = np.maximum(final_model.predict(X_test), 1)
    
    # Add predictions to test dataframe
    test_df['Predicted_Duration'] = predictions
    
    # Save predictions
    os.makedirs('results/predictions', exist_ok=True)
    test_df.to_csv('results/predictions/duration_predictions.csv', index=False)
    
    print("\nPredictions Summary:")
    print(f"Mean Predicted Duration: {predictions.mean():.2f} minutes")
    print(f"Min Predicted Duration: {predictions.min():.2f} minutes")
    print(f"Max Predicted Duration: {predictions.max():.2f} minutes")
    print(f"\nPredictions saved to: results/predictions/duration_predictions.csv")

if __name__ == "__main__":
    main()
