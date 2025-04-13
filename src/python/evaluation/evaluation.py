import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
from typing import Dict, Any
import os
import sys

# Add project root to handle imports if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

# Import the model architecture - adjust path if necessary
from python.models.gruey_architecture import GrueyModel

def load_pytorch_model(model_path: str, model_architecture: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Loads a PyTorch model's state dictionary onto a specified device.

    Args:
        model_path (str): Path to the saved .pth state dictionary file.
        model_architecture (torch.nn.Module): An instance of the model architecture 
                                             (e.g., GrueyModel(input_dim, gru_dim)).
        device (torch.device): The device ('cpu' or 'cuda') to load the model onto.

    Returns:
        torch.nn.Module: The model loaded with the state dictionary and moved to the device.
        
    Raises:
        FileNotFoundError: If the model_path does not exist.
        Exception: For other potential loading errors.
    """
    try:
        print(f"Loading model state dictionary from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model_architecture.load_state_dict(state_dict)
        model_architecture.to(device) # Ensure model is on the correct device
        model_architecture.eval() # Set model to evaluation mode
        print("Model loaded successfully and set to evaluation mode.")
        return model_architecture
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        raise
    except Exception as e:
        print(f"Error loading model state dictionary: {e}")
        raise

def predict_on_loader(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    """Makes predictions on a DataLoader using a trained model.

    Args:
        model (torch.nn.Module): The trained PyTorch model (already loaded and in eval mode).
        data_loader (DataLoader): DataLoader containing the data to predict on.
        device (torch.device): The device the model and data should be on.

    Returns:
        np.ndarray: A numpy array containing the predictions.
    """
    model.eval() # Ensure model is in evaluation mode
    all_predictions = []
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            outputs = model(features)
            all_predictions.append(outputs.cpu().numpy()) # Move predictions to CPU before converting to numpy
            
    # Concatenate predictions from all batches
    predictions_np = np.concatenate(all_predictions, axis=0)
    return predictions_np.squeeze() # Remove dimensions of size 1 if present (e.g., [N, 1] -> [N])

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates standard regression metrics.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        Dict[str, float]: A dictionary containing RMSE, R-squared (R2), and MAE.
    """
    if y_true.shape != y_pred.shape:
         # Allow for broadcasting if y_true is (N, 1) and y_pred is (N,)
        if not (y_true.ndim == 2 and y_true.shape[1] == 1 and y_pred.ndim == 1 and y_true.shape[0] == y_pred.shape[0]):
             raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        y_true = y_true.squeeze() # Ensure y_true is also 1D if y_pred is
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    metrics = {
        'rmse': rmse,
        'r2': r2,
        'mae': mae
    }
    
    print("\n--- Calculated Test Set Metrics ---")
    for name, value in metrics.items():
        print(f"  {name.upper()}: {value:.4f}")
    print("---------------------------------")
    
    return metrics

def evaluate_saved_model(model_path: str, 
                         model_architecture_instance: torch.nn.Module, # Accept instantiated model 
                         test_loader: DataLoader, 
                         y_test: pd.Series, 
                         device: torch.device) -> Dict[str, float]:
    """Loads a saved model state dict into the provided architecture, 
    predicts on test data, and calculates regression metrics.

    Args:
        model_path (str): Path to the saved model state dictionary.
        model_architecture_instance (torch.nn.Module): An instance of the model class 
                                                      (e.g., GrueyModel(...), TabularTransformerModel(...)).
                                                      The state dict will be loaded into this instance.
        test_loader (DataLoader): DataLoader for the test set.
        y_test (pd.Series): The actual target values for the test set.
        device (torch.device): The device to use ('cpu' or 'cuda').

    Returns:
        Dict[str, float]: Dictionary containing RMSE, R2, and MAE.
    """
    # 1. Load the state dict into the provided model instance
    # The instance should already be created with correct dimensions outside this function
    loaded_model = load_pytorch_model(model_path, model_architecture_instance, device)
    
    # 2. Make predictions
    print("\nMaking predictions on the test set with the loaded model...")
    y_pred_np = predict_on_loader(loaded_model, test_loader, device)
    
    # 3. Prepare true values
    y_true_np = y_test.values # Convert pandas Series to numpy array
    
    # Check shapes before calculating metrics
    if y_true_np.shape[0] != y_pred_np.shape[0]:
         raise ValueError(f"Mismatch in number of samples: y_true has {y_true_np.shape[0]} but y_pred has {y_pred_np.shape[0]}")

    # 4. Calculate metrics
    print("\nCalculating final evaluation metrics on the test set...")
    metrics = calculate_regression_metrics(y_true_np, y_pred_np)
    
    return metrics 