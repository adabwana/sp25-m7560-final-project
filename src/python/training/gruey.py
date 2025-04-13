import pandas as pd
import os
import sys
import torch # Add torch import needed in main block
from typing import Tuple, List
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Import StandardScaler
import optuna # Import Optuna

# Add project root for imports from other modules like utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add SRC directory to sys.path instead of project_root for direct execution
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

# Constants (Consider moving to a config file/module later)
#DATA_DIR = os.path.join(project_root, "data", "processed")
#TRAIN_FILE = os.path.join(DATA_DIR, "train_engineered.csv")

# --- Imports from project structure ---
# Place imports here, now that src_path is potentially added
from python.utils.data_utils import load_data
from python.utils.preprocessing import preprocess_data
from python.datasets import TabularDataset
from python.models.gruey_architecture import GrueyModel
from torch.utils.data import DataLoader
# from python.evaluation.evaluation import evaluate_saved_model 

# --- Training Step Function ---
def train_step(model: torch.nn.Module, 
               features: torch.Tensor, 
               targets: torch.Tensor, 
               loss_fn: torch.nn.modules.loss._Loss, # Type hint for loss functions
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> float: # Added device param
    """Performs a single training step: forward, loss, backward, optimize.

    Args:
        model (torch.nn.Module): The model to train.
        features (torch.Tensor): Batch of input features.
        targets (torch.Tensor): Batch of target values.
        loss_fn (torch.nn.modules.loss._Loss): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run the computation on.

    Returns:
        float: The loss value for this batch.
    """
    # Set model to training mode
    model.train()
    # 1. Forward pass
    features, targets = features.to(device), targets.to(device)
    predictions = model(features)
    # 2. Calculate loss
    loss = loss_fn(predictions, targets)
    # 3. Zero gradients before backward pass
    optimizer.zero_grad()
    # 4. Backward pass (compute gradients)
    loss.backward()
    # 5. Optimizer step (update weights)
    optimizer.step()
    # Return scalar loss value
    return loss.item()

# --- Evaluation Function ---
def evaluate_model(model: torch.nn.Module,
                   data_loader: DataLoader,
                   loss_fn: torch.nn.modules.loss._Loss,
                   device: torch.device) -> float: # Added device param
    """Evaluates the model on the provided data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation set (e.g., validation).
        loss_fn (torch.nn.modules.loss._Loss): The loss function.
        device (torch.device): The device to run the computation on.

    Returns:
        float: The average loss over the evaluation dataset.
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad(): # Turn off gradient computation
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

# --- Optuna Objective Function --- 
def objective(trial: optuna.trial.Trial, 
            X_train_scaled: np.ndarray, y_train: pd.Series,
            X_val_scaled: np.ndarray, y_val: pd.Series,
            input_dim: int,
            device: torch.device) -> float:
    """Objective function for Optuna hyperparameter tuning."""

    # --- 1. Suggest Hyperparameters --- 
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gru_dim = trial.suggest_categorical("gru_dim", [32, 64, 128]) # Smaller range for faster tuning
    # gru_expansion = trial.suggest_float("gru_expansion", 0.5, 1.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    # Add other parameters like dropout if desired
    
    # --- Fixed Parameters for Trial ---
    output_dim = 1
    tuning_epochs = 15 # Fewer epochs per trial
    gru_expansion = 1.0 # Fixed for this example

    # --- 2. Setup Model, Optimizer --- 
    model = GrueyModel(
        input_dim=input_dim, 
        gru_dim=gru_dim, 
        output_dim=output_dim, 
        gru_expansion_factor=gru_expansion
    ).to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- 3. Create DataLoaders for this trial --- 
    train_dataset = TabularDataset(X_train_scaled, y_train)
    val_dataset = TabularDataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nTrial {trial.number}: PARAMS lr={lr:.6f}, gru_dim={gru_dim}, batch_size={batch_size}")

    # --- 4. Training & Validation Loop --- 
    for epoch in range(tuning_epochs):
        model.train()
        epoch_train_loss = 0.0
        for features, targets in train_loader:
            loss = train_step(model, features, targets, loss_fn, optimizer, device)
            epoch_train_loss += loss
        # Optional: print avg train loss 
        # avg_epoch_train_loss = epoch_train_loss / len(train_loader)

        # --- Evaluate on Validation Set --- 
        val_loss = evaluate_model(model, val_loader, loss_fn, device)
        print(f"  Epoch {epoch+1}/{tuning_epochs}, Val Loss: {val_loss:.4f}")

        # --- Optuna Reporting & Pruning --- 
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print("  Trial pruned!")
            raise optuna.exceptions.TrialPruned()

    # --- 5. Return Final Metric --- 
    # Return the final validation loss for Optuna to minimize
    return val_loss

def main():
    """Main function changed to orchestrate Optuna hyperparameter tuning."""
    
    # --- Configuration (Fixed parts and defaults) ---
    TARGET_VARIABLE = "Duration_In_Min"
    BASE_FEATURES_TO_DROP = [
        'Student_IDs', 'Semester', 'Class_Standing', 'Major',
        'Expected_Graduation', 'Course_Name', 'Course_Number',
        'Course_Type', 'Course_Code_by_Thousands', 'Check_Out_Time',
        'Session_Length_Category', 'Check_In_Date', 'Semester_Date',
        'Expected_Graduation_Date',
        'Duration_In_Min', 'Occupancy'
    ]
    # Split ratios: 80% train_val, 20% test. Then 80% train, 20% val from train_val
    TEST_SET_RATIO = 0.2 
    VALIDATION_SET_RATIO = 0.2 # Proportion of train_val to use for validation
    N_TRIALS = 30 # Number of Optuna trials to run
    
    # --- Device Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define constants needed for main execution
    DATA_DIR = os.path.join(project_root, "data", "processed")
    TRAIN_FILE = os.path.join(DATA_DIR, "train_engineered.csv")

    try:
        # --- 1. Load Data --- 
        full_df = load_data(TRAIN_FILE)
        print("\nData loaded successfully.")
        
        # --- 2. Split into Train/Validation/Test --- 
        # First split: Separate Test set
        train_val_df, df_test = train_test_split(
            full_df, test_size=TEST_SET_RATIO, random_state=3 # Use fixed random state
        )
        print(f"Data split: Train/Val={len(train_val_df)} rows, Test={len(df_test)} rows")
        # Second split: Separate Train and Validation sets from train_val_df
        df_train, df_val = train_test_split(
            train_val_df, test_size=VALIDATION_SET_RATIO, random_state=3 # Use same state for consistency
        )
        print(f"Further split: Train={len(df_train)} rows, Validation={len(df_val)} rows")

        # --- 3. Preprocess Train, Validation, and Test Sets Separately --- 
        cols_to_drop_this_run = list(set(BASE_FEATURES_TO_DROP) - {TARGET_VARIABLE})
        
        print("\nPreprocessing training data...")
        X_train, y_train = preprocess_data(df_train.copy(), TARGET_VARIABLE, cols_to_drop_this_run)
        print("\nPreprocessing validation data...")
        X_val, y_val = preprocess_data(df_val.copy(), TARGET_VARIABLE, cols_to_drop_this_run)
        # Preprocess test set as well, but keep it separate - NOT used in objective
        # print("\nPreprocessing test data...")
        # X_test, y_test = preprocess_data(df_test.copy(), TARGET_VARIABLE, cols_to_drop_this_run)

        # --- Align columns (Train -> Val, Train -> Test) --- 
        train_cols = X_train.columns
        print("\nAligning Validation columns...")
        val_cols = set(X_val.columns)
        if set(train_cols) != val_cols:
            missing_in_val = list(set(train_cols) - val_cols)
            for col in missing_in_val: X_val[col] = 0
            extra_in_val = list(val_cols - set(train_cols))
            if extra_in_val: X_val = X_val.drop(columns=extra_in_val)
            X_val = X_val[train_cols]
            print("Aligned Validation columns.")
        # Align Test columns similarly if needed later for final eval
        # ... (Test alignment code) ...

        # --- Convert bools --- 
        print("\nConverting boolean columns...")
        for col in X_train.columns: # Iterate through train columns
            if X_train[col].dtype == 'bool':
                X_train[col] = X_train[col].astype(int)
                if col in X_val.columns: X_val[col] = X_val[col].astype(int)
                # if col in X_test.columns: X_test[col] = X_test[col].astype(int)
        
        # --- 4. Feature Scaling (Fit on Train, Transform Train & Val) --- 
        print("\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        # X_test_scaled = scaler.transform(X_test) # Scale test set if using later
        print("Features scaled.")
        input_dim = X_train_scaled.shape[1] 

        # --- 5. Optuna Study Setup --- 
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        
        print(f"\n--- Starting Optuna Study ({N_TRIALS} trials) ---")
        # Pass preprocessed data to the objective function via lambda or functools.partial
        study.optimize(
            lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, input_dim, device),
            n_trials=N_TRIALS
        )
        print("--- Optuna Study Finished ---")

        # --- 6. Output Results --- 
        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("\n--- Tuning Summary ---")
        print(f"Study statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Number of pruned trials: {len(pruned_trials)}")
        print(f"  Number of complete trials: {len(complete_trials)}")

        print("\nBest trial found:")
        best_trial = study.best_trial
        print(f"  Value (Min Validation Loss): {best_trial.value:.4f}")
        print("  Best Parameters: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

            
        if complete_trials: # Check if any trials completed successfully
            # --- Sort trials by validation loss (best first) ---
            complete_trials.sort(key=lambda t: t.value)

            print("\nTop 4 Trials:")
            num_trials_to_save = min(len(complete_trials), 4)
            for i in range(num_trials_to_save):
                trial = complete_trials[i]
                print(f"  Rank {i+1}: Value (Loss): {trial.value:.4f}, Params: {trial.params}")
                
                # --- Save Top N Parameters --- 
                params_save_dir = os.path.join(project_root, "artifacts", "params", "pytorch")
                os.makedirs(params_save_dir, exist_ok=True)
                # Format rank with zero padding (e.g., 01, 02, 03, 04)
                rank_str = str(i + 1).zfill(2)
                params_filename = f"gruey_{rank_str}_params_{TARGET_VARIABLE}.json"
                params_save_path = os.path.join(params_save_dir, params_filename)
                try:
                    import json 
                    with open(params_save_path, 'w') as f:
                        json.dump(trial.params, f, indent=4)
                    print(f"    Parameters saved to: {params_save_path}")
                except Exception as e:
                    print(f"    Error saving parameters for rank {i+1} to {params_save_path}: {e}")
            # --- End Save --- 
        else:
            print("\nNo trials completed successfully. Unable to determine or save best parameters.")
        
        print("\nNOTE: To run final training with these parameters, manually update the ")
        print("constants (LEARNING_RATE, GRU_DIM, BATCH_SIZE) and run the script in its ")
        print("original single-training mode (or modify this script further).")

    except FileNotFoundError as e:
        print(f"\nError: Data file not found - {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

# Removed placeholder code below main function