import pandas as pd
import os
import sys
import torch 
from typing import Tuple, List
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna # Import Optuna
import torch.optim as optim # Import optim

# Add project root for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

# --- Imports from project structure ---
from python.utils.data_utils import load_data
from python.utils.preprocessing import preprocess_data
from python.datasets import TabularDataset
from python.models.transformer_architecture import TabularTransformerModel # Import Transformer
# No final evaluation needed in tuning script
# from python.evaluation.evaluation import evaluate_saved_model

# --- Training Step Function (Same as gruey.py) ---
def train_step(model: torch.nn.Module, 
               features: torch.Tensor, 
               targets: torch.Tensor, 
               loss_fn: torch.nn.modules.loss._Loss, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> float:
    """Performs a single training step."""
    model.train()
    features, targets = features.to(device), targets.to(device)
    predictions = model(features)
    loss = loss_fn(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# --- Evaluation Function (Same as gruey.py) ---
def evaluate_model(model: torch.nn.Module,
                   data_loader: DataLoader,
                   loss_fn: torch.nn.modules.loss._Loss,
                   device: torch.device) -> float:
    """Evaluates the model on validation data during training."""
    model.eval() 
    total_loss = 0.0
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

# --- Optuna Objective Function for Transformer --- 
def objective(trial: optuna.trial.Trial, 
            X_train_scaled: np.ndarray, y_train: pd.Series,
            X_val_scaled: np.ndarray, y_val: pd.Series,
            input_dim: int,
            device: torch.device) -> float:
    """Objective function for Optuna hyperparameter tuning for Transformer."""

    # --- 1. Suggest Hyperparameters --- 
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    d_model = trial.suggest_categorical("d_model", [64, 128, 256]) # Must be divisible by nhead
    nhead = trial.suggest_categorical("nhead", [4, 8]) # Ensure d_model % nhead == 0 later
    d_hid = trial.suggest_categorical("d_hid", [128, 256, 512]) # Hidden dim in feedforward
    nlayers = trial.suggest_int("nlayers", 2, 6) # Number of encoder layers
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Ensure d_model is divisible by nhead
    if d_model % nhead != 0:
        # Option 1: Prune trial (simpler)
        raise optuna.exceptions.TrialPruned(f"d_model {d_model} not divisible by nhead {nhead}")
        # Option 2: Adjust nhead (more complex, might bias search)
        # compatible_nheads = [h for h in [1, 2, 4, 8, 16] if d_model % h == 0]
        # nhead = trial.suggest_categorical("nhead_adjusted", compatible_nheads)
        # Or just pick the closest valid one
        # closest_valid_nhead = min([h for h in [4, 8] if d_model % h == 0], key=lambda h: abs(h - nhead))
        # nhead = closest_valid_nhead

    # --- Fixed Parameters for Trial ---
    output_dim = 1
    tuning_epochs = 15 # Fewer epochs per trial

    # --- 2. Setup Model, Optimizer --- 
    model = TabularTransformerModel(
        input_dim=input_dim, 
        d_model=d_model,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        output_dim=output_dim,
        dropout=dropout
    ).to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Use explicit torch.optim

    # --- 3. Create DataLoaders for this trial --- 
    train_dataset = TabularDataset(X_train_scaled, y_train)
    val_dataset = TabularDataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nTrial {trial.number}: PARAMS lr={lr:.6f}, d_model={d_model}, nhead={nhead}, d_hid={d_hid}, nlayers={nlayers}, dropout={dropout:.2f}, batch_size={batch_size}")

    # --- 4. Training & Validation Loop --- 
    for epoch in range(tuning_epochs):
        model.train()
        epoch_train_loss = 0.0
        for features, targets in train_loader:
            loss = train_step(model, features, targets, loss_fn, optimizer, device)
            epoch_train_loss += loss

        # --- Evaluate on Validation Set --- 
        val_loss = evaluate_model(model, val_loader, loss_fn, device)
        print(f"  Epoch {epoch+1}/{tuning_epochs}, Val Loss: {val_loss:.4f}")

        # --- Optuna Reporting & Pruning --- 
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print("  Trial pruned!")
            raise optuna.exceptions.TrialPruned()

    # --- 5. Return Final Metric --- 
    return val_loss

def main():
    """Main function changed to orchestrate Optuna hyperparameter tuning for Transformer."""
    
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
    TEST_SET_RATIO = 0.2 
    VALIDATION_SET_RATIO = 0.2 
    N_TRIALS = 30 # Number of Optuna trials
    
    # --- Device Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Paths ---
    DATA_DIR = os.path.join(project_root, "data", "processed")
    TRAIN_FILE = os.path.join(DATA_DIR, "train_engineered.csv")
    PARAMS_SAVE_DIR = os.path.join(project_root, "artifacts", "params", "pytorch")
    os.makedirs(PARAMS_SAVE_DIR, exist_ok=True)
    PARAMS_FILENAME = f"best_transformer_params_{TARGET_VARIABLE}.json"
    PARAMS_SAVE_PATH = os.path.join(PARAMS_SAVE_DIR, PARAMS_FILENAME)

    try:
        # --- 1. Load Data --- 
        full_df = load_data(TRAIN_FILE)
        print("\nData loaded successfully.")
        
        # --- 2. Split into Train/Validation/Test --- 
        train_val_df, df_test = train_test_split(full_df, test_size=TEST_SET_RATIO, random_state=42)
        df_train, df_val = train_test_split(train_val_df, test_size=VALIDATION_SET_RATIO, random_state=42)
        print(f"Data split: Train={len(df_train)}, Validation={len(df_val)}, Test={len(df_test)}")

        # --- 3. Preprocess Train, Validation Sets --- 
        cols_to_drop_this_run = list(set(BASE_FEATURES_TO_DROP) - {TARGET_VARIABLE})
        print("\nPreprocessing training data...")
        X_train, y_train = preprocess_data(df_train.copy(), TARGET_VARIABLE, cols_to_drop_this_run)
        print("\nPreprocessing validation data...")
        X_val, y_val = preprocess_data(df_val.copy(), TARGET_VARIABLE, cols_to_drop_this_run)

        # --- Align columns (Train -> Val) --- 
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

        # --- Convert bools --- 
        print("\nConverting boolean columns...")
        for col in X_train.columns:
            if X_train[col].dtype == 'bool':
                X_train[col] = X_train[col].astype(int)
                if col in X_val.columns: X_val[col] = X_val[col].astype(int)
        
        # --- 4. Feature Scaling --- 
        print("\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        print("Features scaled.")
        input_dim = X_train_scaled.shape[1] 

        # --- 5. Optuna Study Setup --- 
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        
        print(f"\n--- Starting Optuna Study for Transformer ({N_TRIALS} trials) ---")
        study.optimize(
            lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, input_dim, device),
            n_trials=N_TRIALS
        )
        print("--- Optuna Study Finished ---")

        # --- 6. Output and Save Results --- 
        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("\n--- Tuning Summary ---")
        print(f"Study statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Number of pruned trials: {len(pruned_trials)}")
        print(f"  Number of complete trials: {len(complete_trials)}")

        if complete_trials: # Check if any trials completed successfully
            print("\nBest trial found:")
            best_trial = study.best_trial
            print(f"  Value (Min Validation Loss): {best_trial.value:.4f}")
            print("  Best Parameters: ")
            best_params = best_trial.params
            for key, value in best_params.items():
                print(f"    {key}: {value}")

            # --- Sort trials by validation loss (best first) ---
            complete_trials.sort(key=lambda t: t.value)

            print("\nTop 4 Trials:")
            num_trials_to_save = min(len(complete_trials), 4)
            for i in range(num_trials_to_save):
                trial = complete_trials[i]
                print(f"  Rank {i+1}: Value (Loss): {trial.value:.4f}, Params: {trial.params}")
                
                # --- Save Top N Parameters --- 
                # PARAMS_SAVE_PATH is defined earlier
                # Format rank with zero padding
                rank_str = str(i + 1).zfill(2)
                # Construct filename using PARAMS_SAVE_DIR and a new format
                params_filename = f"transformer_{rank_str}_params_{TARGET_VARIABLE}.json"
                params_save_path_ranked = os.path.join(PARAMS_SAVE_DIR, params_filename)
                try:
                    import json 
                    with open(params_save_path_ranked, 'w') as f:
                        json.dump(trial.params, f, indent=4)
                    print(f"    Parameters saved to: {params_save_path_ranked}")
                except Exception as e:
                    print(f"    Error saving parameters for rank {i+1} to {params_save_path_ranked}: {e}")
            # --- End Save --- 
        else:
            print("\nNo trials completed successfully. Unable to determine or save best parameters.")
            
        print("\nNOTE: To run final training, manually update constants or create a script ")
        print("that loads these parameters and trains on the full train+validation set.")

    except FileNotFoundError as e:
        print(f"\nError: Data file not found - {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
