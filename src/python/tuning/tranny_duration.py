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
import json # Import json for saving params
from sklearn.metrics import r2_score # Import r2_score

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
from python.datasets.tabular_dataset import TabularDataset
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
                   device: torch.device) -> Tuple[float, float]: # Return both RMSE and R2
    """Evaluates the model on validation data during training. Returns RMSE and R2 score."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            # Store targets and predictions for R2 calculation
            all_targets.append(targets.cpu())
            all_predictions.append(predictions.cpu())

    avg_mse_loss = total_loss / len(data_loader)
    avg_rmse_loss = avg_mse_loss**0.5 # Calculate RMSE from average MSE

    # Concatenate all targets and predictions
    all_targets_np = torch.cat(all_targets).numpy()
    all_predictions_np = torch.cat(all_predictions).numpy()

    # Calculate R2 score
    r2 = r2_score(all_targets_np, all_predictions_np)

    return avg_rmse_loss, r2 # Return both RMSE and R2

# --- Optuna Objective Function for Transformer ---
def objective(trial: optuna.trial.Trial,
            X_train_scaled: np.ndarray, y_train: pd.Series,
            X_val_scaled: np.ndarray, y_val: pd.Series,
            input_dim: int,
            device: torch.device) -> float:
    """Objective function for Optuna hyperparameter tuning for Transformer."""

    # --- 1. Suggest Tunable Hyperparameters ---
    # Tunable (Refined based on previous top results)
    lr = trial.suggest_float("lr", 1e-3, 7e-3, log=True) # Narrowed range
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    # d_model = trial.suggest_categorical("d_model", [64, 128, 256]) # Must be divisible by nhead
    dropout = trial.suggest_float("dropout", 0.05, 0.20) # Narrowed range
    nlayers = trial.suggest_int("nlayers", 2, 3) # Narrowed range
    # d_hid = trial.suggest_categorical("d_hid", [128]) # Fixed below
    nhead = trial.suggest_categorical("nhead", [4, 8]) # Ensure d_model % nhead == 0 later

    # --- Fixed Parameters for Trial Run --- (Defined after tunable suggests)
    fixed_params_for_trial = {
        "d_hid": 128,
        "batch_size": 64,
        "d_model": 128
    }
    trial.set_user_attr("fixed_params", fixed_params_for_trial)

    # Combine fixed and tunable parameters
    current_params = {**fixed_params_for_trial, **trial.params}

    # --- Fixed Parameters for Trial Run --- (Separate from hyperparams)
    output_dim = 1
    tuning_epochs = 20 # Fixed number of epochs for tuning trials

    # d_model is fixed to 128, nhead is 4 or 8. Divisibility check is guaranteed.
    # if current_params["d_model"] % current_params["nhead"] != 0:
    #     # This should not happen with current fixed/suggested values
    #     raise optuna.exceptions.TrialPruned(f"d_model {current_params['d_model']} not divisible by nhead {current_params['nhead']}")

    # --- 2. Setup Model, Optimizer using current_params ---
    model = TabularTransformerModel(
        input_dim=input_dim,
        d_model=current_params["d_model"],
        nhead=current_params["nhead"],
        d_hid=current_params["d_hid"],
        nlayers=current_params["nlayers"],
        output_dim=output_dim,
        dropout=current_params["dropout"]
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=current_params["lr"], weight_decay=current_params["weight_decay"]) # Add weight decay

    # --- 3. Create DataLoaders for this trial using current_params ---
    train_dataset = TabularDataset(X_train_scaled, y_train)
    val_dataset = TabularDataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=current_params["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=current_params["batch_size"], shuffle=False)

    # Updated print statement using combined params dictionary
    print(f"\nTrial {trial.number}: PARAMS lr={current_params['lr']:.6f}, wd={current_params['weight_decay']:.6f}, d_model={current_params['d_model']}, nhead={current_params['nhead']}, "
          f"d_hid={current_params['d_hid']}, nlayers={current_params['nlayers']}, dropout={current_params['dropout']:.2f}, "
          f"batch_size={current_params['batch_size']}, epochs={tuning_epochs}")

    # --- 4. Training & Validation Loop ---
    best_val_loss = float('inf') # Track best validation RMSE for this trial
    best_r2_at_best_rmse = -float('inf') # Initialize R2 tracking
    for epoch in range(tuning_epochs): # Use fixed tuning_epochs from current_params
        model.train()
        epoch_train_loss = 0.0
        for features, targets in train_loader:
            loss = train_step(model, features, targets, loss_fn, optimizer, device)
            epoch_train_loss += loss

        # --- Evaluate on Validation Set ---
        val_rmse, r2 = evaluate_model(model, val_loader, loss_fn, device) # Returns RMSE and R2
        print(f"  Epoch {epoch+1}/{tuning_epochs}, Val RMSE: {val_rmse:.4f}, R2: {r2:.4f}")

        # --- Update Best Validation Loss for this trial ---
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            best_r2_at_best_rmse = r2 # Store the R2 score when RMSE improves

        # --- Optuna Reporting & Pruning ---
        trial.report(val_rmse, epoch) # Report current epoch's RMSE val_loss for pruning
        if trial.should_prune():
            print("  Trial pruned!")
            raise optuna.exceptions.TrialPruned()

    # --- Store Best R2 Score as User Attribute --- 
    trial.set_user_attr("best_r2_score", best_r2_at_best_rmse)

    # --- 5. Return Final Metric (RMSE for Optuna Optimization) ---
    # Return the *best* validation RMSE observed during this trial
    print(f"  Trial {trial.number} finished. Best Val RMSE: {best_val_loss:.4f}, Best R2: {best_r2_at_best_rmse:.4f}")
    return best_val_loss

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
    # Base filename parts, rank will be inserted later
    PARAMS_FILENAME_BASE = f"transformer"
    PARAMS_FILENAME_SUFFIX = f"{TARGET_VARIABLE}_params.json"

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

        print("\nBest trial found:")
        try: # Add try-except in case no trials complete
            best_trial = study.best_trial
            # Retrieve tuned and fixed params for the best trial
            tuned_params_best = best_trial.params
            fixed_params_best = best_trial.user_attrs.get("fixed_params", {})
            all_params_best = {**fixed_params_best, **tuned_params_best} # Combine
            # Retrieve the best R2 score
            best_r2_score = best_trial.user_attrs.get("best_r2_score", None)
            print(f"  Value (Min Validation RMSE): {best_trial.value:.4f}")
            if best_r2_score is not None:
                print(f"  Corresponding R2 Score: {best_r2_score:.4f}")
            print("  Best Parameters (Combined): ")
            for key, value in all_params_best.items(): # Print combined params
                print(f"    {key}: {value}")
        except ValueError:
             print("  No best trial found (likely no trials completed successfully).")

        if complete_trials: # Check if any trials completed successfully
            # --- Sort trials by validation loss (best first) ---
            complete_trials.sort(key=lambda t: t.value)

            print("\nTop 4 Trials (Saving Params):")
            num_trials_to_print = min(len(complete_trials), 10) # Print top 10
            num_trials_to_save = min(len(complete_trials), 4) # Save top 4

            for i in range(num_trials_to_print):
                trial = complete_trials[i]
                # Retrieve tuned params and fixed params (stored as user_attrs)
                tuned_params = trial.params
                fixed_params = trial.user_attrs.get("fixed_params", {}) # Use .get for safety
                # Combine tuned params from trial with fixed params
                all_params = {**fixed_params, **tuned_params}
                # Retrieve the best R2 score for this trial
                r2_score_trial = trial.user_attrs.get("best_r2_score", None)
                r2_print = f", R2: {r2_score_trial:.4f}" if r2_score_trial is not None else ""
                print(f"  Rank {i+1}: Value (RMSE): {trial.value:.4f}{r2_print}, Params: {all_params}") # Print combined and R2

                if i < num_trials_to_save:
                    rank_str = str(i + 1).zfill(2)
                    # Construct filename using PARAMS_SAVE_DIR and base/suffix
                    params_filename = f"{PARAMS_FILENAME_BASE}_{rank_str}_{PARAMS_FILENAME_SUFFIX}"
                    params_save_path = os.path.join(PARAMS_SAVE_DIR, params_filename)
                    try:
                        params_to_save = all_params # Save the combined dict
                        # Add the validation RMSE and R2 score to the dictionary
                        params_to_save["validation_rmse"] = trial.value
                        if r2_score_trial is not None:
                            params_to_save["validation_r2"] = r2_score_trial
                        with open(params_save_path, 'w') as f:
                            json.dump(params_to_save, f, indent=4)
                        print(f"    Parameters saved to: {params_save_path}")
                    except Exception as e:
                        print(f"    Error saving parameters for rank {i+1} to {params_save_path}: {e}")

        else:
            print("\nNo trials completed successfully. Unable to determine or save best parameters.")

        print("\nNOTE: To run final training, manually update constants or create a script ")
        print("that loads these parameters and trains on the full train+validation set.")

    except FileNotFoundError as e:
        print(f"\nError: Data file not found - {e}")
        print(f"Attempted path: {TRAIN_FILE}") # Print path for debugging
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
