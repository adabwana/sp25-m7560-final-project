import pandas as pd
import os
import sys
import torch
from typing import Tuple, List
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
import json
import torch.nn.utils as utils # Import utils for clipping
from sklearn.metrics import r2_score # Import r2_score

# Add project root for imports from other modules like utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add SRC directory to sys.path instead of project_root for direct execution
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

# --- Imports from project structure ---
from python.utils.data_utils import load_data
from python.utils.preprocessing import preprocess_data
# from python.datasets import TabularDataset
# Import the new dataset correctly
from python.datasets.sequence_dataset import SequenceDataset
# Import the new model
from python.models.mtls_architecture import MtlsModel
from torch.utils.data import DataLoader

# --- Training Step Function (Identical to gruey.py) ---
def train_step(model: torch.nn.Module,
               features: torch.Tensor,
               targets: torch.Tensor,
               loss_fn: torch.nn.modules.loss._Loss,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               clip_value: float = 1.0) -> float: # Add clip_value parameter
    model.train()
    features, targets = features.to(device), targets.to(device)
    predictions = model(features)
    loss = loss_fn(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    # --- Add Gradient Clipping ---
    utils.clip_grad_norm_(model.parameters(), clip_value)
    # ---------------------------
    optimizer.step()
    return loss.item()

# --- Evaluation Function (Identical to gruey.py, reports RMSE) ---
def evaluate_model(model: torch.nn.Module,
                   data_loader: DataLoader,
                   loss_fn: torch.nn.modules.loss._Loss,
                   device: torch.device) -> Tuple[float, float]: # Return RMSE and R2
    """Evaluates the model on the provided data loader, calculating RMSE and R2.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation set (e.g., validation).
        loss_fn (torch.nn.modules.loss._Loss): The loss function (used for MSE calculation).
        device (torch.device): The device to run the computation on.

    Returns:
        Tuple[float, float]: A tuple containing the average RMSE and the R2 score.
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features)
            # Assuming targets are (batch_size, 1) and predictions are (batch_size, 1)
            # If targets/predictions have sequence length, adjust loss and R2 calculation
            loss = loss_fn(predictions, targets) # Calculate MSE loss
            total_loss += loss.item()
            # Store targets and predictions for R2 calculation
            # Ensure targets and predictions are appropriately shaped for R2 (e.g., flattened)
            # Assuming targets and predictions are already [batch_size, output_dim] or similar for this non-sequence example
            all_targets.append(targets.cpu()) 
            all_predictions.append(predictions.cpu())

    avg_mse_loss = total_loss / len(data_loader)
    avg_rmse_loss = avg_mse_loss**0.5 # Calculate RMSE

    # Concatenate all targets and predictions
    # Ensure they are correctly shaped before concatenation and R2 calculation
    all_targets_np = torch.cat(all_targets).numpy() 
    all_predictions_np = torch.cat(all_predictions).numpy()

    # Calculate R2 score
    # Handle potential edge cases (e.g., constant predictions) if necessary
    try:
        r2 = r2_score(all_targets_np, all_predictions_np)
    except ValueError: # Handle cases where R2 might be ill-defined (e.g., single sample)
        r2 = float('nan') 

    return avg_rmse_loss, r2

# --- Optuna Objective Function (Adapted for MtlsModel) ---
def objective(trial: optuna.trial.Trial,
            X_train_scaled: np.ndarray, y_train: pd.Series,
            X_val_scaled: np.ndarray, y_val: pd.Series,
            input_dim: int,
            device: torch.device) -> float:
    """Objective function for Optuna hyperparameter tuning for MtlsModel."""

    # --- 1. Define Fixed and Tunable Hyperparameters ---
    # Tunable
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)  # Narrowed and shifted lower
    dropout_rate = trial.suggest_float("dropout_rate", 0.20, 0.40) # Narrowed
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True) # Narrowed and shifted lower
    # lstm_dim = trial.suggest_categorical("lstm_dim", [128, 256]) # Keep fixed for now based on results
    # num_layers = trial.suggest_categorical("num_layers", [1, 2])
    # batch_size = trial.suggest_categorical("batch_size", [16, 32])
    sequence_length = trial.suggest_int("sequence_length", 30, 40)
    lstm_expansion = trial.suggest_float("lstm_expansion", 0.85, 0.95)

    
    # Fixed (based on previous results or decisions)
    lstm_dim = 256 # Fixed based on consistent top performance
    batch_size = 32 # Fixed based on consistent top performance
    num_layers = 1 # Fixed based on consistent top performance
    activation_fn_name = "relu"

    # Store fixed parameters as user attributes
    fixed_params_for_trial = {
        "lstm_dim": lstm_dim,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "activation_fn_name": activation_fn_name
    }
    trial.set_user_attr("fixed_params", fixed_params_for_trial)

    # Combine all parameters for setup and printing (optional but convenient here)
    # Note: trial.params only contains the suggested values at this point
    current_params = {**fixed_params_for_trial, **trial.params}

    # --- Fixed Parameters for Trial Run --- (Separate from hyperparams)
    output_dim = 1
    tuning_epochs = 15

    # --- 2. Setup Model, Optimizer using current_params ---
    model = MtlsModel(
        input_dim=input_dim,
        lstm_dim=current_params["lstm_dim"],
        output_dim=output_dim,
        lstm_expansion_factor=current_params["lstm_expansion"],
        num_layers=current_params["num_layers"],
        dropout_rate=current_params["dropout_rate"],
        activation_fn_name=current_params["activation_fn_name"]
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=current_params["lr"], weight_decay=current_params["weight_decay"])

    # --- 3. Create DataLoaders for this trial using SequenceDataset ---
    try:
        train_dataset = SequenceDataset(X_train_scaled, y_train, current_params["sequence_length"])
        val_dataset = SequenceDataset(X_val_scaled, y_val, current_params["sequence_length"])
    except ValueError as e:
        print(f"Skipping trial due to incompatible sequence_length: {e}")
        return float('inf')

    train_loader = DataLoader(train_dataset, batch_size=current_params["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=current_params["batch_size"], shuffle=False)

    # Updated print statement using combined params dictionary
    print(f"\nTrial {trial.number}: PARAMS lr={current_params['lr']:.6f}, dropout={current_params['dropout_rate']:.2f}, wd={current_params['weight_decay']:.6f}, "
          f"lstm_dim={current_params['lstm_dim']}, layers={current_params['num_layers']}, seq_len={current_params['sequence_length']}, expansion={current_params['lstm_expansion']:.2f}, act={current_params['activation_fn_name']}, "
          f"bs={current_params['batch_size']}, epochs={tuning_epochs}")

    # --- 4. Training & Validation Loop --- (No changes needed here)
    best_val_loss = float('inf') # Initialize best validation loss tracking (RMSE)
    best_r2_at_best_rmse = -float('inf') # Initialize R2 tracking
    for epoch in range(tuning_epochs):
        model.train()
        epoch_train_loss = 0.0
        for features, targets in train_loader:
            loss = train_step(model, features, targets, loss_fn, optimizer, device, clip_value=1.0) # [0.5 1.0] tested, higher is, more likely nan. even 1.0 is behaving better now
            epoch_train_loss += loss

        # --- Evaluate on Validation Set ---
        val_rmse, r2 = evaluate_model(model, val_loader, loss_fn, device)
        print(f"  Epoch {epoch+1}/{tuning_epochs}, Val RMSE: {val_rmse:.4f}, R2: {r2:.4f}")

        # --- Update Best Validation Loss & Corresponding R2 ---
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            best_r2_at_best_rmse = r2 # Store the R2 score when RMSE improves

        # --- Optuna Reporting & Pruning (Using RMSE) ---
        # Report the current epoch's RMSE loss for pruning purposes
        trial.report(val_rmse, epoch)
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
    """Main function to orchestrate Optuna hyperparameter tuning for MtlsModel."""

    # --- Configuration (Using Duration_In_Min as default target) ---
    TARGET_VARIABLE = "Occupancy"
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
    N_TRIALS = 20 # Initial number of Optuna trials

    # --- Device Configuration --- (Identical)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading --- (Identical)
    DATA_DIR = os.path.join(project_root, "data", "processed")
    TRAIN_FILE = os.path.join(DATA_DIR, "train_engineered.csv")

    try:
        # --- 1. Load Data --- (Identical)
        full_df = load_data(TRAIN_FILE)
        print("\nData loaded successfully.")

        # --- 2. Split Data --- (Identical)
        train_val_df, df_test = train_test_split(full_df, test_size=TEST_SET_RATIO, random_state=3)
        print(f"Data split: Train/Val={len(train_val_df)} rows, Test={len(df_test)} rows")
        df_train, df_val = train_test_split(train_val_df, test_size=VALIDATION_SET_RATIO, random_state=3)
        print(f"Further split: Train={len(df_train)} rows, Validation={len(df_val)} rows")

        # --- 3. Preprocessing --- (Identical)
        cols_to_drop_this_run = list(set(BASE_FEATURES_TO_DROP) - {TARGET_VARIABLE})
        print("\nPreprocessing training data...")
        X_train, y_train = preprocess_data(df_train.copy(), TARGET_VARIABLE, cols_to_drop_this_run)
        print("\nPreprocessing validation data...")
        X_val, y_val = preprocess_data(df_val.copy(), TARGET_VARIABLE, cols_to_drop_this_run)

        # --- Align Columns --- (Identical)
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

        # --- Convert Bools --- (Identical)
        print("\nConverting boolean columns...")
        for col in X_train.columns:
            if X_train[col].dtype == 'bool':
                X_train[col] = X_train[col].astype(int)
                if col in X_val.columns: X_val[col] = X_val[col].astype(int)
                # Convert in test set too if needed
                # if 'X_test' in locals() and col in X_test.columns:
                #     X_test[col] = X_test[col].astype(int)

        # --- 4. Scaling --- (Identical)
        print("\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        print("Features scaled.")
        input_dim = X_train_scaled.shape[1]

        # --- 5. Optuna Study --- (Identical logic)
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        print(f"\n--- Starting Optuna Study ({N_TRIALS} trials) for {TARGET_VARIABLE} using MtlsModel ---")
        study.optimize(
            lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, input_dim, device),
            n_trials=N_TRIALS
        )
        print("--- Optuna Study Finished ---")

        # --- 6. Output Results (Adjusted filenames) ---
        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("\n--- Tuning Summary ---")
        print(f"Study statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Number of pruned trials: {len(pruned_trials)}")
        print(f"  Number of complete trials: {len(complete_trials)}")

        print("\nBest trial found:")
        try:
            best_trial = study.best_trial
            # Retrieve tuned and fixed params for the best trial
            tuned_params_best = best_trial.params
            fixed_params_best = best_trial.user_attrs.get("fixed_params", {})
            all_params_best = {**fixed_params_best, **tuned_params_best}
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

        if complete_trials:
            complete_trials.sort(key=lambda t: t.value)
            print("\nTop 10 Trials (Saving Top 4 Params):")
            num_trials_to_print = min(len(complete_trials), 10)
            num_trials_to_save = min(len(complete_trials), 4)

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
                    params_save_dir = os.path.join(project_root, "artifacts", "params", "pytorch")
                    os.makedirs(params_save_dir, exist_ok=True)
                    rank_str = str(i + 1).zfill(2)
                    # Changed filename prefix to mtls_seq_
                    params_filename = f"mtls_seq_{rank_str}_params_{TARGET_VARIABLE}.json"
                    params_save_path = os.path.join(params_save_dir, params_filename)
                    try:
                        params_to_save = all_params
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

        print("\nNOTE: To run final training with these parameters, manually update the ")
        print("constants/arguments for your final training script.")

    except FileNotFoundError as e:
        print(f"\nError: Data file not found - {e}")
        print(f"Attempted path: {TRAIN_FILE}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
