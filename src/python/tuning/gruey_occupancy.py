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
import json # Import json for saving params
from sklearn.metrics import r2_score # Import r2_score

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
from python.datasets.tabular_dataset import TabularDataset
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
                   device: torch.device) -> Tuple[float, float]: # Return both RMSE and R2
    """Evaluates the model on the provided data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation set (e.g., validation).
        loss_fn (torch.nn.modules.loss._Loss): The loss function (still used for calculation).
        device (torch.device): The device to run the computation on.

    Returns:
        Tuple[float, float]: A tuple containing the average RMSE and the R2 score.
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    with torch.no_grad(): # Turn off gradient computation
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features)
            loss = loss_fn(predictions, targets) # Calculate MSE loss
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

# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial,
            X_train_scaled: np.ndarray, y_train: pd.Series,
            X_val_scaled: np.ndarray, y_val: pd.Series,
            input_dim: int,
            device: torch.device) -> float:
    """Objective function for Optuna hyperparameter tuning."""

    # --- 1. Suggest Hyperparameters (Ordered roughly by tuning frequency) ---
    # Tunable
    lr = trial.suggest_float("lr", 1.2e-3, 5e-3, log=True) # Slightly shifted lower bound
    dropout_rate = trial.suggest_float("dropout_rate", 0.25, 0.42) # Kept range
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-6, log=True) # Narrowed upper bound
    # gru_dim = trial.suggest_categorical("gru_dim", [128, 256, 512]) # Fixed to 512
    num_layers = trial.suggest_int("num_layers", 2, 3) # Fixed to 2
    batch_size = trial.suggest_categorical("batch_size", [64, 128]) # Kept [64, 128]
    gru_expansion = trial.suggest_float("gru_expansion", 0.6, 1.4) # Tightened range
    # activation_fn_name = trial.suggest_categorical("activation_fn", ["relu", "tanh", "gelu"]) # Added activation tuning

    # Fixed (based on previous results or decisions)
    gru_dim = 512 # Fixed based on Occupancy results
    # num_layers = 2 # Now tunable
    activation_fn_name = "relu" # Fixed based on results

    # Store fixed parameters as user attributes
    fixed_params_for_trial = {
        "gru_dim": gru_dim,
        # "num_layers": num_layers, # No longer fixed
        "activation_fn_name": activation_fn_name
    }
    trial.set_user_attr("fixed_params", fixed_params_for_trial)

    # Combine all parameters for setup and printing
    current_params = {**fixed_params_for_trial, **trial.params}


    # --- Fixed Parameters for Trial Run --- (Separate from hyperparams)
    output_dim = 1
    tuning_epochs = 20 # Fixed number of epochs for tuning trials

    # --- 2. Setup Model, Optimizer using current_params ---
    model = GrueyModel(
        input_dim=input_dim,
        gru_dim=current_params["gru_dim"],
        output_dim=output_dim,
        gru_expansion_factor=current_params["gru_expansion"],
        num_layers=current_params["num_layers"],
        dropout_rate=current_params["dropout_rate"],
        activation_fn_name=current_params["activation_fn_name"] # Pass activation name
    ).to(device)

    loss_fn = nn.MSELoss()
    # Use suggested weight_decay in the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=current_params["lr"], weight_decay=current_params["weight_decay"])

    # --- 3. Create DataLoaders for this trial ---
    train_dataset = TabularDataset(X_train_scaled, y_train)
    val_dataset = TabularDataset(X_val_scaled, y_val)
    # Use batch_size from current_params
    train_loader = DataLoader(train_dataset, batch_size=current_params["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=current_params["batch_size"], shuffle=False)

    # Updated print statement using combined params dictionary
    print(f"\nTrial {trial.number}: PARAMS lr={current_params['lr']:.6f}, dropout={current_params['dropout_rate']:.2f}, wd={current_params['weight_decay']:.6f}, "
          f"gru_dim={current_params['gru_dim']}, layers={current_params['num_layers']}, expansion={current_params['gru_expansion']:.2f}, act={current_params['activation_fn_name']}, "
          f"bs={current_params['batch_size']}, epochs={tuning_epochs}")

    # --- 4. Training & Validation Loop ---
    best_val_loss = float('inf') # Initialize best validation loss tracking (RMSE)
    best_r2_at_best_rmse = -float('inf') # Initialize R2 tracking
    for epoch in range(tuning_epochs): # Use fixed tuning_epochs
        model.train()
        epoch_train_loss = 0.0
        for features, targets in train_loader:
            loss = train_step(model, features, targets, loss_fn, optimizer, device)
            epoch_train_loss += loss
        # Optional: print avg train loss
        # avg_epoch_train_loss = epoch_train_loss / len(train_loader)

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
    """Main function changed to orchestrate Optuna hyperparameter tuning."""

    # --- Configuration (Fixed parts and defaults) ---
    TARGET_VARIABLE = "Occupancy"
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
    N_TRIALS = 50 # Number of Optuna trials to run (already updated by user)

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
        try: # Add try-except in case no trials complete
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


        if complete_trials: # Check if any trials completed successfully
            # --- Sort trials by validation loss (best first) ---
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
                    params_filename = f"gruey_{rank_str}_{TARGET_VARIABLE}_params.json"
                    params_save_path = os.path.join(params_save_dir, params_filename)
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

        print("\nNOTE: To run final training with these parameters, manually update the ")
        print("constants/arguments for your final training script.")


    except FileNotFoundError as e:
        print(f"\nError: Data file not found - {e}")
        print(f"Attempted path: {TRAIN_FILE}") # Print path for debugging
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

# Ensure no old placeholder code remains 