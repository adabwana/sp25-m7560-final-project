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

# --- Final Training Function --- 
# (Identical to the one added in gruey.py)
def train_final_model(model: torch.nn.Module,
                      train_val_loader: DataLoader,
                      loss_fn: torch.nn.modules.loss._Loss, 
                      optimizer: torch.optim.Optimizer,
                      final_epochs: int,
                      device: torch.device):
    """Trains a model on the combined train+val data for final evaluation."""
    print(f"  Retraining final model for {final_epochs} epochs...")
    model.train()
    for epoch in range(final_epochs):
        epoch_loss = 0.0
        for features, targets in train_val_loader:
            loss = train_step(model, features, targets, loss_fn, optimizer, device)
            epoch_loss += loss
        # Optional: print progress
    print("  Retraining complete.")
    return model

# --- Evaluate on Test Set Function --- 
# (Identical to the one added in gruey.py)
def evaluate_on_test(model: torch.nn.Module, 
                     test_loader: DataLoader, 
                     y_test: pd.Series,
                     device: torch.device) -> Dict[str, float]:
    """Evaluates a trained model on the held-out test set."""
    print("  Evaluating on test set...")
    model.eval()
    y_pred_np = predict_on_loader(model, test_loader, device)
    y_true_np = y_test.values
    if y_true_np.shape[0] != y_pred_np.shape[0]:
         raise ValueError(f"Mismatch samples: y_true {y_true_np.shape[0]} vs y_pred {y_pred_np.shape[0]}")
    if y_true_np.ndim > 1 and y_pred_np.ndim == 1:
        y_true_np = y_true_np.squeeze()
    metrics = calculate_regression_metrics(y_true_np, y_pred_np)
    return metrics

def main():
    """Main function changed to orchestrate Optuna tuning AND final evaluation for Transformer."""
    
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
    FINAL_TRAINING_EPOCHS = 20 # Epochs to train final models
    N_TOP_TRIALS_TO_EVAL = 4 # Evaluate top N trials
    
    # --- Device Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Paths ---
    DATA_DIR = os.path.join(project_root, "data", "processed")
    TRAIN_FILE = os.path.join(DATA_DIR, "train_engineered.csv")
    PARAMS_SAVE_DIR = os.path.join(project_root, "artifacts", "params", "pytorch")
    os.makedirs(PARAMS_SAVE_DIR, exist_ok=True)
    # PARAMS_FILENAME defined later based on rank

    try:
        # --- 1. Load Data & 2. Split Data (Identical to gruey.py) ---
        full_df = load_data(TRAIN_FILE)
        train_val_df, df_test = train_test_split(full_df, test_size=TEST_SET_RATIO, random_state=3)
        df_train, df_val = train_test_split(train_val_df, test_size=VALIDATION_SET_RATIO, random_state=3)
        print(f"Data split: Train={len(df_train)}, Validation={len(df_val)}, Test={len(df_test)}")

        # --- 3. Preprocess Train & Validation (for Optuna) --- 
        cols_to_drop_this_run = list(set(BASE_FEATURES_TO_DROP) - {TARGET_VARIABLE})
        print("\nPreprocessing Train data (for Optuna)...")
        X_train, y_train = preprocess_data(df_train.copy(), TARGET_VARIABLE, cols_to_drop_this_run)
        print("Preprocessing Validation data (for Optuna)...")
        X_val, y_val = preprocess_data(df_val.copy(), TARGET_VARIABLE, cols_to_drop_this_run)
        
        # Align Validation columns
        train_cols = X_train.columns
        val_cols = set(X_val.columns)
        if set(train_cols) != val_cols:
            print("Aligning Validation columns...")
            missing_in_val = list(set(train_cols) - val_cols)
            for col in missing_in_val: X_val[col] = 0
            extra_in_val = list(val_cols - set(train_cols))
            if extra_in_val: X_val = X_val.drop(columns=extra_in_val)
            X_val = X_val[train_cols]

        # Convert bools (Train & Val)
        for col in X_train.columns: 
            if X_train[col].dtype == 'bool':
                X_train[col] = X_train[col].astype(int)
                if col in X_val.columns: X_val[col] = X_val[col].astype(int)
        
        # --- 4. Feature Scaling (for Optuna) --- 
        print("\nScaling features (for Optuna)...")
        scaler_optuna = StandardScaler()
        X_train_scaled = scaler_optuna.fit_transform(X_train)
        X_val_scaled = scaler_optuna.transform(X_val)
        input_dim = X_train_scaled.shape[1] 

        # --- 5. Optuna Study --- 
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        print(f"\n--- Starting Optuna Study for Transformer ({N_TRIALS} trials) ---")
        study.optimize(
            lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, input_dim, device),
            n_trials=N_TRIALS
        )
        print("--- Optuna Study Finished ---")

        # --- 6. Process Results & Retrain/Evaluate Top N --- 
        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("\n--- Tuning Summary ---")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Number of pruned trials: {len(pruned_trials)}")
        print(f"Number of complete trials: {len(complete_trials)}")

        if complete_trials: 
            complete_trials.sort(key=lambda t: t.value)
            num_trials_to_eval = min(len(complete_trials), N_TOP_TRIALS_TO_EVAL)
            print(f"\n--- Retraining and Evaluating Top {num_trials_to_eval} Transformer Trials on Test Set ---")

            # --- Preprocess Combined Train+Val Data and Test Data (for final eval) ---
            # (Identical logic to gruey.py)
            print("\nPreprocessing combined Train+Val data...")
            X_train_val, y_train_val = preprocess_data(train_val_df.copy(), TARGET_VARIABLE, cols_to_drop_this_run)
            print("Preprocessing Test data...")
            X_test, y_test = preprocess_data(df_test.copy(), TARGET_VARIABLE, cols_to_drop_this_run)
            train_val_cols = X_train_val.columns
            test_cols = set(X_test.columns)
            if set(train_val_cols) != test_cols:
                print("Aligning Test columns...")
                missing_in_test = list(set(train_val_cols) - test_cols)
                for col in missing_in_test: X_test[col] = 0
                extra_in_test = list(test_cols - set(train_val_cols))
                if extra_in_test: X_test = X_test.drop(columns=extra_in_test)
                X_test = X_test[train_val_cols]
            for col in X_train_val.columns: 
                if X_train_val[col].dtype == 'bool':
                    X_train_val[col] = X_train_val[col].astype(int)
                    if col in X_test.columns: X_test[col] = X_test[col].astype(int)
            print("Scaling features for final training/evaluation...")
            scaler_final = StandardScaler()
            X_train_val_scaled = scaler_final.fit_transform(X_train_val)
            X_test_scaled = scaler_final.transform(X_test)
            final_input_dim = X_train_val_scaled.shape[1]
            train_val_dataset = TabularDataset(X_train_val_scaled, y_train_val)
            test_dataset = TabularDataset(X_test_scaled, y_test)
            final_batch_size = 64 # Example fixed size
            train_val_loader = DataLoader(train_val_dataset, batch_size=final_batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=final_batch_size, shuffle=False)
            # --- End Data Prep for Final Eval ---

            for i in range(num_trials_to_eval):
                trial = complete_trials[i]
                best_params = trial.params
                print(f"\nRank {i+1}: Value (Val Loss): {trial.value:.4f}, Params: {best_params}")
                
                # --- Instantiate Model with Best Params ---
                # Ensure d_model % nhead == 0 (should be guaranteed if trial completed)
                d_model = best_params['d_model']
                nhead = best_params['nhead']
                if d_model % nhead != 0:
                    print(f"    Warning: Inconsistent state - d_model {d_model} not divisible by nhead {nhead}. Skipping retraining.")
                    continue
                
                model = TabularTransformerModel(
                    input_dim=final_input_dim, 
                    d_model=d_model,
                    nhead=nhead,
                    d_hid=best_params['d_hid'],
                    nlayers=best_params['nlayers'],
                    output_dim=1,
                    dropout=best_params['dropout']
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
                loss_fn = nn.MSELoss()
                
                # --- Retrain on Combined Data --- 
                retrained_model = train_final_model(model, train_val_loader, loss_fn, optimizer, FINAL_TRAINING_EPOCHS, device)

                # --- Evaluate on Test Set --- 
                test_metrics = evaluate_on_test(retrained_model, test_loader, y_test, device)
                # metrics printed within evaluate_on_test
                
                # --- Save Parameters --- 
                params_filename = f"transformer_{str(i+1).zfill(2)}_params_{TARGET_VARIABLE}.json"
                params_save_path = os.path.join(PARAMS_SAVE_DIR, params_filename)
                try:
                    with open(params_save_path, 'w') as f:
                        json.dump(best_params, f, indent=4)
                    print(f"  Parameters saved to: {params_save_path}")
                except Exception as e:
                    print(f"  Error saving parameters for rank {i+1} to {params_save_path}: {e}")
        else:
            print("\nNo trials completed successfully. Cannot evaluate.")
        
        # Remove note about manual update
        # print("\nNOTE: To run final training ...")

    except FileNotFoundError as e:
        print(f"\nError: Data file not found - {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
