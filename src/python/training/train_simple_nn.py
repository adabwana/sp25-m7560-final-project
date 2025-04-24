import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.python.datasets import TabularDataset
#from python.evaluation.evaluation import evaluate_model
from src.python.models.simple_nn import SimpleNN
from src.python.utils.data_utils import load_data
from src.python.utils.preprocessing import preprocess_data

# Add project root for imports from other modules like utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


def train_fold(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


def main(args):
    print("Starting training for SimpleNN...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse hidden_dims and activations from string to list
    import ast
    if hasattr(args, 'hidden_dims'):
        args.hidden_dims = ast.literal_eval(args.hidden_dims)
    else:
        args.hidden_dims = [args.hidden_dim]
    if hasattr(args, 'activations'):
        args.activations = ast.literal_eval(args.activations)
    else:
        args.activations = ['ReLU'] * len(args.hidden_dims)

    # --- 1. Load Data --- 
    df = load_data(args.train_data_path)
    BASE_FEATURES_TO_DROP = [
        'Student_IDs', 'Semester', 'Class_Standing', 'Major',
        'Expected_Graduation', 'Course_Name', 'Course_Number',
        'Course_Type', 'Course_Code_by_Thousands', 'Check_Out_Time',
        'Session_Length_Category', 'Check_In_Date', 'Semester_Date',
        'Expected_Graduation_Date',
        'Duration_In_Min', 'Occupancy'
    ]
    target = 'Duration_In_Min'  # Change to 'Occupancy' if needed
    drop_cols = list(set(BASE_FEATURES_TO_DROP) - {target})
    X, y = preprocess_data(df, target, drop_cols)
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_dim = X_scaled.shape[1]

    # --- HOLDOUT SPLIT (consistent for all models) ---
    holdout_indices_path = os.path.join('data', 'processed', 'holdout_indices.npy')
    holdout_prop = 0.2
    n_samples = X_scaled.shape[0]
    test_size = int(np.floor(holdout_prop * n_samples))
    np.random.seed(args.seed)  # Use same seed for reproducibility
    if os.path.exists(holdout_indices_path):
        holdout_idx = np.load(holdout_indices_path)
        print(f"Loaded holdout indices from {holdout_indices_path}")
    else:
        holdout_idx = np.random.choice(n_samples, size=test_size, replace=False)
        np.save(holdout_indices_path, holdout_idx)
        print(f"Saved new holdout indices to {holdout_indices_path}")
    mask = np.ones(n_samples, dtype=bool)
    mask[holdout_idx] = False
    X_train, y_train = X_scaled[mask], y[mask].reset_index(drop=True)
    X_holdout, y_holdout = X_scaled[~mask], y[~mask].reset_index(drop=True)

    # --- 2. Cross-Validation Setup (on training set only) --- 
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_results = []

    #print(f"Starting {args.n_splits}-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"--- Fold {fold+1}/{args.n_splits} ---")
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # --- Target Scaling ---
        y_scaler = StandardScaler()
        y_train_fold_scaled = y_scaler.fit_transform(y_train_fold.values.reshape(-1, 1)).astype(np.float32).flatten()
        y_val_fold_scaled = y_scaler.transform(y_val_fold.values.reshape(-1, 1)).astype(np.float32).flatten()
        # Convert scaled arrays back to pandas Series with original indices
        y_train_fold_scaled = pd.Series(y_train_fold_scaled, index=y_train_fold.index)
        y_val_fold_scaled = pd.Series(y_val_fold_scaled, index=y_val_fold.index)

        train_dataset = TabularDataset(X_train_fold, y_train_fold_scaled)
        val_dataset = TabularDataset(X_val_fold, y_val_fold_scaled)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # --- 4. Initialize Model, Loss, Optimizer --- 
        model = SimpleNN(input_dim, args.hidden_dims, 1, args.activations).to(device)
        criterion = nn.MSELoss() 
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        # --- 5. Training Loop --- 
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(args.epochs):
            train_loss = train_fold(model, train_loader, criterion, optimizer, device)
            
            # --- 6. Validation --- 
            model.eval()
            val_loss = 0.0
            preds = []
            targets_list = []
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(device), targets.to(device)
                    outputs = model(features)
                    preds.append(outputs.cpu().numpy())
                    targets_list.append(targets.cpu().numpy())
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            # Inverse-transform predictions and targets for metrics on original scale
            preds_concat = np.concatenate(preds, axis=0)
            targets_concat = np.concatenate(targets_list, axis=0)
            preds_orig = y_scaler.inverse_transform(preds_concat)
            targets_orig = y_scaler.inverse_transform(targets_concat)
            mse_orig = mean_squared_error(targets_orig, preds_orig)
            rmse_orig = np.sqrt(mse_orig)
            mae_orig = mean_absolute_error(targets_orig, preds_orig)
            r2_orig = r2_score(targets_orig, preds_orig)
            val_metrics = {
                'rmse_original': rmse_orig,
                'mae_original': mae_orig,
                'r2_original': r2_orig
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_rmse_orig = rmse_orig
                best_val_mae_orig = mae_orig
                best_val_r2_orig = r2_orig
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping triggered after {args.patience} epochs without improvement in fold {fold+1}. Stopped at epoch {epoch+1}.")
                    break
        fold_results.append({
            'fold': int(fold+1),
            'best_val_loss_scaled': float(best_val_loss),
            'best_val_rmse_original': float(best_val_rmse_orig),
            'best_val_mae_original': float(best_val_mae_orig),
            'best_val_r2_original': float(best_val_r2_orig),
            'metrics': {k: float(v) for k, v in val_metrics.items()}
        })
        print(f"Finished Fold {fold+1}. Best Validation Loss (scaled): {best_val_loss:.4f}, Best Val RMSE (orig): {best_val_rmse_orig:.4f}, MAE: {best_val_mae_orig:.4f}, R2: {best_val_r2_orig:.4f}")

    # --- 8. Aggregate and Save Results --- 
    print("\nCross-validation finished.")
    avg_loss = sum(r['best_val_loss_scaled'] for r in fold_results) / len(fold_results)
    avg_rmse_orig = sum(r['best_val_rmse_original'] for r in fold_results) / len(fold_results)
    avg_mae_orig = sum(r['best_val_mae_original'] for r in fold_results) / len(fold_results)
    avg_r2_orig = sum(r['best_val_r2_original'] for r in fold_results) / len(fold_results)
    print(f"\nAverage Best Validation Loss (scaled) across folds: {avg_loss:.4f}")
    print(f"Average Best Validation RMSE (original) across folds: {avg_rmse_orig:.4f}")
    print(f"Average Best Validation MAE (original) across folds: {avg_mae_orig:.4f}")
    print(f"Average Best Validation R2 (original) across folds: {avg_r2_orig:.4f}\n")
    print("==================== Final Cross-Validation Results ====================")
    print(f"RMSE: {avg_rmse_orig:.4f}")
    print(f"MAE: {avg_mae_orig:.4f}")
    print(f"R^2: {avg_r2_orig:.4f}")
    print("=======================================================================\n")

    # Save CV results (optional)
    results_path = os.path.join(args.output_dir, 'simple_nn_cv_results.json')
    with open(results_path, 'w') as f:
        json.dump(fold_results, f, indent=4)
    print(f"CV results saved to {results_path}")

    # --- 9. Retrain on full training set and evaluate on holdout ---
    print("\nRetraining best model on full training set and evaluating on holdout set...")
    # Use same scaling as before
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).astype(np.float32).flatten()
    # Ensure y_train_scaled is a pandas Series for TabularDataset
    y_train_scaled = pd.Series(y_train_scaled, index=y_train.index)
    train_dataset = TabularDataset(X_train, y_train_scaled)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    model = SimpleNN(input_dim, args.hidden_dims, 1, args.activations).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        train_loss = train_fold(model, train_loader, criterion, optimizer, device)
        if train_loss < best_loss:
            best_loss = train_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping on full train after {args.patience} epochs without improvement. Epoch {epoch+1}.")
                break
    # Evaluate on holdout
    model.eval()
    with torch.no_grad():
        holdout_preds = model(torch.tensor(X_holdout, dtype=torch.float32, device=device)).cpu().numpy()
    holdout_preds_orig = y_scaler.inverse_transform(holdout_preds)
    y_holdout_orig = y_holdout.values.reshape(-1, 1)
    holdout_rmse = np.sqrt(mean_squared_error(y_holdout_orig, holdout_preds_orig))
    holdout_mae = mean_absolute_error(y_holdout_orig, holdout_preds_orig)
    holdout_r2 = r2_score(y_holdout_orig, holdout_preds_orig)
    print("\n==================== Holdout Set Results ====================")
    print(f"Holdout RMSE: {holdout_rmse:.4f}")
    print(f"Holdout MAE: {holdout_mae:.4f}")
    print(f"Holdout R^2: {holdout_r2:.4f}")
    print("============================================================\n")

    print("Training script finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Simple Neural Network with Cross-Validation')

    # Data arguments
    parser.add_argument('--train_data_path', type=str, default='data/processed/train_engineered.csv', help='Path to the processed training data')
    parser.add_argument('--test_data_path', type=str, default='data/processed/test_engineered.csv', help='Path to the processed test data') # May not be needed for CV training script
    
    # Model hyperparameters
    parser.add_argument('--input_dim', type=int, default=None, help='Input dimension for the model (default: inferred from data)')
    parser.add_argument('--hidden_dim', type=int, default=12, help='Hidden layer dimension')
    parser.add_argument('--hidden_dims', type=str, default='[12]', help='List of hidden layer sizes, e.g., "[32,16]" or "[64]"')
    parser.add_argument('--activations', type=str, default='["ReLU"]', help='List of activation functions, e.g., "[\"ReLU\", \"Tanh\"]"')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension (e.g., 1 for regression)') # Adjust based on target
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (0.0 for no dropout, e.g., 0.2)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 weight decay (e.g., 1e-4)')

    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Epochs to wait for improvement before early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Cross-validation arguments
    parser.add_argument('--n_splits', type=int, default=5, help='Number of cross-validation folds')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='artifacts/models/pytorch', help='Directory to save models and results')
    parser.add_argument('--params_dir', type=str, default='artifacts/params/pytorch', help='Directory to save parameters')

    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.params_dir, exist_ok=True)

    # Save parameters used for this run (optional)
    params_path = os.path.join(args.params_dir, 'simple_nn_params.json')
    with open(params_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Parameters saved to {params_path}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main(args)
