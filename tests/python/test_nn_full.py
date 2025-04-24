import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/python')))
from models.simple_nn import SimpleNN
from utils.data_utils import load_data
from datasets import TabularDataset
from utils.preprocessing import preprocess_data

def get_holdout_split(X, y, holdout_indices_path):
    holdout_idx = np.load(holdout_indices_path)
    mask = np.ones(X.shape[0], dtype=bool)
    mask[holdout_idx] = False
    X_non_holdout, y_non_holdout = X[mask], y[mask].reset_index(drop=True)
    X_holdout, y_holdout = X[~mask], y[~mask].reset_index(drop=True)
    return X_non_holdout, y_non_holdout, X_holdout, y_holdout

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, max_epochs=100, patience=5, trial_info=None):
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stopped = False
    for epoch in range(max_epochs):
        model.train()
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        # Validation loss for early stopping
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stopped = True
                break
    if early_stopped and trial_info is not None:
        print(f"  Early stopping triggered for {trial_info} at epoch {epoch+1}.")
    return model

def cross_validate(trial_number, n_layers, hidden_dims, activation, learning_rate, batch_size, X_non_holdout, y_non_holdout, dropout=0.0, weight_decay=0.0):
    start_time = time.time()
    print(f"[START] Trial {trial_number} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses, maes, r2s = [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trial_info = f"Trial {trial_number} | Layers: {n_layers} | Hidden: {hidden_dims} | Act: {activation} | LR: {learning_rate} | Batch: {batch_size} | Dropout: {dropout} | WD: {weight_decay}"
    print(f"Starting {trial_info}")
    for train_idx, val_idx in kf.split(X_non_holdout):
        X_train, X_val = X_non_holdout[train_idx], X_non_holdout[val_idx]
        y_train, y_val = y_non_holdout.iloc[train_idx], y_non_holdout.iloc[val_idx]
        model = SimpleNN(X_train.shape[1], hidden_dims, 1, [activation]*n_layers, dropout=dropout).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_dataset = TabularDataset(X_train, y_train)
        val_dataset = TabularDataset(X_val, y_val)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )
        model = train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, max_epochs=100, patience=5, trial_info=trial_info)
        # Validation
        model.eval()
        preds, targets_list = [], []
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                preds.append(outputs.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
        preds_concat = np.concatenate(preds, axis=0)
        targets_concat = np.concatenate(targets_list, axis=0)
        rmse = np.sqrt(mean_squared_error(targets_concat, preds_concat))
        mae = np.mean(np.abs(targets_concat - preds_concat))
        ss_res = np.sum((targets_concat - preds_concat) ** 2)
        ss_tot = np.sum((targets_concat - np.mean(targets_concat)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
    avg_rmse = float(np.mean(rmses))
    avg_mae = float(np.mean(maes))
    avg_r2 = float(np.mean(r2s))
    print(f"  CV Results: RMSE={avg_rmse:.4f} | MAE={avg_mae:.4f} | R2={avg_r2:.4f}")
    print(f"[END] Trial {trial_number} at {time.strftime('%Y-%m-%d %H:%M:%S')} (Elapsed: {time.time() - start_time:.1f}s)")
    return {
        'trial': trial_number,
        'n_layers': n_layers,
        'hidden_dims': hidden_dims,
        'activation': activation,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'dropout': dropout,
        'weight_decay': weight_decay,
        'cv_rmse': avg_rmse,
        'cv_mae': avg_mae,
        'cv_r2': avg_r2
    }

def retrain_and_evaluate_on_holdout(best_params, X_non_holdout, y_non_holdout, X_holdout, y_holdout):
    start_time = time.time()
    print(f"[START] Holdout retrain for trial {best_params.get('trial', '?')} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNN(X_non_holdout.shape[1], best_params['hidden_dims'], 1, [best_params['activation']] * best_params['n_layers'], dropout=best_params.get('dropout', 0.0)).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params.get('weight_decay', 0.0))
    train_dataset = TabularDataset(X_non_holdout, y_non_holdout)
    val_dataset = TabularDataset(X_holdout, y_holdout)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=best_params['batch_size'], shuffle=False, num_workers=2, pin_memory=True
    )
    print(f"\nRetraining best model on full non-holdout set with params: {best_params}")
    model = train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, max_epochs=100, patience=5, trial_info='Best Model Retrain')
    # Evaluate on holdout
    model.eval()
    preds, targets_list = [], []
    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            preds.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
    preds_concat = np.concatenate(preds, axis=0)
    targets_concat = np.concatenate(targets_list, axis=0)
    rmse = np.sqrt(mean_squared_error(targets_concat, preds_concat))
    mae = np.mean(np.abs(targets_concat - preds_concat))
    ss_res = np.sum((targets_concat - preds_concat) ** 2)
    ss_tot = np.sum((targets_concat - np.mean(targets_concat)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print(f"Holdout Results: RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")
    print(f"[END] Holdout retrain for trial {best_params.get('trial', '?')} at {time.strftime('%Y-%m-%d %H:%M:%S')} (Elapsed: {time.time() - start_time:.1f}s)")
    return rmse, mae, r2

def print_results_table(results):
    sorted_results = sorted(results, key=lambda x: x['cv_rmse'])
    print("\n===== Cross-Validation Results (sorted by RMSE) =====")
    print(f"{'Trial':<6} {'Layers':<6} {'Hidden Dims':<20} {'Activation':<12} {'LR':<8} {'Batch':<6} {'Dropout':<8} {'WD':<8} {'CV RMSE':<10} {'CV MAE':<10} {'CV R2':<10}")
    for r in sorted_results:
        print(f"{r['trial']:<6} {r['n_layers']:<6} {str(r['hidden_dims']):<20} {r['activation']:<12} {r['learning_rate']:<8} {r['batch_size']:<6} {r['dropout']:<8.2f} {r['weight_decay']:<8.2e} {r['cv_rmse']:<10.4f} {r['cv_mae']:<10.4f} {r['cv_r2']:<10.4f}")
    print("====================================================\n")

def main():
    df = load_data('data/processed/train_engineered.parquet')
    BASE_FEATURES_TO_DROP = [
        'Student_IDs', 'Semester', 'Class_Standing', 'Major',
        'Expected_Graduation', 'Course_Name', 'Course_Number',
        'Course_Type', 'Course_Code_by_Thousands', 'Check_Out_Time',
        'Session_Length_Category', 'Check_In_Date', 'Semester_Date',
        'Expected_Graduation_Date',
        'Duration_In_Min', 'Occupancy'
    ]
    target = 'Occupancy'
    drop_cols = list(set(BASE_FEATURES_TO_DROP) - {target})
    X, y = preprocess_data(df, target, drop_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    holdout_indices_path = 'data/processed/holdout_indices.npy'
    X_non_holdout, y_non_holdout, X_holdout, y_holdout = get_holdout_split(X_scaled, y, holdout_indices_path)

    search_space = {
        "n_layers": [1, 2, 3],
        "n_units_l0": [12, 50, 100, 150],
        "n_units_l1": [12, 50, 100],
        "n_units_l2": [12, 50],
        "activation": ["ReLU"],
        "learning_rate": [0.01],
        "batch_size": [2048],
        "dropout": [0, 0.2, 0.3],
        "weight_decay": [0, 1e-5, 1e-4]
    }
    configs = []
    for n_layers in search_space['n_layers']:
        for activation in search_space['activation']:
            for learning_rate in search_space['learning_rate']:
                for batch_size in search_space['batch_size']:
                    for dropout in search_space['dropout']:
                        for weight_decay in search_space['weight_decay']:
                            if n_layers == 1:
                                for n0 in search_space['n_units_l0']:
                                    configs.append((n_layers, [n0], activation, learning_rate, batch_size, dropout, weight_decay))
                            elif n_layers == 2:
                                for n0 in search_space['n_units_l0']:
                                    for n1 in search_space['n_units_l1']:
                                        configs.append((n_layers, [n0, n1], activation, learning_rate, batch_size, dropout, weight_decay))
                            elif n_layers == 3:
                                for n0 in search_space['n_units_l0']:
                                    for n1 in search_space['n_units_l1']:
                                        for n2 in search_space['n_units_l2']:
                                            configs.append((n_layers, [n0, n1, n2], activation, learning_rate, batch_size, dropout, weight_decay))
    results = []
    holdout_results = []
    import pandas as pd
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(cross_validate, i, n_layers, hidden_dims, activation, learning_rate, batch_size, X_non_holdout, y_non_holdout, dropout, weight_decay): (i, n_layers, hidden_dims, activation, learning_rate, batch_size, dropout, weight_decay)
                   for i, (n_layers, hidden_dims, activation, learning_rate, batch_size, dropout, weight_decay) in enumerate(configs)}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
    print_results_table(results)
    # Save all cross-validation results to CSV
    df_cv = pd.DataFrame(results)
    df_cv = df_cv.sort_values('cv_rmse')
    cv_results_path = 'artifacts/models/pytorch/nn_cv_occupancy.csv'
    os.makedirs(os.path.dirname(cv_results_path), exist_ok=True)
    df_cv.to_csv(cv_results_path, index=False)
    print(f"All cross-validation results saved to {cv_results_path}\n")
    # Holdout evaluation for all configs
    for res in results:
        holdout_rmse, holdout_mae, holdout_r2 = retrain_and_evaluate_on_holdout(res, X_non_holdout, y_non_holdout, X_holdout, y_holdout)
        holdout_results.append({
            **res,
            'holdout_rmse': holdout_rmse,
            'holdout_mae': holdout_mae,
            'holdout_r2': holdout_r2
        })
    # Save all holdout results to CSV
    df_holdout = pd.DataFrame(holdout_results)
    df_holdout = df_holdout.sort_values('holdout_rmse')
    holdout_results_path = 'artifacts/models/pytorch/nn_holdout_occupancy.csv'
    df_holdout.to_csv(holdout_results_path, index=False)
    print(f"All holdout results saved to {holdout_results_path}\n")
    # Print top 10 for each
    print("Top 10 Cross-Validation Results (sorted by RMSE):")
    print(df_cv.head(10).to_string(index=False))
    print("\nTop 10 Holdout Results (sorted by RMSE):")
    print(df_holdout.head(10).to_string(index=False))
    best = min(results, key=lambda x: x['cv_rmse'])
    print('\nBest trial:')
    print(best)
    # Optionally, print best holdout as well
    best_holdout = min(holdout_results, key=lambda x: x['holdout_rmse'])
    print('\nBest holdout result:')
    print(best_holdout)

if __name__ == '__main__':
    main()
