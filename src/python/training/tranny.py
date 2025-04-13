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
from python.evaluation.evaluation import evaluate_saved_model # Use existing evaluation

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

def main():
    """Main function for Transformer model training."""
    
    # --- Configuration ---
    TARGET_VARIABLE = "Duration_In_Min" 
    BASE_FEATURES_TO_DROP = [
        'Student_IDs', 'Semester', 'Class_Standing', 'Major',
        'Expected_Graduation', 'Course_Name', 'Course_Number',
        'Course_Type', 'Course_Code_by_Thousands', 'Check_Out_Time',
        'Session_Length_Category', 'Check_In_Date', 'Semester_Date',
        'Expected_Graduation_Date',
        'Duration_In_Min', 'Occupancy'
    ]
    TEST_SPLIT_RATIO = 0.8 
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    EPOCHS = 20 # Keep 30 epochs for now
    OUTPUT_DIM = 1
    
    # --- Transformer Hyperparameters ---
    D_MODEL = 128      # Embedding dimension (input feature proj)
    NHEAD = 8          # Number of attention heads (must divide D_MODEL evenly)
    D_HID = 256        # Dimension of feedforward network
    NLAYERS = 4        # Number of TransformerEncoder layers
    DROPOUT = 0.2      # Dropout rate
    # ---------------------------------
    
    # --- Device Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Paths ---
    DATA_DIR = os.path.join(project_root, "data", "processed")
    TRAIN_FILE = os.path.join(DATA_DIR, "train_engineered.csv")
    SAVE_DIR = os.path.join(project_root, "artifacts", "models", "pytorch")
    os.makedirs(SAVE_DIR, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_transformer_model_{TARGET_VARIABLE}.pth")

    try:
        # 1. Load Data (using utils)
        full_df = load_data(TRAIN_FILE)
        print("\nData loaded successfully.")

        # 2. Train/Test Split (using utils method if available, else basic split)
        n_rows = len(full_df)
        split_idx = int(n_rows * TEST_SPLIT_RATIO)
        df_train = full_df.iloc[:split_idx].copy()
        df_test = full_df.iloc[split_idx:].copy()
        print(f"Data split: Train={len(df_train)} rows, Test={len(df_test)} rows")

        # 3. Preprocess Data (using utils)
        cols_to_drop_this_run = list(set(BASE_FEATURES_TO_DROP) - {TARGET_VARIABLE})
        print("\nPreprocessing training data...")
        X_train, y_train = preprocess_data(df_train, TARGET_VARIABLE, cols_to_drop_this_run)
        print("\nPreprocessing test data...")
        X_test, y_test = preprocess_data(df_test, TARGET_VARIABLE, cols_to_drop_this_run)

        # Align columns (same logic as gruey.py)
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        if train_cols != test_cols:
            print("Warning: Train and test columns differ after preprocessing!")
            missing_in_test = list(train_cols - test_cols)
            for col in missing_in_test: X_test[col] = 0
            extra_in_test = list(test_cols - train_cols)
            if extra_in_test: X_test = X_test.drop(columns=extra_in_test)
            X_test = X_test[X_train.columns]
            print("Aligned test columns.")

        # Convert bools (same logic)
        print("\nConverting boolean columns...")
        for col in X_train.columns:
            if X_train[col].dtype == 'bool':
                X_train[col] = X_train[col].astype(int)
                X_test[col] = X_test[col].astype(int)

        # 4. Feature Scaling (same logic)
        print("\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Features scaled.")
        input_dim = X_train_scaled.shape[1] # Get input dim AFTER scaling

        # 5. Create Datasets and DataLoaders (using datasets.py)
        print("\nCreating Datasets and DataLoaders...")
        train_dataset = TabularDataset(X_train_scaled, y_train)
        test_dataset = TabularDataset(X_test_scaled, y_test)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print("DataLoaders created.")

        # 6. Initialize Transformer Model, Loss, Optimizer
        print(f"\nInitializing TabularTransformerModel (Input Dim: {input_dim})...")
        model = TabularTransformerModel(
            input_dim=input_dim,
            d_model=D_MODEL,
            nhead=NHEAD,
            d_hid=D_HID,
            nlayers=NLAYERS,
            output_dim=OUTPUT_DIM,
            dropout=DROPOUT
        ).to(device)
        
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print("Model, Loss Function, Optimizer initialized.")
        print(model)

        # 7. Training Loop
        print(f"\nStarting training for {EPOCHS} epochs...")
        print(f"Best model will be saved to: {MODEL_SAVE_PATH}")
        best_eval_loss = float('inf')

        for epoch in range(EPOCHS):
            # Training step
            epoch_train_loss = 0.0
            model.train()
            for features, targets in train_loader:
                loss = train_step(model, features, targets, loss_fn, optimizer, device)
                epoch_train_loss += loss
            avg_epoch_train_loss = epoch_train_loss / len(train_loader)

            # Validation step
            avg_epoch_val_loss = evaluate_model(model, test_loader, loss_fn, device)
            
            print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Train Loss: {avg_epoch_train_loss:.4f}, Avg Val Loss: {avg_epoch_val_loss:.4f}")

            # Save best model
            if avg_epoch_val_loss < best_eval_loss:
                best_eval_loss = avg_epoch_val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"  New best model saved with validation loss: {best_eval_loss:.4f}")

        print("\nTraining finished.")
        print(f"Best model saved to {MODEL_SAVE_PATH} with validation loss: {best_eval_loss:.4f}")

        # 8. Final Evaluation on Test Set (using evaluation.py)
        print("\n--- Evaluating Best Transformer Model on Test Set ---")
        if os.path.exists(MODEL_SAVE_PATH):
            # Need to re-instantiate the architecture to pass to the evaluation function
            eval_model_instance = TabularTransformerModel(
                 input_dim=input_dim,
                 d_model=D_MODEL,
                 nhead=NHEAD,
                 d_hid=D_HID,
                 nlayers=NLAYERS,
                 output_dim=OUTPUT_DIM,
                 dropout=DROPOUT # Use the same dropout, although it's disabled by model.eval()
            ).to(device) # Important: move the instance to the correct device!

            test_metrics = evaluate_saved_model(
                 model_path=MODEL_SAVE_PATH,
                 model_architecture_instance=eval_model_instance, # Pass the instantiated model
                 test_loader=test_loader,
                 y_test=y_test, 
                 device=device
             )
            # Metrics printed within the function
        else:
             print("Skipping final evaluation: Model file not found.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
