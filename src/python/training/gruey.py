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
from python.evaluation.evaluation import evaluate_saved_model 

# --- Training Step Function ---
def train_step(model: torch.nn.Module, 
               features: torch.Tensor, 
               targets: torch.Tensor, 
               loss_fn: torch.nn.modules.loss._Loss, # Type hint for loss functions
               optimizer: torch.optim.Optimizer) -> float:
    """Performs a single training step: forward, loss, backward, optimize.

    Args:
        model (torch.nn.Module): The model to train.
        features (torch.Tensor): Batch of input features.
        targets (torch.Tensor): Batch of target values.
        loss_fn (torch.nn.modules.loss._Loss): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.

    Returns:
        float: The loss value for this batch.
    """
    # Set model to training mode
    model.train()
    # 1. Forward pass
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
                   loss_fn: torch.nn.modules.loss._Loss) -> float:
    """Evaluates the model on the provided data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation set (e.g., validation).
        loss_fn (torch.nn.modules.loss._Loss): The loss function.

    Returns:
        float: The average loss over the evaluation dataset.
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad(): # Turn off gradient computation
        for features, targets in data_loader:
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def main():
    """Main function to orchestrate data loading, preprocessing, training."""
    
    # --- Configuration ---
    TARGET_VARIABLE = "Duration_In_Min" # Or "Occupancy"
    BASE_FEATURES_TO_DROP = [
        'Student_IDs', 'Semester', 'Class_Standing', 'Major',
        'Expected_Graduation', 'Course_Name', 'Course_Number',
        'Course_Type', 'Course_Code_by_Thousands', 'Check_Out_Time',
        'Session_Length_Category', 'Check_In_Date', 'Semester_Date',
        'Expected_Graduation_Date',
        'Duration_In_Min', 'Occupancy' # Include both potential targets
    ]
    TEST_SPLIT_RATIO = 0.8 # 80% train
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    EPOCHS = 30 # Start with a small number for testing
    GRU_DIM = 64
    GRU_EXPANSION = 1.0
    OUTPUT_DIM = 1
    
    # --- Device Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # --------------------------
    
    # Define constants needed for main execution
    # Ensure project_root is defined (should be from path setup above)
    DATA_DIR = os.path.join(project_root, "data", "processed")
    TRAIN_FILE = os.path.join(DATA_DIR, "train_engineered.csv")

    try:
        # 1. Load Data
        full_df = load_data(TRAIN_FILE)
        print("\nData loaded successfully:")
        print(full_df.head())
        
        # 2. Train/Test Split (Manual non-shuffled)
        n_rows = len(full_df)
        split_idx = int(n_rows * TEST_SPLIT_RATIO)
        if split_idx <= 0 or split_idx >= n_rows:
            raise ValueError("Invalid train/test split size.")
        df_train = full_df.iloc[:split_idx].copy()
        df_test = full_df.iloc[split_idx:].copy()
        print(f"\nData split: Train={len(df_train)} rows, Test={len(df_test)} rows")

        # 3. Preprocess Data
        cols_to_drop_this_run = list(set(BASE_FEATURES_TO_DROP) - {TARGET_VARIABLE})
        print("\nPreprocessing training data...")
        X_train, y_train = preprocess_data(df_train, TARGET_VARIABLE, cols_to_drop_this_run)
        print("\nPreprocessing test data...")
        X_test, y_test = preprocess_data(df_test, TARGET_VARIABLE, cols_to_drop_this_run)

        # Ensure columns match after dummy encoding
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        if train_cols != test_cols:
            print("Warning: Train and test columns differ after preprocessing!")
            missing_in_test = list(train_cols - test_cols)
            for col in missing_in_test:
                X_test[col] = 0
            if missing_in_test:
                print(f"Added missing columns to test: {missing_in_test}")
            extra_in_test = list(test_cols - train_cols)
            if extra_in_test:
                 X_test = X_test.drop(columns=extra_in_test)
                 print(f"Removed extra columns from test: {extra_in_test}")
            X_test = X_test[X_train.columns]
            print("Aligned test columns with training columns.")
        
        # --- Ensure all columns are numeric for PyTorch ---
        print("\nConverting boolean columns to int...")
        for col in X_train.columns:
            if X_train[col].dtype == 'bool':
                X_train[col] = X_train[col].astype(int)
                X_test[col] = X_test[col].astype(int) # Apply same to test set

        # --- Add Feature Scaling ---
        print("\nScaling features using StandardScaler...")
        scaler = StandardScaler()
        # Fit on training data and transform both train and test
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Features scaled.")
        # --- End Feature Scaling ---

        # --- Debug: Check dtypes before creating Dataset ---
        # Note: X_train_scaled and X_test_scaled are now numpy arrays
        print("\n--- X_train_scaled info after scaling: ---")
        print(f"Type: {type(X_train_scaled)}, Shape: {X_train_scaled.shape}, Dtype: {X_train_scaled.dtype}")
        # print(X_train.info()) # X_train is still a DataFrame, info is less relevant now
        # non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns
        # if not non_numeric_cols.empty:
        #     print(f"Warning: Non-numeric columns found in X_train: {list(non_numeric_cols)}")
            # Optionally print the head of non-numeric columns for inspection
            # print(X_train[non_numeric_cols].head())
        # --- End Debug ---

        # 4. Create Datasets and DataLoaders
        print("\nCreating Datasets and DataLoaders...")
        # Pass the scaled numpy arrays directly to the dataset
        train_dataset = TabularDataset(X_train_scaled, y_train) 
        test_dataset = TabularDataset(X_test_scaled, y_test) 
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"Datasets created with scaled data. Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

        # 5. Initialize Model, Loss, Optimizer
        input_dim = X_train_scaled.shape[1] # Get input dim from scaled array
        print(f"\nInput dimension for model: {input_dim}")
        model = GrueyModel(
            input_dim=input_dim, 
            gru_dim=GRU_DIM, 
            output_dim=OUTPUT_DIM, 
            gru_expansion_factor=GRU_EXPANSION
        )
        model.to(device) # Move model to the designated device
        loss_fn = torch.nn.MSELoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print("Model, Loss Function, Optimizer initialized.")
        print(model)

        # 6. Training Loop
        print(f"\nStarting training for {EPOCHS} epochs...")
        best_eval_loss = float('inf') # Initialize best evaluation loss
        # --- Define Save Path for Best Model ---
        save_dir = os.path.join(project_root, "artifacts", "models", "pytorch")
        os.makedirs(save_dir, exist_ok=True) # Ensure directory exists
        model_save_path = os.path.join(save_dir, f"best_gruey_model_{TARGET_VARIABLE}.pth")
        print(f"Best model will be saved to: {model_save_path}")

        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            model.train() 
            for batch_idx, (features, targets) in enumerate(train_loader):
                # --- Training Step --- 
                loss = train_step(model, features, targets, loss_fn, optimizer)
                epoch_loss += loss
                # Optional: Print batch loss periodically
                # if batch_idx % 100 == 0:
                #    print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.4f}")

            avg_epoch_loss = epoch_loss / len(train_loader)
            
            # --- Evaluation Step ---
            avg_eval_loss = evaluate_model(model, test_loader, loss_fn)
            
            print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Train Loss: {avg_epoch_loss:.4f}, Avg Val Loss: {avg_eval_loss:.4f}")

            # --- Save Best Model ---            
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"  New best model saved with validation loss: {best_eval_loss:.4f}")

        print("\nTraining finished.")
        # --- Removed redundant save block for final model --- 
        # Final message now correctly refers to the best model saved during training
        print(f"Best model saved to {model_save_path} with validation loss: {best_eval_loss:.4f}")

        # --- Final Evaluation on Test Set ---
        print("\n--- Evaluating Best Model on Test Set ---")
        # Ensure y_test (pandas Series) is available here
        if 'y_test' in locals() and 'test_loader' in locals() and os.path.exists(model_save_path):
             # Instantiate the GrueyModel architecture to pass to the evaluation function
             eval_model_instance = GrueyModel(
                 input_dim=input_dim, 
                 gru_dim=GRU_DIM,
                 output_dim=OUTPUT_DIM,
                 gru_expansion_factor=GRU_EXPANSION
             ).to(device) # Move instance to the correct device

             test_metrics = evaluate_saved_model(
                 model_path=model_save_path,
                 model_architecture_instance=eval_model_instance, # Pass the instantiated model
                 test_loader=test_loader,
                 y_test=y_test, # Pass the original y_test Series
                 device=device
             )
             # Metrics are already printed within the function
        else:
             print("Skipping final evaluation: Required data/model not available.")
        # --------------------------------------

    except (FileNotFoundError, IOError) as e:
        print(f"\nError during data loading/preprocessing: {e}")
    except ValueError as e:
        print(f"\nConfiguration or data error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging

if __name__ == '__main__':
    main()

# Removed placeholder code below main function