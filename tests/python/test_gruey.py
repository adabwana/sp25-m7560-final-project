import pytest
import pandas as pd
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add src directory to path (assuming test is run from project root)
project_root = os.getcwd() # Get current working directory
src_path = os.path.join(project_root, 'src') # Construct path to src
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import the actual function to be tested
try:
    # Import using full path from src/python
    from python.utils.data_utils import load_data
except ImportError as e:
    # If gruey.py isn't found or causes issues, skip tests
    pytest.skip(f"Skipping tests: Could not import load_data from python.utils.data_utils. Error: {e}", allow_module_level=True)

# --- Test Cases for load_data ---

def test_load_data_success():
    """Tests if load_data successfully loads a valid CSV file."""
    # Use the dummy file created for tests
    test_file = os.path.join(os.path.dirname(__file__), 'dummy_train_data.csv')
    df = load_data(test_file)
    assert isinstance(df, pd.DataFrame)
    # Add specific checks for the dummy data
    assert not df.empty
    assert list(df.columns) == ['col1', 'col2', 'target']
    assert len(df) == 3

def test_load_data_file_not_found():
    """Tests if load_data raises FileNotFoundError for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        # Assuming load_data is imported correctly now
        load_data("non_existent_file.csv")

# Add more tests later: e.g., handling empty files, incorrect formats

# --- Test Cases for preprocess_data (Placeholder) ---

# Import the actual function
try:
    # Import using full path from src/python
    from python.utils.preprocessing import preprocess_data
except ImportError as e:
    # If gruey.py isn't found or causes issues, skip tests
    pytest.skip(f"Skipping tests: Could not import preprocess_data from python.utils.preprocessing. Error: {e}", allow_module_level=True)

@pytest.fixture
def sample_raw_data():
    """Provides a sample raw DataFrame for preprocessing tests."""
    return pd.DataFrame({
        'Student_IDs': [1, 2, 3],
        'Check_In_Time': ['08:30:00', '09:15:00', '14:05:00'],
        'Course_Type': ['Lecture', 'Lab', 'Lecture'],
        'Building': ['A', 'B', 'A'],
        'Some_Numeric_Feature': [10.5, 22.1, 8.0],
        'Duration_In_Min': [50, 75, 60],  # Target 1
        'Occupancy': [15, 25, 20]       # Target 2
    })

def test_preprocess_data_shapes_types(sample_raw_data):
    """Tests the output shapes and types from preprocess_data."""
    target = 'Duration_In_Min'
    # Mirror cols_to_drop from R recipe/python utils (adjust as needed)
    base_features_to_drop = [
        'Student_IDs', 'Semester', 'Class_Standing', 'Major',
        'Expected_Graduation', 'Course_Name', 'Course_Number',
        'Course_Type', 'Course_Code_by_Thousands', 'Check_Out_Time',
        'Session_Length_Category', 'Check_In_Date', 'Semester_Date',
        'Expected_Graduation_Date',
        'Duration_In_Min', 'Occupancy' # Include both potential targets
    ]
    # Remove the actual target for this run
    cols_to_drop_this_run = list(set(base_features_to_drop) - {target})
    
    # Filter cols_to_drop_this_run to only those present in sample_raw_data
    cols_to_drop_filtered = [col for col in cols_to_drop_this_run if col in sample_raw_data.columns]

    # Use placeholder function for now
    X, y = preprocess_data(sample_raw_data.copy(), target, cols_to_drop_filtered) # Pass a copy
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    # Add specific checks for the results on sample_raw_data
    assert target not in X.columns # Target column removed from X
    assert 'Student_IDs' not in X.columns # Example dropped column
    assert 'Occupancy' not in X.columns
    assert 'Check_In_Time' not in X.columns # Original time column dropped
    assert 'Check_In_Time_Minutes' in X.columns # New time feature added
    # Check dummy var for the 'Building' column (which shouldn't have been dropped)
    assert 'Building_B' in X.columns
    assert X.select_dtypes(include='object').empty # No object columns left
    assert y.name == target
    assert not X.isnull().any().any() # Check for NaNs introduced (shouldn't be any here)

# Add more tests later: e.g., handling empty files, incorrect formats

# --- Test Cases for TabularDataset and DataLoader (Placeholders) ---

# Import the actual class
try:
    # Import using full path from src/python
    from python.datasets import TabularDataset
except ImportError as e:
    # If gruey.py isn't found or causes issues, skip tests
    pytest.skip(f"Skipping tests: Could not import TabularDataset from python.datasets. Error: {e}", allow_module_level=True)

@pytest.fixture
def sample_preprocessed_data():
    """Provides sample preprocessed data (like output from preprocess_data)."""
    X = pd.DataFrame({
        'feat1_num': [1.0, 2.0, 3.0, 4.0],
        'feat2_cat_a': [1, 0, 1, 0],
        'feat3_cat_b': [0, 1, 0, 1]
    })
    y = pd.Series([10, 20, 15, 25])
    return X, y

def test_tabular_dataset_init_len_getitem(sample_preprocessed_data):
    """Tests TabularDataset initialization, __len__, and __getitem__."""
    X, y = sample_preprocessed_data
    dataset = TabularDataset(X, y)
    
    # Test __len__
    assert len(dataset) == len(X)
    
    # Test __getitem__
    idx = 1
    features_item, target_item = dataset[idx]
    
    assert isinstance(features_item, torch.Tensor)
    assert features_item.dtype == torch.float32
    assert features_item.shape == (X.shape[1],) # Shape of a single feature row
    # Check values (convert expected row to numpy for comparison)
    assert torch.equal(features_item, torch.tensor(X.iloc[idx].values, dtype=torch.float32))
    
    assert isinstance(target_item, torch.Tensor)
    assert target_item.dtype == torch.float32
    assert target_item.shape == (1,) # Shape of a single target value (unsqueezed)
    # Check value
    assert torch.equal(target_item, torch.tensor([y.iloc[idx]], dtype=torch.float32))

def test_dataloader_creation(sample_preprocessed_data):
    """Tests creating a DataLoader and iterating one batch."""
    X, y = sample_preprocessed_data
    batch_size = 2
    dataset = TabularDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get one batch
    features_batch, target_batch = next(iter(dataloader))
    
    assert isinstance(features_batch, torch.Tensor)
    assert features_batch.dtype == torch.float32
    assert features_batch.shape == (batch_size, X.shape[1])
    
    assert isinstance(target_batch, torch.Tensor)
    assert target_batch.dtype == torch.float32
    assert target_batch.shape == (batch_size, 1)

    # Check first element of the batch matches first element of dataset (shuffle=False)
    assert torch.equal(features_batch[0], torch.tensor(X.iloc[0].values, dtype=torch.float32))
    assert torch.equal(target_batch[0], torch.tensor([y.iloc[0]], dtype=torch.float32))

# --- Test Cases for GrueyModel (Placeholders) ---

# Import the actual model
try:
    # Import using full path from src/python
    from python.models.gruey_architecture import GrueyModel
except ImportError as e:
    # If gruey.py isn't found or causes issues, skip tests
    pytest.skip(f"Skipping tests: Could not import GrueyModel from python.models.gruey_architecture. Error: {e}", allow_module_level=True)

@pytest.fixture
def model_config():
    return {
        "input_dim": 3, # Matches sample_preprocessed_data features
        "gru_dim": 16,
        "output_dim": 1
    }

def test_gruey_model_init(model_config):
    """Tests GrueyModel initialization."""
    model = GrueyModel(**model_config)
    assert isinstance(model, GrueyModel)
    assert model.input_dim == model_config["input_dim"]
    assert model.gru_dim == model_config["gru_dim"]

def test_gruey_model_forward(model_config, sample_preprocessed_data):
    """Tests the forward pass of GrueyModel."""
    X, y = sample_preprocessed_data
    # Convert sample data to tensor
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    batch_size = X_tensor.shape[0]
    
    model = GrueyModel(**model_config)
    
    # Forward pass
    output = model(X_tensor)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, model_config["output_dim"])
    assert output.dtype == torch.float32

# --- Test Cases for Training Loop Components (Placeholders) ---

# Import the actual function
try:
    # Import using full path from src/python
    from python.training.gruey import train_step
    from python.models.gruey_architecture import GrueyModel
except ImportError as e:
    # If gruey.py isn't found or causes issues, skip tests
    pytest.skip(f"Skipping tests: Could not import from python.training.gruey or python.models.gruey_architecture. Error: {e}", allow_module_level=True)

def test_train_step(model_config, sample_preprocessed_data):
    """Tests the basic execution of a single training step."""
    X, y = sample_preprocessed_data
    features = torch.tensor(X.values, dtype=torch.float32)
    targets = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    
    # Use the actual model now
    model = GrueyModel(**model_config)
    model.train() # Set model to training mode
    
    # Store initial parameters
    initial_params = {name: p.clone() for name, p in model.named_parameters()}
    
    # Define loss and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Execute actual train_step
    loss = train_step(model, features, targets, loss_fn, optimizer)
    
    # Check loss type
    assert isinstance(loss, float), "Loss should be a float"
    
    # Check if gradients were computed (requires_grad should be True for model params)
    grads_exist = any(p.grad is not None for p in model.parameters())
    assert grads_exist, "Gradients should exist after train_step"
    
    # Check if parameters were updated
    params_changed = False
    for name, p in model.named_parameters():
        if not torch.equal(p, initial_params[name]):
            params_changed = True
            break
    assert params_changed, "Parameters should change after optimizer step" 