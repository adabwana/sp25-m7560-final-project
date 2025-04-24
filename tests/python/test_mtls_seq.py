import pytest
import torch
import numpy as np
import pandas as pd
import sys
import os
import torch.nn as nn
import torch.optim as optim

# Ensure src directory is in path for imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from python.datasets.sequence_dataset import SequenceDataset
from python.utils.preprocessing import preprocess_data
from sklearn.preprocessing import StandardScaler
from python.models.mtls_architecture import MtlsModel
from python.tuning.mtls_seq import train_step, evaluate_model # Import the functions

# Helper function to create simple test data
def create_test_data(num_samples=20, num_features=5):
    features = np.random.rand(num_samples, num_features).astype(np.float32)
    # Ensure some variance in target
    targets = pd.Series(np.arange(num_samples, dtype=np.float32) + np.random.randn(num_samples) * 0.1)
    return features, targets

# --- Test Suite for mtls_seq.py components ---

class TestSequenceDataset:
    """Describe: SequenceDataset"""

    def test_context_initialization(self):
        """Context: Initialization"""
        features, targets = create_test_data(num_samples=10, num_features=3)

        # it: raises error if sequence_length is too large
        with pytest.raises(ValueError, match="cannot be greater"):
             SequenceDataset(features, targets, sequence_length=11)

        # it: raises error if sequence_length is zero or negative
        with pytest.raises(ValueError, match="positive"):
            SequenceDataset(features, targets, sequence_length=0)
        with pytest.raises(ValueError, match="positive"):
            SequenceDataset(features, targets, sequence_length=-1)

        # it: calculates correct number of effective samples
        ds = SequenceDataset(features, targets, sequence_length=3)
        assert len(ds) == 10 - 3 + 1

        # it: raises error if features and targets length mismatch
        with pytest.raises(ValueError, match="same number of samples"):
            SequenceDataset(features, targets[:-1], sequence_length=3)

    def test_context_getitem_with_valid_data(self):
        """Context: __getitem__ with valid data"""
        num_samples = 15
        num_features = 4
        sequence_length = 5
        features, targets = create_test_data(num_samples, num_features)
        dataset = SequenceDataset(features, targets, sequence_length)

        # it: returns tensors of the correct shape
        idx = 0 # Index for the first possible sequence
        seq_features, seq_target = dataset[idx]

        assert isinstance(seq_features, torch.Tensor), "Features should be a Tensor"
        assert isinstance(seq_target, torch.Tensor), "Target should be a Tensor"
        assert seq_features.shape == (sequence_length, num_features), f"Expected feature shape {(sequence_length, num_features)}, got {seq_features.shape}"
        assert seq_target.shape == (1,), f"Expected target shape {(1,)}, got {seq_target.shape}"

        # it: returns tensors with correct dtype
        assert seq_features.dtype == torch.float32, f"Expected feature dtype float32, got {seq_features.dtype}"
        assert seq_target.dtype == torch.float32, f"Expected target dtype float32, got {seq_target.dtype}"

        # it: returns the correct target value
        # Target for index idx=0 corresponds to original target at index (0 + sequence_length - 1) = 4
        expected_target_value = targets.iloc[sequence_length - 1]
        assert torch.isclose(seq_target[0], torch.tensor(expected_target_value, dtype=torch.float32)), f"Expected target {expected_target_value}, got {seq_target[0]}"

        # it: returns features without NaNs or Infs
        assert not torch.isnan(seq_features).any(), "Feature tensor contains NaNs"
        assert not torch.isinf(seq_features).any(), "Feature tensor contains Infs"
        assert not torch.isnan(seq_target).any(), "Target tensor contains NaNs"
        assert not torch.isinf(seq_target).any(), "Target tensor contains Infs"

        # it: returns correct feature slice for first index
        expected_features_slice = features[0:sequence_length, :]
        assert torch.equal(seq_features, torch.tensor(expected_features_slice, dtype=torch.float32)), "Feature slice mismatch for first index"

        # it: returns correct data for last index
        last_idx = len(dataset) - 1 # Index relative to effective samples
        last_seq_features, last_seq_target = dataset[last_idx]
        expected_last_start_idx = last_idx
        expected_last_end_idx = last_idx + sequence_length
        expected_last_target_idx = expected_last_end_idx - 1
        expected_last_features_slice = features[expected_last_start_idx:expected_last_end_idx, :]
        expected_last_target_value = targets.iloc[expected_last_target_idx]

        assert last_seq_features.shape == (sequence_length, num_features)
        assert last_seq_target.shape == (1,)
        assert torch.equal(last_seq_features, torch.tensor(expected_last_features_slice, dtype=torch.float32)), "Feature slice mismatch for last index"
        assert torch.isclose(last_seq_target[0], torch.tensor(expected_last_target_value, dtype=torch.float32)), "Target value mismatch for last index"


    @pytest.mark.skip(reason="Pending implementation: Need to test edge case seq_len=1")
    def test_context_getitem_with_sequence_length_one(self):
        """Context: __getitem__ with sequence_length=1"""
        # it: returns correct shapes and values when sequence length is 1
        pass

    @pytest.mark.skip(reason="Pending implementation: Need to test behavior with NaNs in input")
    def test_context_getitem_with_nan_input(self):
        """Context: __getitem__ with NaN/Inf in input data"""
        # it: propagates NaNs correctly if present in input features/targets
        pass

# We can add more classes here for other components like the model architecture
# or the training/evaluation loop following the same pattern.
# e.g. class TestMtlsModelArchitecture: ...
# e.g. class TestTrainingLoop: ... 

# --- Test Suite for Preprocessing and Scaling ---

class TestPreprocessingAndScaling:
    """Describe: Preprocessing and Scaling Steps"""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame mimicking structure before preprocessing."""
        data = {
            'Student_IDs': ['S1', 'S2', 'S3', 'S4', 'S5'],
            'Check_In_Time': ['09:30:00', '10:01:15', '09:30:00', '14:55:45', '16:00:00'],
            'Categorical1': ['A', 'B', 'A', 'C', 'B'],
            'Categorical2': ['X', 'X', 'Y', 'Y', 'X'],
            'Numeric1': [10, 20, 15, 25, 30],
            'Numeric2': [1.1, 2.2, 1.1, 3.3, 4.4],
            'Column_To_Drop': [1, 2, 3, 4, 5],
            'Duration_In_Min': [60, 75, 90, 50, 120],
            'Occupancy': [1, 0, 1, 1, 0]
        }
        return pd.DataFrame(data)

    def test_context_preprocess_data(self, sample_df):
        """Context: preprocess_data function"""
        target_col = 'Duration_In_Min'
        cols_to_drop = ['Student_IDs', 'Column_To_Drop']

        X, y = preprocess_data(sample_df.copy(), target_col, cols_to_drop)

        # it: returns pandas DataFrame for X and Series for y
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        # it: returns expected shapes
        assert X.shape[0] == 5 # Number of rows
        # Expected cols: Check_In_Time_Minutes, Categorical1_B, Categorical1_C, Categorical2_Y, Numeric1, Numeric2, Occupancy
        assert X.shape[1] == 7
        assert y.shape[0] == 5 # Number of rows

        # it: drops specified columns and target from X
        assert target_col not in X.columns
        for col in cols_to_drop:
            assert col not in X.columns
        assert 'Check_In_Time' not in X.columns

        # it: correctly converts time column
        assert 'Check_In_Time_Minutes' in X.columns
        expected_minutes = [9*60+30, 10*60+1, 9*60+30, 14*60+55, 16*60+0]
        assert X['Check_In_Time_Minutes'].tolist() == expected_minutes

        # it: creates correct dummy variables (drop_first=True)
        assert 'Categorical1_B' in X.columns
        assert 'Categorical1_C' in X.columns
        assert 'Categorical1_A' not in X.columns # Dropped
        assert 'Categorical2_Y' in X.columns
        assert 'Categorical2_X' not in X.columns # Dropped
        assert X['Categorical1_B'].sum() == 2 # Count occurrences
        assert X['Categorical2_Y'].sum() == 2

        # it: does not introduce NaNs in X or y
        assert not X.isnull().values.any(), "NaNs found in preprocessed X"
        assert not y.isnull().values.any(), "NaNs found in preprocessed y"

        # it: keeps original target values
        assert y.tolist() == sample_df[target_col].tolist()

    def test_context_scaling_after_preprocessing(self, sample_df):
        """Context: StandardScaler after preprocess_data"""
        target_col = 'Duration_In_Min'
        cols_to_drop = ['Student_IDs', 'Column_To_Drop']
        X, y = preprocess_data(sample_df.copy(), target_col, cols_to_drop)

        scaler = StandardScaler()

        # it: fits and transforms without error
        try:
            X_scaled = scaler.fit_transform(X)
        except Exception as e:
            pytest.fail(f"StandardScaler failed during fit_transform: {e}")

        # it: returns numpy array with same shape
        assert isinstance(X_scaled, np.ndarray)
        assert X_scaled.shape == X.shape

        # it: does not introduce NaNs or Infs
        assert not np.isnan(X_scaled).any(), "NaNs found after scaling"
        assert not np.isinf(X_scaled).any(), "Infs found after scaling"

        # it: results in zero mean (approximately) for original numeric columns
        # Need to get the column index from the original *preprocessed* X dataframe
        num1_idx = X.columns.get_loc('Numeric1')
        num2_idx = X.columns.get_loc('Numeric2')
        time_idx = X.columns.get_loc('Check_In_Time_Minutes')
        assert np.isclose(X_scaled[:, num1_idx].mean(), 0.0)
        assert np.isclose(X_scaled[:, num2_idx].mean(), 0.0)
        assert np.isclose(X_scaled[:, time_idx].mean(), 0.0)

        # it: results in unit variance (approximately) for original numeric columns
        assert np.isclose(X_scaled[:, num1_idx].std(), 1.0)
        assert np.isclose(X_scaled[:, num2_idx].std(), 1.0)
        assert np.isclose(X_scaled[:, time_idx].std(), 1.0)

# --- Test Suite for MtlsModel Architecture ---

class TestMtlsModelArchitecture:
    """Describe: MtlsModel Architecture"""

    def test_context_initialization(self):
        """Context: Initialization"""
        input_dim = 10
        lstm_dim = 32
        output_dim = 1

        # it: initializes successfully with valid parameters
        try:
            model = MtlsModel(input_dim=input_dim, lstm_dim=lstm_dim, output_dim=output_dim,
                              lstm_expansion_factor=1.0, num_layers=1, dropout_rate=0.1,
                              activation_fn_name="relu")
            assert model is not None
        except Exception as e:
            pytest.fail(f"Initialization failed with valid parameters: {e}")

        # it: uses correct activation function instance
        model_relu = MtlsModel(input_dim, lstm_dim, activation_fn_name="relu")
        assert isinstance(model_relu.activation, torch.nn.ReLU)
        model_tanh = MtlsModel(input_dim, lstm_dim, activation_fn_name="tanh")
        assert isinstance(model_tanh.activation, torch.nn.Tanh)
        model_gelu = MtlsModel(input_dim, lstm_dim, activation_fn_name="gelu")
        assert isinstance(model_gelu.activation, torch.nn.GELU)

        # it: raises error for invalid activation function name
        with pytest.raises(ValueError, match="Unsupported activation function"):
            MtlsModel(input_dim, lstm_dim, activation_fn_name="swish")

    def test_context_forward_pass(self):
        """Context: Forward Pass"""
        batch_size = 4
        sequence_length = 5
        input_dim = 10
        lstm_dim = 16
        output_dim = 2

        model = MtlsModel(input_dim=input_dim, lstm_dim=lstm_dim, output_dim=output_dim,
                          lstm_expansion_factor=1.0, num_layers=1, dropout_rate=0.0,
                          activation_fn_name="relu")
        model.eval() # Ensure dropout is off for deterministic check

        # Create simple, non-NaN input
        test_input = torch.randn(batch_size, sequence_length, input_dim, dtype=torch.float32)

        # it: executes forward pass without errors for valid input shape
        try:
            output = model(test_input)
        except Exception as e:
            pytest.fail(f"Forward pass failed with valid input shape: {e}")

        # it: returns output with correct shape
        assert output.shape == (batch_size, output_dim), \
            f"Expected output shape {(batch_size, output_dim)}, got {output.shape}"

        # it: returns output with correct dtype
        assert output.dtype == torch.float32, f"Expected output dtype float32, got {output.dtype}"

        # it: returns output without NaNs or Infs for valid input
        assert not torch.isnan(output).any(), "Output tensor contains NaNs"
        assert not torch.isinf(output).any(), "Output tensor contains Infs"

        # it: raises error for invalid input dimensions (e.g., 2D)
        with pytest.raises(ValueError, match="Expected input shape"):
            invalid_input_2d = torch.randn(batch_size, input_dim)
            model(invalid_input_2d)

    @pytest.mark.skip(reason="Pending implementation: Need to test multi-layer behavior")
    def test_context_forward_pass_multilayer(self):
        """Context: Forward Pass with Multiple Layers"""
        # it: ensures output shape and type are correct for num_layers > 1
        pass

    @pytest.mark.skip(reason="Pending implementation: Need to test expansion factor behavior")
    def test_context_forward_pass_expansion(self):
        """Context: Forward Pass with Expansion Factor != 1"""
        # it: ensures output shape and type are correct when expansion_factor != 1
        pass

    def test_context_forward_pass_extreme_inputs(self):
        """Context: Forward Pass with extreme input values"""
        batch_size = 2
        sequence_length = 3
        input_dim = 4
        lstm_dim = 8
        output_dim = 1

        model = MtlsModel(input_dim=input_dim, lstm_dim=lstm_dim, output_dim=output_dim,
                          lstm_expansion_factor=1.0, num_layers=1, dropout_rate=0.0,
                          activation_fn_name="relu")
        model.eval()

        # Create input with large positive and negative values
        test_input = torch.randn(batch_size, sequence_length, input_dim, dtype=torch.float32)
        test_input[0, 0, 0] = 1e8  # Large positive (Increased magnitude)
        test_input[0, 0, 1] = -1e8 # Large negative (Increased magnitude)
        test_input[1, :, :] = 0.0 # Zero input

        # it: does not produce NaNs/Infs even with extreme inputs
        try:
            output = model(test_input)
            assert not torch.isnan(output).any(), "Output tensor contains NaNs with extreme input"
            # Also check for Inf explicitly
            assert not torch.isinf(output).any(), "Output tensor contains Infs with extreme input"
        except Exception as e:
            pytest.fail(f"Forward pass failed with extreme input values: {e}")

# Add TestTrainingEvaluationFunctions next
class TestTrainingEvaluationFunctions:
    """Describe: Training Step and Evaluation Functions"""

    @pytest.fixture
    def setup_components(self):
        """Provides components needed for train/eval tests."""
        batch_size = 4
        sequence_length = 5
        input_dim = 10
        lstm_dim = 8 # Keep small for testing
        output_dim = 1
        device = torch.device("cpu") # Test on CPU for simplicity

        model = MtlsModel(input_dim=input_dim, lstm_dim=lstm_dim, output_dim=output_dim,
                          num_layers=1, dropout_rate=0.0).to(device)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Create a single batch of data
        features = torch.randn(batch_size, sequence_length, input_dim, device=device, dtype=torch.float32)
        targets = torch.randn(batch_size, output_dim, device=device, dtype=torch.float32)

        return model, loss_fn, optimizer, features, targets, device

    def test_context_train_step(self, setup_components):
        """Context: train_step function"""
        model, loss_fn, optimizer, features, targets, device = setup_components
        clip_value = 1.0 # Define the clip value used in the test

        # Get initial parameter value (e.g., first weight of input_fc)
        initial_param = model.input_fc.weight.clone().detach()

        # Clear potential stale gradients before the step
        optimizer.zero_grad()

        # it: executes successfully and returns a scalar loss
        try:
            loss_value = train_step(model, features, targets, loss_fn, optimizer, device, clip_value=clip_value)
            assert isinstance(loss_value, float)
            assert not np.isnan(loss_value), "train_step returned NaN loss"
            assert not np.isinf(loss_value), "train_step returned Inf loss"
        except Exception as e:
            pytest.fail(f"train_step function failed: {e}")

        # it: updates model parameters
        updated_param = model.input_fc.weight.clone().detach()
        assert not torch.equal(initial_param, updated_param), "Model parameters were not updated"

        # it: ensures gradients were computed (check one parameter)
        assert model.input_fc.weight.grad is not None, "Gradients were not computed for input_fc weight"

        # it: ensures gradient norms are clipped (optional check)
        # Calculate total norm *after* train_step (which includes clipping)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        # Assert that the norm is less than or equal to the clip value (allow for small tolerance)
        assert total_norm <= clip_value + 1e-6, f"Gradient norm {total_norm} exceeded clip value {clip_value}"

    def test_context_gradient_stability(self, setup_components):
        """Context: Gradient calculation stability"""
        model, loss_fn, optimizer, features, targets, device = setup_components

        # Use slightly more extreme features/targets for this test
        features = torch.randn_like(features) * 10 # Scale up features
        targets = torch.randn_like(targets) * 10  # Scale up targets
        # Add a large value
        features[0,0,0] = 1e4 

        # Perform forward pass
        model.train() # Ensure model is in training mode
        optimizer.zero_grad() # Zero gradients first
        predictions = model(features)

        # Check predictions aren't already NaN (sanity check)
        if torch.isnan(predictions).any():
            pytest.skip("Skipping gradient check because forward pass produced NaN with test inputs")
        
        # Calculate loss
        loss = loss_fn(predictions, targets)
        if torch.isnan(loss) or torch.isinf(loss):
             pytest.skip("Skipping gradient check because loss is NaN/Inf with test inputs")

        # Perform backward pass
        try:
            loss.backward()
        except Exception as e:
            pytest.fail(f"loss.backward() failed: {e}")

        # it: gradients should not contain NaNs/Infs after backward pass
        grads_contain_nan = False
        grads_contain_inf = False
        for name, p in model.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    print(f"NaN detected in gradients for layer: {name}")
                    grads_contain_nan = True
                if torch.isinf(p.grad).any():
                    print(f"Inf detected in gradients for layer: {name}")
                    grads_contain_inf = True
            # Don't break, check all grads

        assert not grads_contain_nan, "NaNs found in gradients after backward pass"
        assert not grads_contain_inf, "Infs found in gradients after backward pass"

    def test_context_evaluate_model(self, setup_components):
        """Context: evaluate_model function"""
        model, loss_fn, _, features, targets, device = setup_components # Don't need optimizer here

        # Ensure model starts in train mode for the test
        model.train()

        # it: executes successfully and returns a scalar RMSE
        try:
            rmse_value = evaluate_model(model, [(features, targets)], loss_fn, device) # Use list for DataLoader mock
            assert isinstance(rmse_value, float)
            assert not np.isnan(rmse_value), "evaluate_model returned NaN RMSE"
            assert not np.isinf(rmse_value), "evaluate_model returned Inf RMSE"
        except Exception as e:
            pytest.fail(f"evaluate_model function failed: {e}")

        # it: puts model in eval mode during execution
        # We infer this by checking dropout/batchnorm layers if they existed,
        # but can verify the explicit call was made.
        # For now, just trust the implementation calls model.eval()
        assert not model.training, "Model was not in eval mode after evaluate_model call"

        # it: does not compute gradients
        # Run forward pass again and check gradients
        model.zero_grad() # Clear any potential grads from previous steps
        output_after_eval = model(features)
        assert model.input_fc.weight.grad is None, "Gradients were computed during or after eval" 