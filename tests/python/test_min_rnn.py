import pytest
import torch
import sys
import os

# Add the specific model directories to the Python path for direct import
tests_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(tests_dir, '..', '..'))
models_path = os.path.join(project_root, 'src', 'python', 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)

# Also add src/python path to resolve potential dependencies within models
src_python_path = os.path.join(project_root, 'src', 'python')
if src_python_path not in sys.path:
    sys.path.insert(0, src_python_path) # Insert at 0 to prioritize

# Try importing models - handle potential ModuleNotFoundError if structure is different
try:
    # Import directly now that models_path is in sys.path
    from mingru import minGRU
except (ModuleNotFoundError, ImportError) as e:
    print(f"Import Error minGRU: {e}", file=sys.stderr) # Debug output
    pytest.skip(f"Skipping minGRU tests: Could not import from {models_path}. Error: {e}", allow_module_level=True)
    minGRU = None # Define as None

try:
    # Import directly now that models_path is in sys.path
    from minlstm import minLSTM
except (ModuleNotFoundError, ImportError) as e:
    print(f"Import Error minLSTM: {e}", file=sys.stderr) # Debug output
    pytest.skip(f"Skipping minLSTM tests: Could not import from {models_path}. Error: {e}", allow_module_level=True)
    minLSTM = None # Define as None

# --- Test Fixtures ---

@pytest.fixture
def config():
    """Provides common configuration for tests."""
    return {
        "dim": 32,
        "batch_size": 4,
        "seq_len_parallel": 10,
        "seq_len_sequential": 1,
        "expansion_factor": 1.5
    }

@pytest.fixture
def sample_input_parallel(config):
    """Provides sample input for parallel processing tests."""
    return torch.randn(config["batch_size"], config["seq_len_parallel"], config["dim"])

@pytest.fixture
def sample_input_sequential(config):
    """Provides sample input for sequential processing tests."""
    return torch.randn(config["batch_size"], config["seq_len_sequential"], config["dim"])

@pytest.fixture
def sample_prev_hidden(config):
    """Provides sample previous hidden state."""
    dim_inner = int(config["dim"] * config["expansion_factor"])
    # Hidden state shape matches inner dim before projection (for GRU/LSTM logic)
    # Note: Check if the models expect prev_hidden to be dim_inner or dim
    # Based on the code, prev_hidden is compared/lerped with g(hidden) which is dim_inner.
    # Let's assume prev_hidden should match the *output* dim (dim) and models handle internally.
    # RETHINK: The parallel log_values seems to use prev_hidden.log() directly.
    #          Let's assume prev_hidden should be log-space positive, matching the output 'out'
    #          before the final projection. Let's use dim_inner for safety.
    # CORRECTION: The output `out` is dim_inner before final projection. prev_hidden should match this.
    #             The value itself should represent the state *after* non-linearity (like g(h))
    #             Let's use positive values for log-space compatibility.
    return torch.rand(config["batch_size"], 1, dim_inner) + 0.1 # Positive values


# --- minGRU Tests ---

@pytest.mark.skipif(minGRU is None, reason="minGRU not imported")
def test_mingru_initialization(config):
    """Test if minGRU initializes correctly."""
    model = minGRU(dim=config["dim"])
    assert isinstance(model, minGRU)
    model_expanded = minGRU(dim=config["dim"], expansion_factor=config["expansion_factor"])
    assert isinstance(model_expanded, minGRU)

@pytest.mark.skipif(minGRU is None, reason="minGRU not imported")
def test_mingru_forward_parallel(config, sample_input_parallel):
    """Test the forward pass in parallel mode (seq_len > 1)."""
    model = minGRU(dim=config["dim"], expansion_factor=config["expansion_factor"])
    output = model(sample_input_parallel)
    assert output.shape == (config["batch_size"], config["seq_len_parallel"], config["dim"])

@pytest.mark.skipif(minGRU is None, reason="minGRU not imported")
def test_mingru_forward_sequential(config, sample_input_sequential, sample_prev_hidden):
    """Test the forward pass in sequential mode (seq_len == 1)."""
    model = minGRU(dim=config["dim"], expansion_factor=config["expansion_factor"])
    
    # Test without prev_hidden
    output_no_prev = model(sample_input_sequential, prev_hidden=None)
    assert output_no_prev.shape == (config["batch_size"], config["seq_len_sequential"], config["dim"])

    # Need prev_hidden matching *output* dimension if projection happens
    # If proj_out=True (default with expansion), prev_hidden is dim_inner, output is dim. Mismatch.
    # Let's test with proj_out=False first, where output is dim_inner.
    model_no_proj = minGRU(dim=config["dim"], expansion_factor=1.0, proj_out=False)
    dim_inner = config["dim"]
    prev_hidden_no_proj = torch.rand(config["batch_size"], 1, dim_inner) + 0.1
    output_with_prev = model_no_proj(sample_input_sequential, prev_hidden=prev_hidden_no_proj)
    assert output_with_prev.shape == (config["batch_size"], config["seq_len_sequential"], dim_inner)
    
    # How to test with proj_out=True? The internal state is dim_inner, output is dim.
    # The `prev_hidden` likely needs to be the internal state before projection.
    # Let's assume the `prev_hidden` argument should be the *output* of the previous step.
    # This implies the sequential mode needs careful handling of projection if expansion != 1.
    # The current implementation seems to use prev_hidden *before* final projection (`to_out`).
    # Let's provide prev_hidden matching the internal dim_inner size.
    dim_inner_expanded = int(config["dim"] * config["expansion_factor"])
    prev_hidden_expanded = torch.rand(config["batch_size"], 1, dim_inner_expanded) + 0.1
    output_proj = model(sample_input_sequential, prev_hidden=prev_hidden_expanded)
    assert output_proj.shape == (config["batch_size"], config["seq_len_sequential"], config["dim"])


@pytest.mark.skipif(minGRU is None, reason="minGRU not imported")
def test_mingru_forward_return_hidden(config, sample_input_parallel):
    """Test the return_next_prev_hidden flag."""
    model = minGRU(dim=config["dim"], expansion_factor=config["expansion_factor"])
    dim_inner = int(config["dim"] * config["expansion_factor"])
    
    output, next_hidden = model(sample_input_parallel, return_next_prev_hidden=True)
    assert output.shape == (config["batch_size"], config["seq_len_parallel"], config["dim"])
    assert next_hidden.shape == (config["batch_size"], 1, dim_inner) # Should be internal state before projection

@pytest.mark.skipif(minGRU is None, reason="minGRU not imported")
def test_mingru_forward_with_prev_hidden_parallel(config, sample_input_parallel, sample_prev_hidden):
    """Test parallel mode with an initial hidden state."""
    model = minGRU(dim=config["dim"], expansion_factor=config["expansion_factor"])
    # Provide prev_hidden matching internal dim
    output = model(sample_input_parallel, prev_hidden=sample_prev_hidden)
    assert output.shape == (config["batch_size"], config["seq_len_parallel"], config["dim"])

@pytest.mark.skipif(minGRU is None, reason="minGRU not imported")
def test_mingru_equivalence_parallel_sequential(config):
    """Test numerical equivalence between parallel and sequential modes."""
    model = minGRU(dim=config["dim"], expansion_factor=config["expansion_factor"])
    model.eval() # Ensure consistent behavior (e.g., dropout)
    
    inp = torch.randn(1, config["seq_len_parallel"], config["dim"]) # Use batch_size=1 for simplicity
    dim_inner = int(config["dim"] * config["expansion_factor"])

    # Parallel execution
    with torch.no_grad():
        parallel_out = model(inp)
        parallel_last_out = parallel_out[:, -1:]

    # Sequential execution
    prev_hidden = None
    sequential_last_out = None
    with torch.no_grad():
        for i in range(config["seq_len_parallel"]):
            token_input = inp[:, i:i+1, :] # Get current token
            current_out, next_hidden = model(token_input, prev_hidden, return_next_prev_hidden=True)
            sequential_last_out = current_out
            prev_hidden = next_hidden # Use the returned internal state
    
    assert sequential_last_out is not None, "Sequential loop did not run"
    # Compare the final output of the last step
    assert torch.allclose(parallel_last_out, sequential_last_out, atol=1e-5), \
        f"Max diff: {torch.max(torch.abs(parallel_last_out - sequential_last_out))}"


# --- minLSTM Tests ---

@pytest.mark.skipif(minLSTM is None, reason="minLSTM not imported")
def test_minlstm_initialization(config):
    """Test if minLSTM initializes correctly."""
    model = minLSTM(dim=config["dim"])
    assert isinstance(model, minLSTM)
    model_expanded = minLSTM(dim=config["dim"], expansion_factor=config["expansion_factor"])
    assert isinstance(model_expanded, minLSTM)

@pytest.mark.skipif(minLSTM is None, reason="minLSTM not imported")
def test_minlstm_forward_parallel(config, sample_input_parallel):
    """Test the forward pass in parallel mode (seq_len > 1)."""
    model = minLSTM(dim=config["dim"], expansion_factor=config["expansion_factor"])
    output = model(sample_input_parallel)
    assert output.shape == (config["batch_size"], config["seq_len_parallel"], config["dim"])

@pytest.mark.skipif(minLSTM is None, reason="minLSTM not imported")
def test_minlstm_forward_sequential(config, sample_input_sequential, sample_prev_hidden):
    """Test the forward pass in sequential mode (seq_len == 1)."""
    model = minLSTM(dim=config["dim"], expansion_factor=config["expansion_factor"])
    
    # Test without prev_hidden
    output_no_prev = model(sample_input_sequential, prev_hidden=None)
    assert output_no_prev.shape == (config["batch_size"], config["seq_len_sequential"], config["dim"])

    # Similar to GRU, prev_hidden likely needs to match internal dim_inner size
    dim_inner_expanded = int(config["dim"] * config["expansion_factor"])
    prev_hidden_expanded = torch.rand(config["batch_size"], 1, dim_inner_expanded) + 0.1
    output_proj = model(sample_input_sequential, prev_hidden=prev_hidden_expanded)
    assert output_proj.shape == (config["batch_size"], config["seq_len_sequential"], config["dim"])

@pytest.mark.skipif(minLSTM is None, reason="minLSTM not imported")
def test_minlstm_forward_return_hidden(config, sample_input_parallel):
    """Test the return_next_prev_hidden flag."""
    model = minLSTM(dim=config["dim"], expansion_factor=config["expansion_factor"])
    dim_inner = int(config["dim"] * config["expansion_factor"])
    
    output, next_hidden = model(sample_input_parallel, return_next_prev_hidden=True)
    assert output.shape == (config["batch_size"], config["seq_len_parallel"], config["dim"])
    # LSTM hidden state matches output state dimensionally
    assert next_hidden.shape == (config["batch_size"], 1, dim_inner) # Should be internal state before projection

@pytest.mark.skipif(minLSTM is None, reason="minLSTM not imported")
def test_minlstm_forward_with_prev_hidden_parallel(config, sample_input_parallel, sample_prev_hidden):
    """Test parallel mode with an initial hidden state."""
    model = minLSTM(dim=config["dim"], expansion_factor=config["expansion_factor"])
    # Provide prev_hidden matching internal dim
    output = model(sample_input_parallel, prev_hidden=sample_prev_hidden)
    assert output.shape == (config["batch_size"], config["seq_len_parallel"], config["dim"])

@pytest.mark.skipif(minLSTM is None, reason="minLSTM not imported")
def test_minlstm_equivalence_parallel_sequential(config):
    """Test numerical equivalence between parallel and sequential modes."""
    model = minLSTM(dim=config["dim"], expansion_factor=config["expansion_factor"])
    model.eval() # Ensure consistent behavior
    
    inp = torch.randn(1, config["seq_len_parallel"], config["dim"]) # Use batch_size=1
    dim_inner = int(config["dim"] * config["expansion_factor"])

    # Parallel execution
    with torch.no_grad():
        parallel_out = model(inp)
        parallel_last_out = parallel_out[:, -1:]

    # Sequential execution
    prev_hidden = None
    sequential_last_out = None
    with torch.no_grad():
        for i in range(config["seq_len_parallel"]):
            token_input = inp[:, i:i+1, :] # Get current token
            current_out, next_hidden = model(token_input, prev_hidden, return_next_prev_hidden=True)
            sequential_last_out = current_out
            prev_hidden = next_hidden # Use the returned internal state
            
    assert sequential_last_out is not None, "Sequential loop did not run"
    # Compare the final output of the last step
    assert torch.allclose(parallel_last_out, sequential_last_out, atol=1e-5), \
        f"Max diff: {torch.max(torch.abs(parallel_last_out - sequential_last_out))}"

# Add more tests as needed:
# - Test gradients (requires loss and backward pass)
# - Test specific edge cases or numerical stability if concerns arise
# - Test different expansion_factor and proj_out combinations explicitly 