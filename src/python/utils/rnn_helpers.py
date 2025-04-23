import torch
import torch.nn.functional as F

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def log_g(x):
    """Calculates log(1 + exp(x)), also known as softplus."""
    return F.softplus(x)

def g(x):
    """Identity function, often used in simplified RNN formulations."""
    return x

# Replace the placeholder with the implementation from mingru.py
def heinsen_associative_scan_log(log_coeffs, log_values):
    """Associative scan implementation in log space.
    Based on mingru.py version, uses logcumsumexp.

    Args:
        log_coeffs (torch.Tensor): Logarithm of factors. Shape (B, S, D).
        log_values (torch.Tensor): Logarithm of values. Shape (B, S, D).

    Returns:
        torch.Tensor: Result of the scan (in normal space).
    """
    # Ensure dimensions are correct (B, S, D)
    # Allow padding if b includes initial state (S_b = S_a + 1)
    if log_coeffs.ndim != 3 or log_values.ndim != 3:
        raise ValueError(f"Input tensors must be 3D (B, S, D), got {log_coeffs.shape} and {log_values.shape}")
    if log_values.shape[1] == log_coeffs.shape[1] + 1:
         # log_coeffs needs initial padding if log_values has h_0
         # Pad with 0 = log(1)
         log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0), value=0)
    elif log_values.shape[1] != log_coeffs.shape[1]:
         raise ValueError(f"Dimension mismatch: log_values ({log_values.shape[1]}) should be equal to log_coeffs ({log_coeffs.shape[1]}) or log_coeffs+1")

    # Check if logcumsumexp exists (requires recent PyTorch versions)
    if not hasattr(torch, 'logcumsumexp'):
        raise RuntimeError("torch.logcumsumexp is required but not found. Please update PyTorch.")

    a_star = log_coeffs.cumsum(dim = 1)
    
    # Subtract a_star before logcumsumexp, add back after
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim = 1)
    
    log_h = a_star + log_h0_plus_b_star
    
    # Return result in normal space (exponentiate)
    result = log_h.exp()

    # Handle potential NaNs/Infs resulting from exp(large negative) or exp(large positive)
    # Clamp extreme values and replace NaNs with 0.
    # Define clamp bounds (adjust if necessary)
    clamp_min = -1e6 # Allow large negative numbers, but not -Inf
    clamp_max = 1e6  # Prevent Inf
    result = torch.nan_to_num(result, nan=0.0, posinf=clamp_max, neginf=clamp_min)
    result = torch.clamp(result, min=clamp_min, max=clamp_max)

    # If initial state was included, the result includes it at the start.
    # Slice to match original sequence length if necessary.
    # The calling function (minlstm) slices appropriately based on original seq_len.
    return result

    # --- End Placeholder ---

    # NOTE: A truly robust parallel log-domain scan for logaddexp 
    # might require a different algorithm, possibly involving matrix exponentiation 
    # or more sophisticated numerical techniques if stability over long sequences is critical.
    # The version above is a structural adaptation. 



# def heinsen_associative_scan_log(a, b):
#     """Associative scan implementation in log space, inspired by Wolf Heinsen's work.
#     Performs parallel prefix sum for operations like h_t = f_t * h_{t-1} + i_t * x_t in log domain.
#     Equivalent to log(h_t) = logaddexp(log(f_t) + log(h_{t-1}), log(i_t) + log(x_t))

#     Args:
#         a (torch.Tensor): Logarithm of the recurrent factor (e.g., log(f_t)). Shape (batch, seq_len, dim).
#         b (torch.Tensor): Logarithm of the incoming value (e.g., log(i_t) + log(x_t)). Shape (batch, seq_len, dim).
#                          Can also include the initial state log(h_0) prepended if needed.

#     Returns:
#         torch.Tensor: The result of the parallel scan in log space. Shape (batch, seq_len, dim).
#     """
#     # Ensure dimensions are correct (B, S, D)
#     if a.ndim != 3 or b.ndim != 3:
#         raise ValueError(f"Input tensors must be 3D (B, S, D), got {a.shape} and {b.shape}")
    
#     # Pad 'a' for the scan logic if b includes initial state
#     if b.shape[1] == a.shape[1] + 1:
#         # Pad 'a' with 0 at the beginning (log(1) = 0, identity for multiplication)
#         a = F.pad(a, (0, 0, 1, 0), value=0) # Pad sequence dim
#     elif b.shape[1] != a.shape[1]:
#          raise ValueError(f"Dimension mismatch: b ({b.shape[1]}) should be equal to a ({a.shape[1]}) or a+1")

#     # Check for stability: replace -inf with a very small number log-space
#     # log(0) = -inf, log(exp(-30)) is small enough for most practical purposes
#     # A clamp might be safer than replacing, depends on expected inputs
#     # a = torch.clamp(a, min=-30) # Optional clamp
#     # b = torch.clamp(b, min=-30) # Optional clamp

#     # Simple cumulative sum approach (may lack numerical stability of more complex scans)
#     # This version assumes the logaddexp structure: log(scan(f, i*x))
#     # More robust implementations might exist, see papers on parallel scans for RNNs.
    
#     # Perform the scan using logaddexp logic implicitly via cumsum
#     # This simplified version directly computes prefix sums, which might not be 
#     # exactly what's needed for the logaddexp associative operation. 
#     # A correct parallel logaddexp scan is more complex.
    
#     # --- Placeholder for a potentially more accurate scan --- 
#     # For now, using a simplified approach based on common associative scan patterns.
#     # The exact formula depends on how 'a' and 'b' map to the LSTM update.
#     # Let's use the structure similar to one often seen with GRU/RNN scans:
#     a_cumsum = a.cumsum(dim=1)
    
#     # Shift b for the scan relation: b_t relates to a_t * state_{t-1}
#     # We need b_{t-1} effectively. Pad b start with 0 (log(0) = -inf, additive identity in logaddexp)
#     # Using a large negative number instead of -inf for stability.
#     b_shifted = F.pad(b[:, :-1], (0, 0, 1, 0), value=-torch.inf) 

#     # Combine terms: log(a_t) + log(state_{t-1}) is roughly a_cumsum related.
#     # log(b_t) needs to be added correctly.
#     # This step needs careful derivation based on the exact associative op for logaddexp.
#     # Using the provided minGRU version as a template:
#     temp_b = b_shifted + a # This assumes b_shifted is log(h_{t-1})
#     temp_b_cumsum = temp_b.cumsum(dim = 1)
    
#     # This recombination resembles standard parallel prefix sums structure.
#     scanned_log = temp_b_cumsum - a_cumsum + b 

#     # If initial state h_0 was prepended to b, remove the first element from output
#     if scanned_log.shape[1] == a.shape[1]: # Check if b had initial state included
#         return scanned_log[:, 1:]
#     else:
#         return scanned_log
#     # --- End Placeholder ---

#     # NOTE: A truly robust parallel log-domain scan for logaddexp 
#     # might require a different algorithm, possibly involving matrix exponentiation 
#     # or more sophisticated numerical techniques if stability over long sequences is critical.
#     # The version above is a structural adaptation. 