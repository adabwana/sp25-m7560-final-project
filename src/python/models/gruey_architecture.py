import torch
import sys
import os

# Add project root to handle imports correctly if this file is run directly
# or imported from different locations.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class GrueyModel(torch.nn.Module):
    """Model architecture using minGRU for tabular data.
    
    Treats input features as a sequence of length 1.
    Uses minGRU to process the input and a final linear layer for output.
    """
    def __init__(self, input_dim: int, gru_dim: int, output_dim: int = 1, gru_expansion_factor: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.gru_dim = gru_dim
        self.output_dim = output_dim
        
        # Import minGRU here, assuming src/python/models is findable via sys.path
        # The path addition above should help ensure this works.
        try:
            # Use relative import if mingru is in the same directory
            from .mingru import minGRU 
        except ImportError:
             # Fallback to absolute import if run differently (e.g. from root with src in path)
             try:
                 from python.models.mingru import minGRU
             except ImportError as e:
                raise ImportError(f"Failed to import minGRU. Ensure 'src' is in Python path or models are installed. Error: {e}")

        # Using expansion factor 1 and no projection in minGRU by default for simplicity
        # The input dimension to minGRU must match gru_dim if no input projection is used.
        # Let's add an input linear layer to project input_dim to gru_dim.
        self.input_fc = torch.nn.Linear(input_dim, gru_dim)
        
        # minGRU layer - input dim must match its internal dim (gru_dim here)
        self.gru = minGRU(dim=gru_dim, expansion_factor=gru_expansion_factor, proj_out=False) # proj_out=False keeps output dim = gru_dim
        
        # Output layer takes the GRU output (gru_dim) and maps to output_dim
        self.output_fc = torch.nn.Linear(gru_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Add sequence dimension: (batch_size, num_features) -> (batch_size, 1, num_features)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3 or x.shape[1] != 1:
             raise ValueError(f"Expected input shape (batch_size, num_features) or (batch_size, 1, num_features), got {x.shape}")

        # Project input features to GRU dimension
        # Input to input_fc: (batch_size, 1, input_dim)
        # Output from input_fc: (batch_size, 1, gru_dim)
        projected_x = torch.relu(self.input_fc(x)) # Apply activation
        
        # Pass through minGRU
        # Input to gru: (batch_size, 1, gru_dim)
        # Since return_next_prev_hidden=False (default), output is the sequence output
        # Output from gru: (batch_size, 1, gru_dim) because proj_out=False
        gru_out = self.gru(projected_x)
        
        # We only need the output from the last (only) time step
        # Squeeze the sequence dimension: (batch_size, 1, gru_dim) -> (batch_size, gru_dim)
        last_step_out = gru_out.squeeze(1)
        
        # Pass through the final output layer
        # Input to output_fc: (batch_size, gru_dim)
        # Output from output_fc: (batch_size, output_dim)
        output = self.output_fc(last_step_out)
        
        return output 