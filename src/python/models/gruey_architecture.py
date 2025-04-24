import torch
import sys
import os
import torch.nn as nn # Import nn

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
    def __init__(self, input_dim: int, gru_dim: int, output_dim: int = 1, gru_expansion_factor: float = 1.0, num_layers: int = 1, dropout_rate: float = 0.0, activation_fn_name: str = "relu"):
        super().__init__()
        self.input_dim = input_dim
        self.gru_dim = gru_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate # Store dropout rate
        
        # Map activation function name to module
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU
        }
        if activation_fn_name not in activation_map:
            raise ValueError(f"Unsupported activation function: {activation_fn_name}")
        self.activation = activation_map[activation_fn_name]() # Instantiate the chosen activation

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
        
        # Define dropout layer once to reuse
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the actual output dimension from the GRU layer
        gru_output_dim = int(gru_dim * gru_expansion_factor)

        # Create a ModuleList to hold the GRU stack and intermediate layers
        self.layers = nn.ModuleList()
        
        # Add the first minGRU layer
        self.layers.append(minGRU(dim=gru_dim, expansion_factor=gru_expansion_factor, proj_out=False))

        # Add subsequent layers (Intermediate Linear + Activation + Dropout + minGRU)
        for _ in range(num_layers - 1):
            # Add projection layer if expansion factor is not 1.0
            if gru_expansion_factor != 1.0:
                 self.layers.append(nn.Linear(gru_output_dim, gru_dim))
                 self.layers.append(self.activation) # Use chosen activation
                 self.layers.append(self.dropout) # Use the defined dropout layer
            # Else (if expansion is 1.0), output dim = input dim, no projection needed
            
            # Add the next minGRU layer
            # It expects input dim = gru_dim (either directly from previous GRU or after projection)
            self.layers.append(minGRU(dim=gru_dim, expansion_factor=gru_expansion_factor, proj_out=False))

        # Output layer takes the output of the *last* GRU layer (which is gru_output_dim)
        self.output_fc = torch.nn.Linear(gru_output_dim, output_dim)

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
        projected_x = self.activation(self.input_fc(x)) # Use chosen activation
        projected_x = self.dropout(projected_x) # Apply dropout after initial projection
        
        # Pass through the stack of layers (GRU, Linear, Activation, Dropout, GRU, ...)
        # Input to first GRU: (batch_size, 1, gru_dim)
        current_x = projected_x
        for layer in self.layers:
            # Need to handle shape for Linear layer if present
            if isinstance(layer, nn.Linear):
                # Linear expects (batch_size, features), GRU outputs (batch_size, seq_len, features)
                # We assume seq_len is 1 here
                current_x = current_x.squeeze(1) # (B, 1, F) -> (B, F)
                current_x = layer(current_x)     # (B, F) -> (B, F_new)
                # Add sequence dim back for next GRU layer if it exists
                current_x = current_x.unsqueeze(1) # (B, F_new) -> (B, 1, F_new)
            else: # It's a GRU or Activation or Dropout
                current_x = layer(current_x) # Apply GRU or Activation or Dropout
                # Output shape from GRU: (batch_size, 1, gru_output_dim)
                # Output shape from Activation/Dropout: same as input

        # Output from the last layer in the ModuleList (should be a minGRU)
        gru_out = current_x
        
        # We only need the output from the last (only) time step
        # Squeeze the sequence dimension: (batch_size, 1, gru_output_dim) -> (batch_size, gru_output_dim)
        last_step_out = gru_out.squeeze(1)
        
        last_step_out = self.dropout(last_step_out) # Apply dropout before final layer
        
        # Pass through the final output layer
        # Input to output_fc: (batch_size, gru_output_dim)
        # Output from output_fc: (batch_size, output_dim)
        output = self.output_fc(last_step_out)
        
        return output 