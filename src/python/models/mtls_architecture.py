import torch
import sys
import os
import torch.nn as nn # Import nn

# Import minLSTM at the top level
try:
    # Use relative import if minlstm is in the same directory
    from .minlstm import minLSTM
except ImportError:
     # Fallback to absolute import if run differently
     try:
         from python.models.minlstm import minLSTM
     except ImportError as e:
        raise ImportError(f"Failed to import minLSTM. Ensure 'src' is in Python path or models are installed. Error: {e}")

# Add project root to handle imports correctly if this file is run directly
# or imported from different locations.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add SRC directory to sys.path instead of project_root for direct execution
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

class MtlsModel(torch.nn.Module):
    """Model architecture using minLSTM for tabular data.

    Treats input features as a sequence of length 1.
    Uses minLSTM layers and a final linear layer for output.
    Includes dropout and configurable activation functions.
    """
    def __init__(self, input_dim: int, lstm_dim: int, output_dim: int = 1, lstm_expansion_factor: float = 1.0, num_layers: int = 1, dropout_rate: float = 0.0, activation_fn_name: str = "relu"):
        super().__init__()
        self.input_dim = input_dim
        self.lstm_dim = lstm_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Map activation function name to module
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU
        }
        if activation_fn_name not in activation_map:
            raise ValueError(f"Unsupported activation function: {activation_fn_name}")
        self.activation = activation_map[activation_fn_name]() # Instantiate the chosen activation

        # Input linear layer to project input_dim to lstm_dim
        self.input_fc = torch.nn.Linear(input_dim, lstm_dim)

        # Define dropout layer once to reuse
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate the actual output dimension from the LSTM layer
        lstm_output_dim = int(lstm_dim * lstm_expansion_factor)

        # Create a ModuleList to hold the LSTM stack and intermediate layers
        self.layers = nn.ModuleList()

        # Add the first minLSTM layer
        self.layers.append(minLSTM(dim=lstm_dim, expansion_factor=lstm_expansion_factor, proj_out=False))

        # Add subsequent layers (Intermediate Linear + Activation + Dropout + minLSTM)
        for _ in range(num_layers - 1):
            # Add projection layer if expansion factor is not 1.0
            if lstm_expansion_factor != 1.0:
                 self.layers.append(nn.Linear(lstm_output_dim, lstm_dim))
                 self.layers.append(self.activation) # Use chosen activation
                 self.layers.append(self.dropout) # Use the defined dropout layer

            # Add the next minLSTM layer
            # It expects input dim = lstm_dim
            self.layers.append(minLSTM(dim=lstm_dim, expansion_factor=lstm_expansion_factor, proj_out=False))

        # Output layer takes the output of the *last* LSTM layer (which is lstm_output_dim)
        self.output_fc = torch.nn.Linear(lstm_output_dim, output_dim)

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

        # Project input features to LSTM dimension
        projected_x = self.activation(self.input_fc(x)) # Use chosen activation
        projected_x = self.dropout(projected_x) # Apply dropout after initial projection

        # Pass through the stack of layers (LSTM, Linear, Activation, Dropout, LSTM, ...)
        current_x = projected_x
        prev_hidden = None # minLSTM needs explicit hidden state passing (unlike nn.GRU)

        for layer in self.layers:
            if isinstance(layer, minLSTM):
                # Pass previous hidden state if it exists
                current_x, prev_hidden = layer(current_x, prev_hidden=prev_hidden, return_next_prev_hidden=True)
            elif isinstance(layer, nn.Linear):
                # Reset prev_hidden before Linear projection
                prev_hidden = None
                # Linear expects (batch_size, features)
                current_x = current_x.squeeze(1) # (B, 1, F) -> (B, F)
                current_x = layer(current_x)     # (B, F) -> (B, F_new)
                # Add sequence dim back for next LSTM layer
                current_x = current_x.unsqueeze(1) # (B, F_new) -> (B, 1, F_new)
            else: # Activation or Dropout
                # Reset prev_hidden if activation/dropout is applied after Linear
                # Hidden state doesn't make sense for these stateless layers
                # prev_hidden = None # This might be too aggressive? Let's see.
                current_x = layer(current_x) # Apply Activation or Dropout

        # Output from the last layer in the ModuleList (should be a minLSTM)
        lstm_out = current_x

        # We only need the output from the last (only) time step
        last_step_out = lstm_out.squeeze(1)

        last_step_out = self.dropout(last_step_out) # Apply dropout before final layer

        # Pass through the final output layer
        output = self.output_fc(last_step_out)

        return output 