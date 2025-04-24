import torch
import torch.nn as nn
import math
import os
import sys

# Add project root for potential utility imports if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TabularTransformerModel(nn.Module):
    """Transformer model adapted for tabular data regression."""
    def __init__(self, 
                 input_dim: int,        # Number of input features
                 d_model: int,          # Embedding dimension (must be divisible by nhead)
                 nhead: int,            # Number of attention heads
                 d_hid: int,            # Dimension of the feedforward network model
                 nlayers: int,          # Number of nn.TransformerEncoderLayer layers
                 output_dim: int = 1,   # Output dimension (1 for regression)
                 dropout: float = 0.2): # Dropout rate
        super().__init__()
        self.model_type = 'Transformer'
        
        # 1. Input Projection: Linear layer to project input features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        # 4. Output Layer: Maps the pooled output to the target dimension
        self.output_fc = nn.Linear(d_model, output_dim)

        self.d_model = d_model
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_proj.weight.data.uniform_(-initrange, initrange)
        self.input_proj.bias.data.zero_()
        self.output_fc.weight.data.uniform_(-initrange, initrange)
        self.output_fc.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape [batch_size, num_features]

        Returns:
            Output tensor, shape [batch_size, output_dim]
        """
        # 1. Project input features to d_model
        # Input: [batch_size, num_features]
        # Output: [batch_size, d_model]
        x = self.input_proj(x) * math.sqrt(self.d_model)
        
        # TransformerEncoderLayer expects input shape [batch_size, seq_len, features] 
        # or [seq_len, batch_size, features] depending on batch_first.
        # We treat the features as a sequence of length 1 for tabular data.
        # Add sequence dimension: [batch_size, d_model] -> [batch_size, 1, d_model]
        x = x.unsqueeze(1) 
        
        # Apply positional encoding (expects [seq_len, batch_size, features] by default)
        # Transpose for PositionalEncoding: [batch_size, 1, d_model] -> [1, batch_size, d_model]
        # x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1) # Apply PE if needed
        # Note: Positional encoding might not be very effective for seq_len=1. 
        # Let's skip PE for simplicity in this tabular adaptation.
        
        # 2. Pass through Transformer Encoder 
        # Input: [batch_size, 1, d_model]
        # Output: [batch_size, 1, d_model]
        transformer_output = self.transformer_encoder(x)
        
        # 3. Pooling: Use the output of the only sequence step
        # Squeeze the sequence dimension: [batch_size, 1, d_model] -> [batch_size, d_model]
        pooled_output = transformer_output.squeeze(1)
        
        # 4. Final Output Layer
        # Input: [batch_size, d_model]
        # Output: [batch_size, output_dim]
        output = self.output_fc(pooled_output)
        
        return output 