import torch
import torch.nn as nn
from typing import List, Union

class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Union[List[int], int], output_dim: int, activation: Union[str, List[str]] = 'ReLU', dropout: float = 0.0):
        super(SimpleNN, self).__init__()
        # Allow hidden_dims to be int or list of ints
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.hidden_dims = hidden_dims
        # Allow activation to be str or list of str
        if isinstance(activation, str):
            activations = [activation] * len(hidden_dims)
        else:
            activations = activation
        assert len(activations) == len(hidden_dims), "Length of activations must match number of hidden layers"
        self.activations = activations
        self.dropout = dropout

        layers = []
        prev_dim = input_dim
        for h_dim, act in zip(hidden_dims, activations):
            layers.append(nn.Linear(prev_dim, h_dim))
            act_layer = getattr(nn, act)() if hasattr(nn, act) else nn.ReLU()
            layers.append(act_layer)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Example usage (optional, for basic testing)
if __name__ == '__main__':
    # Define dimensions (replace with actual dimensions)
    input_dimension = 10
    hidden_dimensions = [20, 16, 8]  # Example: 3 hidden layers
    output_dimension = 1 # Or number of classes
    activations = ['ReLU', 'Tanh', 'ReLU']
    dropout_rate = 0.5

    # Create model instance
    model = SimpleNN(input_dimension, hidden_dimensions, output_dimension, activations, dropout_rate)
    print("SimpleNN model created:")
    print(model)

    # Example input tensor (batch_size, input_dim)
    example_input = torch.randn(5, input_dimension)
    output = model(example_input)
    print("\nExample input shape:", example_input.shape)
    print("Example output shape:", output.shape)
