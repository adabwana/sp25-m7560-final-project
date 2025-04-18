import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Example usage (optional, for basic testing)
if __name__ == '__main__':
    # Define dimensions (replace with actual dimensions)
    input_dimension = 10
    hidden_dimension = 20
    output_dimension = 1 # Or number of classes

    # Create model instance
    model = SimpleNN(input_dimension, hidden_dimension, output_dimension)
    print("SimpleNN model created:")
    print(model)

    # Example input tensor (batch_size, input_dim)
    example_input = torch.randn(5, input_dimension)
    output = model(example_input)
    print("\nExample input shape:", example_input.shape)
    print("Example output shape:", output.shape)
