import unittest
import torch

# Adjust the import based on your project structure
from src.python.models.simple_nn import SimpleNN 

class TestSimpleNN(unittest.TestCase):

    def test_model_creation(self):
        """Test if the SimpleNN model can be created."""
        input_dim = 10
        hidden_dim = 20
        output_dim = 1
        try:
            model = SimpleNN(input_dim, hidden_dim, output_dim)
            self.assertIsNotNone(model, "Model creation failed.")
        except Exception as e:
            self.fail(f"Model creation raised an exception: {e}")

    def test_forward_pass(self):
        """Test the forward pass of the SimpleNN model."""
        input_dim = 10
        hidden_dim = 20
        output_dim = 1
        batch_size = 5

        model = SimpleNN(input_dim, hidden_dim, output_dim)
        # Create a dummy input tensor
        dummy_input = torch.randn(batch_size, input_dim)

        try:
            output = model(dummy_input)
            # Check if the output shape is correct (batch_size, output_dim)
            self.assertEqual(output.shape, (batch_size, output_dim), 
                             f"Output shape mismatch. Expected {(batch_size, output_dim)}, got {output.shape}")
        except Exception as e:
            self.fail(f"Forward pass raised an exception: {e}")

    # Add more tests as needed, e.g., for specific layer properties, edge cases

if __name__ == '__main__':
    unittest.main()
