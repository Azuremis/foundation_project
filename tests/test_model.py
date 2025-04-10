"""
Tests for the MNIST model architecture and training.
"""
import unittest
import torch
import sys
import os

# Add the src directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model_architecture import SimpleCNN, MLP

class TestModelArchitecture(unittest.TestCase):
    """Test cases for model architectures."""
    
    def test_cnn_forward(self):
        """Test that CNN model can perform a forward pass."""
        model = SimpleCNN()
        # Create a fake batch of MNIST images (1x28x28)
        x = torch.randn(10, 1, 28, 28)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (10, 10))
    
    def test_mlp_forward(self):
        """Test that MLP model can perform a forward pass."""
        model = MLP()
        # Create a fake batch of MNIST images (1x28x28)
        x = torch.randn(10, 1, 28, 28)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (10, 10))
    
    def test_predict_method(self):
        """Test the prediction methods of both models."""
        cnn = SimpleCNN()
        mlp = MLP()
        
        # Create a single fake MNIST image
        x = torch.randn(1, 1, 28, 28)
        
        # Get predictions
        cnn_digit, cnn_confidence = cnn.predict(x)
        mlp_digit, mlp_confidence = mlp.predict(x)
        
        # Check types
        self.assertIsInstance(cnn_digit, int)
        self.assertIsInstance(cnn_confidence, float)
        self.assertIsInstance(mlp_digit, int)
        self.assertIsInstance(mlp_confidence, float)
        
        # Check ranges
        self.assertTrue(0 <= cnn_digit <= 9)
        self.assertTrue(0.0 <= cnn_confidence <= 1.0)
        self.assertTrue(0 <= mlp_digit <= 9)
        self.assertTrue(0.0 <= mlp_confidence <= 1.0)

if __name__ == '__main__':
    unittest.main() 