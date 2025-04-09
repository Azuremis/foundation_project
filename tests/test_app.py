"""
Tests for the Streamlit application.

Note: Testing Streamlit apps is challenging because they're designed to run 
interactively. These are simple tests to verify imports and basic functionality.
"""
import unittest
import sys
import os
import io
import torch
import numpy as np
from PIL import Image

# Add the src directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock streamlit to avoid errors when importing app modules
class MockSt:
    """Simple mock class for streamlit functions."""
    @staticmethod
    def cache_resource(func):
        return func
    
    @staticmethod
    def set_page_config(*args, **kwargs):
        pass

# Mock the streamlit module
sys.modules['streamlit'] = MockSt()
sys.modules['streamlit_drawable_canvas'] = MockSt()

# Now import modules from the app
from data_utils import (
    preprocess_image, preprocess_drawn_image, 
    tensor_to_image, image_to_bytes, bytes_to_image
)

class TestDataUtils(unittest.TestCase):
    """Test cases for data utilities."""
    
    def test_image_conversions(self):
        """Test image conversion functions."""
        # Create a simple 28x28 test image
        test_array = np.zeros((28, 28), dtype=np.uint8)
        test_array[10:20, 10:20] = 255  # White square in the middle
        test_image = Image.fromarray(test_array, mode='L')
        
        # Test image to bytes conversion
        img_bytes = image_to_bytes(test_image)
        self.assertIsInstance(img_bytes, bytes)
        
        # Test bytes to image conversion
        recovered_image = bytes_to_image(img_bytes)
        self.assertIsInstance(recovered_image, Image.Image)
        
        # Basic check that the images are similar
        # Convert both to arrays and check a few pixels
        recovered_array = np.array(recovered_image.convert('L'))
        self.assertEqual(recovered_array[15, 15], 255)
        self.assertEqual(recovered_array[5, 5], 0)
    
    def test_tensor_to_image(self):
        """Test tensor to image conversion."""
        # Create a normalized test tensor (simulating MNIST)
        test_tensor = torch.zeros(1, 28, 28)
        test_tensor[:, 10:20, 10:20] = 1.0  # Set a square to 1.0 (white after denormalization)
        
        # Convert to image
        img = tensor_to_image(test_tensor)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (28, 28))
        
        # Check a few pixels
        img_array = np.array(img)
        self.assertTrue(img_array[15, 15] > 200)  # Should be close to white
        self.assertTrue(img_array[5, 5] < 50)     # Should be close to black

if __name__ == '__main__':
    unittest.main() 