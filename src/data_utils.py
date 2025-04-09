"""
Data utility functions for processing MNIST-like images.
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import io

# Standard transformation for MNIST images
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def preprocess_image(image):
    """
    Preprocess an uploaded image for MNIST inference.
    
    Args:
        image: PIL Image or file-like object
        
    Returns:
        torch.Tensor: Tensor ready for model inference
    """
    if not isinstance(image, Image.Image):
        # If it's a file or file-like object, open it
        image = Image.open(image).convert('L')
    
    # Apply transformations
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor

def preprocess_drawn_image(image_data):
    """
    Preprocess image drawn on canvas for MNIST inference.
    
    Args:
        image_data: Data from drawable canvas
        
    Returns:
        torch.Tensor: Tensor ready for model inference
    """
    # Convert to PIL Image
    image = Image.fromarray(image_data.astype('uint8'))
    
    # Convert to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Invert colors if needed (MNIST has white digits on black background)
    image = Image.fromarray(255 - np.array(image))
    
    # Apply transformations
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor

def tensor_to_image(tensor):
    """
    Convert a tensor to a PIL Image.
    
    Args:
        tensor: PyTorch tensor with shape [1, 28, 28]
        
    Returns:
        PIL.Image: Grayscale image
    """
    # Denormalize
    tensor = tensor * 0.3081 + 0.1307
    
    # Convert to numpy and scale to 0-255
    image_np = tensor.squeeze().cpu().numpy() * 255
    image_np = image_np.astype(np.uint8)
    
    # Create PIL Image
    image = Image.fromarray(image_np, mode='L')
    return image

def image_to_bytes(image):
    """
    Convert PIL Image to bytes for database storage.
    
    Args:
        image: PIL Image
        
    Returns:
        bytes: Image in bytes format
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def bytes_to_image(image_bytes):
    """
    Convert bytes from database to PIL Image.
    
    Args:
        image_bytes: Bytes representation of image
        
    Returns:
        PIL.Image: Image object
    """
    img = Image.open(io.BytesIO(image_bytes))
    return img 