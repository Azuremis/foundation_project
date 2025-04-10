"""
Inference utilities for MNIST digit classification.
"""
import os
import torch
from data_utils import preprocess_image, preprocess_drawn_image
from db.db_utils import log_prediction
from dotenv import load_dotenv
from model_architecture import SimpleCNN, MLP

# Load environment variables
load_dotenv()

def load_model(model_path=None):
    """
    Load the trained model from disk.
    
    Args:
        model_path: Path to the saved model file (optional)
        
    Returns:
        model: Loaded PyTorch model
    """
    # Use provided path, environment variable, or default
    if model_path is None:
        model_path = os.environ.get('MODEL_PATH', '../model/saved_model/mnist_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Determine model type
    model_dir = os.path.dirname(model_path)
    model_info_path = os.path.join(model_dir, 'model_info.txt')
    
    # Default to CNN if model info not available
    model_type = 'cnn'
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_type = f.read().strip()
    
    # Create the model instance based on type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == 'cnn':
        model = SimpleCNN().to(device)
    else:
        model = MLP().to(device)
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def predict_from_upload(model, image_file, save_to_db=True):
    """
    Make a prediction on an uploaded image file.
    
    Args:
        model: Trained PyTorch model
        image_file: Uploaded file object
        save_to_db: Whether to log the prediction to the database
        
    Returns:
        tuple: (predicted_digit, confidence)
    """
    # Preprocess the image
    tensor = preprocess_image(image_file)
    
    # Make prediction
    digit, confidence = model.predict(tensor)
    
    # Log to database if requested
    if save_to_db:
        # We don't have user label at this point
        log_prediction(
            predicted_digit=digit,
            confidence=confidence
        )
    
    return digit, confidence

def predict_from_canvas(model, canvas_data, save_to_db=True):
    """
    Make a prediction on an image drawn on canvas.
    
    Args:
        model: Trained PyTorch model
        canvas_data: Data from the drawable canvas
        save_to_db: Whether to log the prediction to the database
        
    Returns:
        tuple: (predicted_digit, confidence)
    """
    # Preprocess the drawn image
    tensor = preprocess_drawn_image(canvas_data)
    
    # Make prediction
    digit, confidence = model.predict(tensor)
    
    # Log to database if requested
    if save_to_db:
        # We don't have user label at this point
        log_prediction(
            predicted_digit=digit,
            confidence=confidence
        )
    
    return digit, confidence

def update_prediction_with_user_label(predicted_digit, user_label, confidence=None, image_data=None):
    """
    Update a prediction in the database with user feedback.
    
    Args:
        predicted_digit: The model's prediction
        user_label: User's label (ground truth)
        confidence: Confidence score (optional)
        image_data: Binary image data (optional)
        
    Returns:
        bool: Success status
    """
    return log_prediction(
        predicted_digit=predicted_digit,
        user_label=user_label,
        confidence=confidence,
        image_data=image_data
    ) 