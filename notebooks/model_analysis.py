#!/usr/bin/env python3
"""
MNIST Model Analysis Script

This script helps analyze why the model might not be performing well
on handwritten digits despite good metrics during training.
Run this in a notebook environment or convert to a notebook.
"""

# Add parent directory to path so we can import from src
import sys
import os
sys.path.append('..')

# Import required libraries
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import io

# Import application modules
from src.inference import load_model
from src.data_utils import preprocess_image, preprocess_drawn_image
try:
    from src.model_architecture import SimpleCNN, MLP
except ImportError:
    print("Could not import model architectures. Please check the path.")

# 1. Load the model
print("Loading model...")
model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Add predict method to model if it doesn't have one
if not hasattr(model, 'predict'):
    def predict(self, tensor):
        """Add predict method to model for consistency with application"""
        self.eval()
        with torch.no_grad():
            tensor = tensor.to(device)
            outputs = self(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            return predicted.item(), confidence.item()
    
    # Monkey patch the model
    model.predict = predict.__get__(model)

# 2. Load test data
print("Loading MNIST test data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

# 3. Evaluate on test set
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    # Confusion matrix
    confusion_matrix = torch.zeros(10, 10)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update confusion matrix
            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    accuracy = 100 * correct / total
    return accuracy, confusion_matrix

print("Evaluating model on MNIST test set...")
accuracy, confusion_matrix = evaluate_model(model, test_loader)
print(f"Test Accuracy: {accuracy:.2f}%")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(range(10))
plt.yticks(range(10))

# Add text annotations
for i in range(10):
    for j in range(10):
        plt.text(j, i, f"{confusion_matrix[i, j]:.0f}", 
                 ha="center", va="center", 
                 color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black")

plt.savefig('notebooks/confusion_matrix.png')
plt.close()
print("Saved confusion matrix to notebooks/confusion_matrix.png")

# 4. Visualize predictions
def visualize_predictions(model, test_loader, num_samples=10):
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    fig = plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        ax = fig.add_subplot(2, 5, i+1)
        
        # Display the image
        img = images[i].squeeze().cpu().numpy()
        ax.imshow(img, cmap='gray')
        
        # Get the prediction
        with torch.no_grad():
            output = model(images[i:i+1].to(device))
            probs = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(probs, 1)
            confidence = probs[0][predicted].item()
        
        # Set the title with true label, prediction and confidence
        title = f"True: {labels[i]}, Pred: {predicted.item()}\nConf: {confidence:.2f}"
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('notebooks/predictions_visualization.png')
    plt.close()
    print("Saved predictions visualization to notebooks/predictions_visualization.png")

# Get a batch of test images
test_loader_viz = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)
visualize_predictions(model, test_loader_viz)

# 5. Print model architecture
print("\nModel Architecture:")
print(model)

# 6. Helper functions for analyzing your own digits
def process_image(image_path):
    """Process an image file and make a prediction"""
    # Load an image
    img = Image.open(image_path)
    
    # Convert to tensor using our preprocessing
    with open(image_path, 'rb') as f:
        tensor = preprocess_image(f)
    
    # Save the preprocessed image
    plt.figure(figsize=(5, 5))
    plt.imshow(tensor.squeeze().cpu().numpy(), cmap='gray')
    plt.title("After Preprocessing")
    plt.axis('off')
    preprocessed_path = image_path.rsplit('.', 1)[0] + '_preprocessed.png'
    plt.savefig(preprocessed_path)
    plt.close()
    
    # Make a prediction
    digit, confidence = model.predict(tensor)
    print(f"Predicted digit for {image_path}: {digit}, Confidence: {confidence:.4f}")
    print(f"Saved preprocessed image to {preprocessed_path}")
    
    return tensor

# 7. Helper for applying transformations to improve recognition
def apply_image_transformations(tensor, image_name):
    """Apply various transformations to see if they improve recognition"""
    # Define transformations to try
    def invert(x):
        return 1.0 - x

    def threshold(x, threshold=0.5):
        return (x > threshold).float()

    def increase_contrast(x, factor=2.0):
        mean = x.mean()
        return torch.clamp((x - mean) * factor + mean, 0, 1)

    transforms_list = [
        ('Original', lambda x: x),
        ('Inverted', invert),
        ('Thresholded', threshold),
        ('Contrast Enhanced', increase_contrast)
    ]
    
    fig, axes = plt.subplots(1, len(transforms_list), figsize=(15, 3))
    
    for i, (transform_name, transform_fn) in enumerate(transforms_list):
        transformed = transform_fn(tensor.clone())
        img = transformed.squeeze().cpu().numpy()
        
        # Get prediction
        digit, confidence = model.predict(transformed)
        
        # Display
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"{transform_name}\nPred: {digit}, Conf: {confidence:.2f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'notebooks/{image_name}_transformations.png')
    plt.close()
    print(f"Saved transformations to notebooks/{image_name}_transformations.png")

# 8. Compare data distributions
def compare_distributions(test_loader):
    """Compare the distribution of MNIST test images"""
    # Extract features from MNIST test images
    mnist_features = []
    mnist_labels = []
    
    for images, labels in test_loader:
        for i in range(len(images)):
            img = images[i].squeeze().cpu().numpy()
            # Simple features: mean and std of pixel values
            mean = np.mean(img)
            std = np.std(img)
            mnist_features.append([mean, std])
            mnist_labels.append(labels[i].item())
    
    # Plot the distribution of MNIST features
    mnist_features = np.array(mnist_features)
    plt.figure(figsize=(10, 6))
    plt.scatter(mnist_features[:, 0], mnist_features[:, 1], c=mnist_labels, 
                cmap='tab10', alpha=0.5, s=5)
    plt.colorbar(label='Digit')
    plt.xlabel('Mean Pixel Value')
    plt.ylabel('Std Dev of Pixel Values')
    plt.title('Distribution of MNIST Test Images')
    plt.savefig('notebooks/mnist_distribution.png')
    plt.close()
    print("Saved MNIST distribution plot to notebooks/mnist_distribution.png")

# Generate distribution comparison
compare_distributions(test_loader)

print("\n\nInstructions for analyzing your handwritten digits:")
print("1. Place your digit images in the notebooks/your_digits/ folder")
print("2. Run the following for each image:")
print("   tensor = process_image('notebooks/your_digits/your_image.png')")
print("   apply_image_transformations(tensor, 'your_image')")
print("\nPossible issues with handwritten digit recognition:")
print("1. Data Distribution Mismatch: Your handwriting style differs from MNIST")
print("2. Preprocessing Issues: Normalization or thresholding may need adjustment")
print("3. Model Architecture Limitations: The model might not be robust enough")
print("\nRecommendations:")
print("1. Augment Training Data: Train with rotations, shifts, and noise")
print("2. Improve Preprocessing: Adjust the preprocessing pipeline")
print("3. Try Transfer Learning: Use a model pre-trained on more diverse data")
print("4. Collect User Data: Build a dataset of your specific handwriting style")
print("5. Use Ensemble Methods: Combine multiple models or preprocessing techniques")

# Create a directory for user's own digits
os.makedirs('notebooks/your_digits', exist_ok=True)
print("\nCreated notebooks/your_digits/ directory for your samples") 