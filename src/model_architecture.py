"""
Neural network architecture for MNIST digit classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for MNIST digit classification.
    Architecture:
        - Conv2d: 1 -> 32 channels, 3x3 kernel
        - ReLU activation
        - MaxPool2d: 2x2
        - Conv2d: 32 -> 64 channels, 3x3 kernel
        - ReLU activation
        - MaxPool2d: 2x2
        - Flatten
        - Linear: 1600 -> 128
        - ReLU activation
        - Linear: 128 -> 10 (output classes)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Input: [batch_size, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch_size, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch_size, 64, 7, 7]
        x = torch.flatten(x, 1)               # -> [batch_size, 64*7*7]
        x = F.relu(self.fc1(x))               # -> [batch_size, 128]
        x = self.fc2(x)                       # -> [batch_size, 10]
        return x
    
    def predict(self, x):
        """
        Make a prediction with confidence scores.
        
        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return predicted_class, confidence


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron for MNIST digit classification.
    Architecture:
        - Flatten
        - Linear: 784 -> 128
        - ReLU activation
        - Linear: 128 -> 64
        - ReLU activation
        - Linear: 64 -> 10 (output classes)
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        # Input: [batch_size, 1, 28, 28]
        x = self.flatten(x)                  # -> [batch_size, 784]
        x = F.relu(self.fc1(x))              # -> [batch_size, 128]
        x = F.relu(self.fc2(x))              # -> [batch_size, 64]
        x = self.fc3(x)                      # -> [batch_size, 10]
        return x
    
    def predict(self, x):
        """
        Make a prediction with confidence scores.
        
        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return predicted_class, confidence 