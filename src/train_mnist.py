"""
Training script for MNIST digit classifier.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dotenv import load_dotenv

# Import model architecture(s)
from model_architecture import SimpleCNN, MLP

# Load environment variables
load_dotenv()

# Debug mode from environment variable
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')

def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        device: Device to train on (cpu or cuda)
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        epoch: Current epoch number
    
    Returns:
        float: Average loss over the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Print statistics every 100 batches or more frequently in debug mode
        log_interval = 10 if DEBUG else 100
        if batch_idx % log_interval == log_interval - 1:
            print(f'Epoch: {epoch}, Batch: {batch_idx+1}, Loss: {running_loss/log_interval:.3f}, Accuracy: {100*correct/total:.2f}%')
            running_loss = 0.0
    
    return running_loss / len(train_loader)

def validate(model, device, test_loader, criterion):
    """
    Validate the model on test data.
    
    Args:
        model: The neural network model
        device: Device to evaluate on (cpu or cuda)
        test_loader: DataLoader for test data
        criterion: Loss function
    
    Returns:
        tuple: (average loss, accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return test_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    parser.add_argument('--model-type', type=str, choices=['cnn', 'mlp'], default='cnn',
                        help='Type of model to train (cnn or mlp)')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the trained model')
    parser.add_argument('--output-dir', type=str, 
                        default=os.environ.get('MODEL_PATH', '../model/saved_model').rsplit('/', 1)[0],
                        help='directory to save the model')
    parser.add_argument('--debug', action='store_true', default=DEBUG,
                        help='enable debug mode with more verbose output')
    
    args = parser.parse_args()
    
    # Set debug mode
    global DEBUG
    DEBUG = args.debug
    
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Check if CUDA is available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    # Initialize model based on user choice
    if args.model_type == 'cnn':
        model = SimpleCNN().to(device)
    else:
        model = MLP().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, accuracy = validate(model, device, test_loader, criterion)
        
        # Save model if it has the best accuracy so far
        if accuracy > best_accuracy and args.save_model:
            best_accuracy = accuracy
            
            # Ensure directory exists
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save model
            torch.save(model, os.path.join(args.output_dir, 'mnist_model.pth'))
            print(f'Model saved with accuracy: {accuracy:.2f}%')
    
    print("Training completed!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")

if __name__ == '__main__':
    main() 