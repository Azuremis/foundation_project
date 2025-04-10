#!/bin/bash
# Setup script for MNIST project on a fresh VPS
# This script should be run as a non-root user with sudo privileges

# Exit on any error
set -e

echo "Starting setup for MNIST project..."

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install required packages
echo "Installing required packages..."
sudo apt-get install -y git docker.io docker-compose

# Add current user to the docker group
echo "Adding user to docker group..."
sudo usermod -aG docker $USER
echo "NOTE: You may need to log out and back in for docker group changes to take effect."

# Check Docker and Docker Compose versions
echo "Checking Docker version..."
docker --version
echo "Checking Docker Compose version..."
docker-compose --version

# Clone the repository (if not already cloned)
if [ ! -d "mnist_project" ]; then
  echo "Cloning the repository..."
  read -p "Enter the GitHub repository URL: " REPO_URL
  git clone $REPO_URL mnist_project
  cd mnist_project
else
  echo "Repository already exists. Pulling latest changes..."
  cd mnist_project
  git pull
fi

# Make sure the correct directories exist
echo "Ensuring directories exist..."
mkdir -p model/saved_model

# Create a secure .env file if it doesn't exist
if [ ! -f ".env" ]; then
  echo "Creating .env file..."
  # Generate random password for database
  DB_PASSWORD=$(openssl rand -base64 12)
  
  cat > .env << EOF
# Database credentials
DB_HOST=db
DB_NAME=mnistlogs
DB_USER=mnistuser
DB_PASSWORD=${DB_PASSWORD}

# Application settings
DEBUG=False
MODEL_PATH=model/saved_model/mnist_model.pth

# Docker settings
POSTGRES_DB=mnistlogs
POSTGRES_USER=mnistuser
POSTGRES_PASSWORD=${DB_PASSWORD}
EOF
  
  echo ".env file created with secure random password!"
  echo "⚠️ Make sure to backup your .env file in a secure location."
else
  echo ".env file already exists. Keeping existing configuration."
fi

# Build and start containers
echo "Building and starting Docker containers..."
docker-compose build
docker-compose up -d

# Check if containers are running
echo "Checking container status..."
docker-compose ps

echo "Setup complete! The application should be running at http://$(hostname -I | awk '{print $1}')/"
echo "You can check the logs with: docker-compose logs -f" 