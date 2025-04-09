#!/bin/bash
# Deployment script for MNIST project
# This script pulls the latest changes and rebuilds containers

# Exit on any error
set -e

echo "Starting deployment for MNIST project..."

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
  echo "Error: docker-compose.yml not found. Are you in the project root directory?"
  exit 1
fi

# Backup database
echo "Creating database backup..."
BACKUP_FILE="backup_$(date +%Y-%m-%d_%H-%M-%S).sql"
docker exec -t mnist_db pg_dump -U mnistuser mnistlogs > $BACKUP_FILE
echo "Database backed up to $BACKUP_FILE"

# Pull latest changes
echo "Pulling latest changes from git repository..."
git pull

# Rebuild and restart containers
echo "Rebuilding and restarting containers..."
docker-compose down
docker-compose build
docker-compose up -d

# Check if containers are running
echo "Checking container status..."
docker-compose ps

echo "Deployment complete! The updated application should be running."
echo "You can check the logs with: docker-compose logs -f" 