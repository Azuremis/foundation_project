# MNIST Digit Classifier

An interactive web application that allows users to draw digits (0-9) which are then classified using a PyTorch model trained on the MNIST dataset. The application records all predictions in a PostgreSQL database for performance tracking.

## Live Demo

**Access the live application:** [http://49.13.204.139](http://49.13.204.139)

## Features

- **Interactive Drawing Canvas**: Draw any digit (0-9) directly in your browser
- **Real-time Prediction**: Instantly see the model's prediction and confidence level
- **Feedback Collection**: Submit the correct digit to help improve the model's performance
- **Prediction History**: All predictions are logged with timestamps for analysis

## Technical Overview

This project demonstrates a complete machine learning application deployment with:

- **Machine Learning**: PyTorch model trained on the MNIST dataset (99.2% test accuracy)
- **Front-End**: Streamlit web application with interactive drawing canvas
- **Database**: PostgreSQL for storing prediction data
- **Deployment**: Docker containerized application running on a Hetzner VPS

## Usage Instructions

1. Open the application in your browser
2. Use your mouse or touchscreen to draw a digit (0-9) on the canvas
3. Click "Predict" to see the model's classification result
4. View the predicted digit and confidence score
5. Optionally provide the correct digit if the prediction was incorrect
6. Previous predictions will be displayed in the history section

## Architecture

The application consists of three main components:

1. **Model Service**: Serves the trained PyTorch MNIST classifier
2. **Web Application**: Streamlit interface for user interaction
3. **Database**: PostgreSQL instance for logging and analytics

All components are containerized using Docker and orchestrated with Docker Compose.

## Database Schema

The PostgreSQL database stores the following information for each prediction:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Unique identifier for each prediction |
| timestamp | TIMESTAMP | When the prediction was made |
| predicted_digit | INTEGER | The digit predicted by the model (0-9) |
| confidence | FLOAT | Confidence score (0-1) of the prediction |
| true_digit | INTEGER | User-provided correct digit (if submitted) |
| image_data | BYTEA | Binary representation of the drawn digit |

## Technical Implementation

- **Model**: Convolutional Neural Network trained on MNIST dataset
- **Web Interface**: Streamlit with custom JavaScript for canvas drawing
- **API**: FastAPI endpoint for model inference
- **Database**: PostgreSQL with connection pooling
- **Deployment**: Containerized with Docker on Hetzner VPS

## Local Development

If you want to run the application locally:

```bash
# Clone the repository
git clone https://github.com/yourusername/foundation_project.git
cd foundation_project

# Create a .env file with the necessary environment variables
cp .env.example .env
# Edit .env with your settings
nano .env

# Start the application using Docker Compose
docker-compose up
```

Access the application at http://localhost:8501 for local development or at http://49.13.204.139:8501 for the deployed version.

## Environment Variables

The application uses environment variables for configuration. Create a `.env` file in the root directory with these variables:

```
# Database credentials
DB_HOST=db
DB_NAME=mnistlogs
DB_USER=mnistuser
DB_PASSWORD=your_secure_password

# Application settings
DEBUG=False
MODEL_PATH=model/saved_model/mnist_model.pth

# Docker settings
POSTGRES_DB=mnistlogs
POSTGRES_USER=mnistuser
POSTGRES_PASSWORD=your_secure_password
```

For production environments, make sure to:
1. Use strong, unique passwords
2. Backup your `.env` file securely
3. Add `.env` to your `.gitignore` file to prevent committing credentials to version control

## Repository Structure

```
.
├── model/              # Model definition and serving
│   └── saved_model/    # Trained model storage
├── src/                # Source code
│   ├── app.py          # Main Streamlit application 
│   ├── hello_app.py    # Simple test application
│   ├── train_mnist.py  # Model training script
│   ├── db/             # Database utilities
│   └── ...
├── scripts/            # Deployment & automation scripts
├── tests/              # Unit and integration tests
├── docker-compose.yml  # Container orchestration
├── Dockerfile          # Container definition
├── .env                # Environment variables (not in git)
└── README.md           # This documentation
```

## About the Model

The model architecture is a convolutional neural network with:
- 2 convolutional layers (32 and 64 filters)
- Max pooling layers
- Dropout for regularization
- 2 fully connected layers

Training metrics:
- Training accuracy: 99.8%
- Validation accuracy: 99.2%
- Test accuracy: 99.2%

## License

MIT