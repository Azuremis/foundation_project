FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ /app/
COPY model/ /app/model/

# Create data directory for MNIST dataset
RUN mkdir -p /app/data

# Make sure model directory exists
RUN mkdir -p /app/model/saved_model

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model/saved_model/mnist_model.pth
ENV DEBUG=False

# These environment variables will be overridden by docker-compose 
# from the .env file, but provide defaults for standalone usage
ENV DB_HOST=localhost
ENV DB_NAME=mnistlogs
ENV DB_USER=mnistuser 
ENV DB_PASSWORD=changeme

# Expose Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"] 