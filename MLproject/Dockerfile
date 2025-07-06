# Dockerfile for Diabetes Classification Model
# Author: Alpian Khairi

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create model directory if it doesn't exist
RUN mkdir -p model

# Copy the trained diabetes model
COPY diabetes_model_ci.pkl model/model.pkl

# Expose MLflow model serving port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command for model serving
CMD ["mlflow", "models", "serve", "-m", "model", "-h", "0.0.0.0", "-p", "8080"]