# Dockerfile
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

# Expose MLflow model serving port
EXPOSE 8080

# Default command for model serving
CMD ["mlflow", "models", "serve", "-m", "model", "-h", "0.0.0.0", "-p", "8080"]