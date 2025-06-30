# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY conda.yaml .
RUN pip install conda-env-to-requirements
RUN conda-env-to-requirements conda.yaml > requirements.txt
RUN pip install -r requirements.txt

# Copy application files
COPY modelling.py .
COPY iris_preprocessing.csv .

# Expose MLflow model serving port
EXPOSE 8080

# Default command for model serving
CMD ["mlflow", "models", "serve", "-m", "model", "-h", "0.0.0.0", "-p", "8080"]