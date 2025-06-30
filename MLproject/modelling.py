# modelling.py (for MLProject)
"""
MLProject Entry Point for Model Training
Author: alpian_khairi_C1BO
Description: Configurable model training for MLflow Projects
"""

import sys
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Load preprocessed data"""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_model(data_path, model_type="random_forest", use_tuning=True):
    """Train model based on configuration"""
    # MLflow Project automatically logs the entry point parameters
    # We only log additional parameters that aren't already logged
    
    # Log additional parameters (not duplicating entry point parameters)
    mlflow.log_param("author", "alpian_khairi_C1BO")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    # Configure model
    if model_type == "random_forest":
        if use_tuning:
            # Hyperparameter tuning
            param_grid = {
               'n_estimators': [50, 100, 200],
               'max_depth': [3, 5, 7, None],
               'min_samples_split': [2, 5, 10]
            }
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate and log metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    logger.info(f"Model training completed - Accuracy: {accuracy:.4f}")
    
    return model

def main():
    """Main function for MLProject entry point"""
    parser = argparse.ArgumentParser(description='Train Iris Classification Model')
    parser.add_argument('data_path', type=str, help='Path to preprocessed data')
    parser.add_argument('model_type', type=str, default='random_forest', 
                       help='Type of model to train')
    parser.add_argument('use_tuning', type=str, default='true', 
                       help='Whether to use hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Convert string boolean to actual boolean
    use_tuning = args.use_tuning.lower() in ['true', '1', 'yes', 'on']
    
    print(f"Starting MLProject training...")
    print(f"Author: alpian_khairi_C1BO")
    print(f"Data path: {args.data_path}")
    print(f"Model type: {args.model_type}")
    print(f"Use tuning: {use_tuning}")
    
    # Don't set experiment here - MLflow Project will handle it
    # mlflow.set_experiment("iris_mlproject_alpian_khairi")
    
    model = train_model(args.data_path, args.model_type, use_tuning)
    print("MLProject training completed successfully!")

if __name__ == "__main__":
    main()