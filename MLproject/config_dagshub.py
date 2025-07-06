"""
DagsHub Configuration for MLflow Tracking
Author: alvian2022
"""

import os
import dagshub
import mlflow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def setup_dagshub_tracking():
    """Setup DagsHub as MLflow tracking server"""
    
    # DagsHub repository details - GANTI DENGAN REPO ANDA
    dagshub_repo_owner = "alvian2022"
    dagshub_repo_name = "diabetes-classification"
    
    # Initialize DagsHub
    dagshub.init(repo_owner=dagshub_repo_owner, 
                 repo_name=dagshub_repo_name, 
                 mlflow=True)
    
    # Set MLflow tracking URI
    mlflow_tracking_uri = f"https://dagshub.com/{dagshub_repo_owner}/{dagshub_repo_name}.mlflow"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    print(f"✅ DagsHub tracking setup completed!")
    print(f"📊 MLflow Tracking URI: {mlflow_tracking_uri}")
    print(f"🔗 Repository: https://dagshub.com/{dagshub_repo_owner}/{dagshub_repo_name}")
    
    return mlflow_tracking_uri

def setup_dagshub_auth():
    """Setup DagsHub authentication using environment variables"""
    
    # Check if token is in environment
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    
    if not dagshub_token:
        print("⚠️  DAGSHUB_TOKEN not found in environment variables")
        print("Please set your DagsHub token:")
        print("export DAGSHUB_TOKEN=your_token_here")
        return False
    
    # Set authentication for MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = "alvian2022"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    
    print("✅ DagsHub authentication setup completed!")
    return True

if __name__ == "__main__":
    print("Testing DagsHub configuration...")
    setup_dagshub_auth()
    setup_dagshub_tracking()