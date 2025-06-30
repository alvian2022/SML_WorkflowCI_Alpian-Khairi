"""
MLProject modelling.py - IMPROVED VERSION
CI/CD Model Training for MLflow Project with Better Local Artifact Management
Author: alpian_khairi_C1BO
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import argparse
import os
import sys
import logging
import time
import shutil
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import dagshub
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLProjectTrainer:
    """MLProject model trainer with dual tracking (remote + local artifacts)"""
    
    def __init__(self, args):
        """Initialize trainer with arguments"""
        self.args = args
        self.model = None
        self.target_names = ['Setosa', 'Versicolor', 'Virginica']
        self.use_remote_tracking = False
        self.local_artifacts_dir = "model_artifacts"
        self.setup_local_artifacts_dir()
        self.setup_mlflow()
    
    def setup_local_artifacts_dir(self):
        """Setup local artifacts directory"""
        os.makedirs(self.local_artifacts_dir, exist_ok=True)
        logger.info(f"Local artifacts directory: {self.local_artifacts_dir}")
    
    def test_dagshub_connection(self, timeout=10):
        """Test connection to DagsHub with retry mechanism"""
        try:
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            dagshub_url = "https://dagshub.com/alvian2022/iris-classification"
            response = session.get(dagshub_url, timeout=timeout)
            
            if response.status_code == 200:
                logger.info("✅ DagsHub connection test successful")
                return True
            else:
                logger.warning(f"⚠️ DagsHub returned status code: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ DagsHub connection test failed: {str(e)}")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Unexpected error testing DagsHub connection: {str(e)}")
            return False
    
    def setup_mlflow(self):
        """Setup MLflow tracking with dual storage (remote + local)"""
        try:
            dagshub_token = os.getenv('DAGSHUB_TOKEN')
            
            if dagshub_token and self.test_dagshub_connection():
                try:
                    # DagsHub configuration
                    dagshub_repo_owner = "alvian2022"
                    dagshub_repo_name = "iris-classification"
                    
                    logger.info("Initializing DagsHub connection...")
                    
                    # Set timeouts for MLflow
                    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "30"
                    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "3"
                    
                    dagshub.init(
                        repo_owner=dagshub_repo_owner, 
                        repo_name=dagshub_repo_name, 
                        mlflow=True
                    )
                    
                    mlflow_tracking_uri = f"https://dagshub.com/{dagshub_repo_owner}/{dagshub_repo_name}.mlflow"
                    mlflow.set_tracking_uri(mlflow_tracking_uri)
                    
                    # Set authentication
                    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_repo_owner
                    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
                    
                    # Test MLflow connection
                    try:
                        mlflow.get_tracking_uri()
                        self.use_remote_tracking = True
                        logger.info(f"✅ DagsHub tracking setup successful: {mlflow_tracking_uri}")
                        logger.info("📊 Metrics and models will be logged to DagsHub")
                        logger.info("💾 Artifacts will also be saved locally")
                    except Exception as e:
                        logger.warning(f"⚠️ MLflow connection test failed: {str(e)}")
                        raise e
                        
                except Exception as e:
                    logger.warning(f"⚠️ DagsHub setup failed: {str(e)}")
                    logger.info("Falling back to local MLflow tracking...")
                    self.setup_local_tracking()
            else:
                logger.info("ℹ️ DagsHub token not available or connection failed, using local MLflow tracking")
                self.setup_local_tracking()
            
            # Set experiment with retry mechanism
            self.setup_experiment()
            
        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
            logger.info("Falling back to local MLflow tracking")
            self.setup_local_tracking()
    
    def setup_local_tracking(self):
        """Setup local MLflow tracking"""
        try:
            # Create local mlruns directory
            os.makedirs("mlruns", exist_ok=True)
            mlflow.set_tracking_uri("file:./mlruns")
            self.use_remote_tracking = False
            logger.info("✅ Local MLflow tracking setup completed")
            logger.info("📁 Experiments will be stored in ./mlruns")
        except Exception as e:
            logger.error(f"Error setting up local tracking: {str(e)}")
    
    def setup_experiment(self):
        """Setup MLflow experiment with retry mechanism"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                mlflow.set_experiment(self.args.experiment_name)
                logger.info(f"✅ Experiment set: {self.args.experiment_name}")
                return
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} to set experiment failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("Failed to set experiment after all retries")
                    raise e
    
    def load_data(self):
        """Load and prepare data"""
        try:
            logger.info(f"Loading data from {self.args.data_path}")
            
            if not os.path.exists(self.args.data_path):
                logger.warning(f"Data file {self.args.data_path} not found. Creating sample data...")
                self.create_sample_data()
            
            df = pd.read_csv(self.args.data_path)
            
            # Separate features and target
            X = df.drop('target', axis=1)
            y = df['target']
            
            logger.info(f"Data loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.args.random_state, stratify=y
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_sample_data(self):
        """Create sample iris data if not available"""
        try:
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['target'] = iris.target
            df.to_csv(self.args.data_path, index=False)
            logger.info(f"Sample data created: {self.args.data_path}")
        except Exception as e:
            logger.error(f"Error creating sample data: {str(e)}")
            raise
    
    def safe_mlflow_operation(self, operation, *args, **kwargs):
        """Execute MLflow operation with retry mechanism"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                logger.warning(f"MLflow operation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    if self.use_remote_tracking:
                        logger.warning("Remote tracking failed, switching to local tracking")
                        self.setup_local_tracking()
                        self.setup_experiment()
                        return operation(*args, **kwargs)
                    else:
                        raise e
    
    def save_artifact_locally(self, file_path, description=""):
        """Save artifact to local artifacts directory"""
        try:
            if os.path.exists(file_path):
                local_path = os.path.join(self.local_artifacts_dir, os.path.basename(file_path))
                shutil.copy2(file_path, local_path)
                logger.info(f"Saved locally: {local_path} {description}")
                return local_path
            else:
                logger.warning(f"File not found for local save: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error saving artifact locally: {str(e)}")
            return None
    
    def train_model(self):
        """Train model with MLflow tracking and local artifacts"""
        try:
            logger.info("Starting CI/CD model training...")
            
            # Load data
            X_train, X_test, y_train, y_test = self.load_data()
            
            # Start MLflow run with retry mechanism
            run = self.safe_mlflow_operation(
                mlflow.start_run, 
                run_name=f"ci_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            with run:
                # Log parameters
                params = {
                    "author": "alpian_khairi_C1BO",
                    "training_type": "ci_cd_workflow",
                    "data_path": self.args.data_path,
                    "n_estimators": self.args.n_estimators,
                    "max_depth": self.args.max_depth,
                    "random_state": self.args.random_state,
                    "timestamp": datetime.now().isoformat(),
                    "mlproject_run": True,
                    "tracking_type": "remote" if self.use_remote_tracking else "local",
                    "local_artifacts_dir": self.local_artifacts_dir
                }
                
                for key, value in params.items():
                    self.safe_mlflow_operation(mlflow.log_param, key, value)
                
                # Initialize and train model
                self.model = RandomForestClassifier(
                    n_estimators=self.args.n_estimators,
                    max_depth=self.args.max_depth,
                    random_state=self.args.random_state
                )
                
                logger.info("Training Random Forest model...")
                self.model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = self.model.predict(X_test)
                y_pred_proba = self.model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Log metrics
                metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "training_samples": len(X_train),
                    "test_samples": len(X_test)
                }
                
                for key, value in metrics.items():
                    self.safe_mlflow_operation(mlflow.log_metric, key, value)
                
                # Create and log visualizations
                self.create_visualizations(y_test, y_pred, y_pred_proba)
                
                # Log classification report
                self.log_classification_report(y_test, y_pred)
                
                # Save and log model
                model_file = "trained_model_ci.pkl"
                joblib.dump(self.model, model_file)
                
                # Save model locally
                local_model_path = self.save_artifact_locally(model_file, "(model)")
                
                # Log model to MLflow
                try:
                    self.safe_mlflow_operation(
                        mlflow.sklearn.log_model,
                        self.model, 
                        "model",
                        registered_model_name=self.args.model_name
                    )
                    logger.info("✅ Model registered successfully")
                except Exception as e:
                    logger.warning(f"Failed to register model: {str(e)}")
                    # Still log the model without registration
                    self.safe_mlflow_operation(mlflow.sklearn.log_model, self.model, "model")
                
                # Log artifact to MLflow
                self.safe_mlflow_operation(mlflow.log_artifact, model_file)
                
                # Create summary
                self.create_training_summary(metrics)
                
                logger.info("Model training completed successfully!")
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"Precision: {precision:.4f}")
                logger.info(f"Recall: {recall:.4f}")
                logger.info(f"F1-Score: {f1:.4f}")
                logger.info(f"Tracking Type: {'Remote (DagsHub)' if self.use_remote_tracking else 'Local'}")
                logger.info(f"Local Artifacts: {self.local_artifacts_dir}")
                
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'model_path': model_file,
                    'local_model_path': local_model_path,
                    'tracking_type': 'remote' if self.use_remote_tracking else 'local',
                    'local_artifacts_dir': self.local_artifacts_dir
                }
                
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def create_visualizations(self, y_test, y_pred, y_pred_proba):
        """Create and log visualizations with local backup"""
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.target_names,
                   yticklabels=self.target_names)
        plt.title('Confusion Matrix - CI/CD Training\nAuthor: alpian_khairi_C1BO')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        cm_file = 'confusion_matrix_ci.png'
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        
        # Save locally and log to MLflow
        self.save_artifact_locally(cm_file, "(confusion matrix)")
        self.safe_mlflow_operation(mlflow.log_artifact, cm_file)
        plt.close()
        
        # Feature Importance
        plt.figure(figsize=(10, 6))
        feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.title('Feature Importance - CI/CD Training\nAuthor: alpian_khairi_C1BO')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        fi_file = 'feature_importance_ci.png'
        plt.savefig(fi_file, dpi=300, bbox_inches='tight')
        
        # Save locally and log to MLflow
        self.save_artifact_locally(fi_file, "(feature importance)")
        self.safe_mlflow_operation(mlflow.log_artifact, fi_file)
        plt.close()
        
        logger.info("Visualizations created and logged")
    
    def log_classification_report(self, y_test, y_pred):
        """Log classification report with local backup"""
        
        report_text = classification_report(y_test, y_pred, 
                                          target_names=self.target_names)
        
        report_file = 'classification_report_ci.txt'
        with open(report_file, 'w') as f:
            f.write("CLASSIFICATION REPORT - CI/CD TRAINING\n")
            f.write("="*50 + "\n")
            f.write(f"Author: alpian_khairi_C1BO\n")
            f.write(f"Training Type: MLProject CI/CD Workflow\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Model: Random Forest\n")
            f.write(f"Parameters: n_estimators={self.args.n_estimators}, max_depth={self.args.max_depth}\n")
            f.write(f"Tracking: {'Remote (DagsHub)' if self.use_remote_tracking else 'Local'}\n")
            f.write(f"Local Artifacts: {self.local_artifacts_dir}\n")
            f.write("="*50 + "\n\n")
            f.write(report_text)
        
        # Save locally and log to MLflow
        self.save_artifact_locally(report_file, "(classification report)")
        self.safe_mlflow_operation(mlflow.log_artifact, report_file)
    
    def create_training_summary(self, metrics):
        """Create comprehensive training summary"""
        summary_file = "training_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write("# MLProject Training Summary\n\n")
            f.write(f"**Author:** alpian_khairi_C1BO\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Training Type:** CI/CD MLProject Workflow\n")
            f.write(f"**Tracking:** {'Remote (DagsHub)' if self.use_remote_tracking else 'Local MLflow'}\n\n")
            
            f.write("## Model Configuration\n")
            f.write(f"- **Algorithm:** Random Forest Classifier\n")
            f.write(f"- **N Estimators:** {self.args.n_estimators}\n")
            f.write(f"- **Max Depth:** {self.args.max_depth}\n")
            f.write(f"- **Random State:** {self.args.random_state}\n\n")
            
            f.write("## Performance Metrics\n")
            f.write(f"- **Accuracy:** {metrics['accuracy']:.4f}\n")
            f.write(f"- **Precision:** {metrics['precision']:.4f}\n")
            f.write(f"- **Recall:** {metrics['recall']:.4f}\n")
            f.write(f"- **F1-Score:** {metrics['f1_score']:.4f}\n\n")
            
            f.write("## Artifacts Generated\n")
            f.write("### MLflow Tracked Files\n")
            f.write("- Model (registered in MLflow)\n")
            f.write("- Metrics and parameters\n")
            f.write("- Confusion matrix visualization\n")
            f.write("- Feature importance plot\n")
            f.write("- Classification report\n\n")
            
            f.write("### Local Backup Files\n")
            f.write(f"Location: `./{self.local_artifacts_dir}/`\n")
            f.write("- trained_model_ci.pkl\n")
            f.write("- confusion_matrix_ci.png\n")
            f.write("- feature_importance_ci.png\n")
            f.write("- classification_report_ci.txt\n")
            f.write("- training_summary.md\n\n")
            
            if self.use_remote_tracking:
                f.write("## Remote Tracking\n")
                f.write("✅ **DagsHub Integration Active**\n")
                f.write("- Experiments: https://dagshub.com/alvian2022/iris-classification\n")
                f.write("- Model Registry: Available in DagsHub\n")
                f.write("- Metrics Dashboard: Available in DagsHub MLflow UI\n\n")
            else:
                f.write("## Local Tracking\n")
                f.write("📁 **Local MLflow Tracking**\n")
                f.write("- Experiments stored in: ./mlruns\n")
                f.write("- View with: `mlflow ui`\n\n")
        
        # Save locally and log to MLflow
        self.save_artifact_locally(summary_file, "(training summary)")
        self.safe_mlflow_operation(mlflow.log_artifact, summary_file)
        logger.info(f"Training summary created: {summary_file}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MLProject Model Training')
    
    parser.add_argument('--data_path', type=str, default='iris_preprocessing.csv',
                       help='Path to preprocessed data')
    parser.add_argument('--experiment_name', type=str, 
                       default='iris_classification_ci_alpian_khairi',
                       help='MLflow experiment name')
    parser.add_argument('--model_name', type=str, default='iris_classifier_ci',
                       help='Registered model name')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of trees in random forest')
    parser.add_argument('--max_depth', type=int, default=10,
                       help='Maximum depth of trees')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    return parser.parse_args()

def main():
    """Main function"""
    print("="*60)
    print("MLPROJECT CI/CD MODEL TRAINING")
    print("Author: alpian_khairi_C1BO")
    print("Version: 3.0 (Enhanced with Local Artifacts)")
    print("="*60)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        logger.info(f"Starting training with parameters:")
        logger.info(f"  Data path: {args.data_path}")
        logger.info(f"  Experiment: {args.experiment_name}")
        logger.info(f"  Model name: {args.model_name}")
        logger.info(f"  N estimators: {args.n_estimators}")
        logger.info(f"  Max depth: {args.max_depth}")
        logger.info(f"  Random state: {args.random_state}")
        
        # Initialize trainer
        trainer = MLProjectTrainer(args)
        
        # Train model
        results = trainer.train_model()
        
        print("\n" + "="*60)
        print("CI/CD TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"Model saved: {results['model_path']}")
        print(f"Tracking: {results['tracking_type'].upper()}")
        print(f"Local artifacts: {results['local_artifacts_dir']}")
        
        print("\nArtifacts generated:")
        print("  MLflow Tracked:")
        print("    - Model (registered)")
        print("    - Metrics and parameters")
        print("    - All visualization files")
        print("  Local Backup:")
        print("    - trained_model_ci.pkl")
        print("    - confusion_matrix_ci.png")
        print("    - feature_importance_ci.png")
        print("    - classification_report_ci.txt")
        print("    - training_summary.md")
        
        if results['tracking_type'] == 'remote':
            print("\n✅ Remote tracking active - check DagsHub for full experiment data")
            print("📁 Local backups available for offline access")
        else:
            print("\n📁 Local tracking active - run 'mlflow ui' to view experiments")
        
        # Return exit code 0 for successful completion
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"ERROR: Training failed - {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()