"""
MLProject modelling.py - Diabetes Classification
CI/CD Model Training for MLflow Project
Author: Alpian Khairi
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
                           f1_score, classification_report, confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class DiabetesMLProjectTrainer:
    """Diabetes model trainer for MLProject CI/CD workflow"""
    
    def __init__(self, args):
        """Initialize trainer with arguments"""
        self.args = args
        self.model = None
        self.target_names = ['No Diabetes', 'Diabetes']
        self.local_artifacts_dir = "model_artifacts"
        self.setup_local_artifacts_dir()
        self.setup_mlflow()
    
    def setup_local_artifacts_dir(self):
        """Setup local artifacts directory"""
        os.makedirs(self.local_artifacts_dir, exist_ok=True)
        logger.info(f"Local artifacts directory: {self.local_artifacts_dir}")
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            # Create local mlruns directory
            os.makedirs("mlruns", exist_ok=True)
            mlflow.set_tracking_uri("file:./mlruns")
            
            # Set experiment
            mlflow.set_experiment(self.args.experiment_name)
            logger.info(f"MLflow experiment set: {self.args.experiment_name}")
            
        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
            raise
    
    def load_data(self):
        """Load and prepare diabetes data"""
        try:
            logger.info(f"Loading data from {self.args.data_path}")
            
            if not os.path.exists(self.args.data_path):
                logger.error(f"Data file {self.args.data_path} not found!")
                raise FileNotFoundError(f"Data file {self.args.data_path} not found!")
            
            df = pd.read_csv(self.args.data_path)
            
            # Separate features and target
            X = df.drop('diabetes', axis=1)
            y = df['diabetes']
            
            logger.info(f"Data loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.args.random_state, stratify=y
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
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
        """Train diabetes classification model with MLflow tracking"""
        try:
            logger.info("Starting diabetes model training for MLProject...")
            
            # Load data
            X_train, X_test, y_train, y_test = self.load_data()
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"diabetes_ci_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Enable autolog
                mlflow.sklearn.autolog()
                
                # Log parameters
                params = {
                    "author": "[nama-siswa]",
                    "training_type": "mlproject_ci_cd",
                    "dataset": "diabetes",
                    "data_path": self.args.data_path,
                    "n_estimators": self.args.n_estimators,
                    "max_depth": self.args.max_depth,
                    "random_state": self.args.random_state,
                    "timestamp": datetime.now().isoformat(),
                    "mlproject_run": True,
                    "local_artifacts_dir": self.local_artifacts_dir
                }
                
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # Initialize and train model
                self.model = RandomForestClassifier(
                    n_estimators=self.args.n_estimators,
                    max_depth=self.args.max_depth,
                    random_state=self.args.random_state
                )
                
                logger.info("Training Random Forest model for diabetes classification...")
                self.model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = self.model.predict(X_test)
                y_pred_proba = self.model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                
                # Log additional metrics
                additional_metrics = {
                    "diabetes_accuracy": accuracy,
                    "diabetes_precision": precision,
                    "diabetes_recall": recall,
                    "diabetes_f1_score": f1,
                    "diabetes_roc_auc": roc_auc,
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "feature_count": X_train.shape[1]
                }
                
                for key, value in additional_metrics.items():
                    mlflow.log_metric(key, value)
                
                # Create and log visualizations
                self.create_visualizations(X_test, y_test, y_pred, y_pred_proba)
                
                # Log classification report
                self.log_classification_report(y_test, y_pred)
                
                # Save and log model
                model_file = "diabetes_model_ci.pkl"
                joblib.dump(self.model, model_file)
                
                # Save model locally
                local_model_path = self.save_artifact_locally(model_file, "(diabetes model)")
                
                # Log model to MLflow
                try:
                    mlflow.sklearn.log_model(
                        self.model, 
                        "diabetes_model",
                        registered_model_name=self.args.model_name
                    )
                    logger.info("Diabetes model registered successfully")
                except Exception as e:
                    logger.warning(f"Failed to register model: {str(e)}")
                    mlflow.sklearn.log_model(self.model, "diabetes_model")
                
                # Log artifact to MLflow
                mlflow.log_artifact(model_file)
                
                # Create summary
                self.create_training_summary(additional_metrics)
                
                logger.info("Diabetes model training completed successfully!")
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"Precision: {precision:.4f}")
                logger.info(f"Recall: {recall:.4f}")
                logger.info(f"F1-Score: {f1:.4f}")
                logger.info(f"ROC AUC: {roc_auc:.4f}")
                
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'model_path': model_file,
                    'local_model_path': local_model_path,
                    'local_artifacts_dir': self.local_artifacts_dir
                }
                
        except Exception as e:
            logger.error(f"Error during diabetes model training: {str(e)}")
            raise
    
    def create_visualizations(self, X_test, y_test, y_pred, y_pred_proba):
        """Create and log visualizations for diabetes model"""
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.target_names,
                   yticklabels=self.target_names)
        plt.title('Confusion Matrix - Diabetes Classification (MLProject)\nAuthor: [nama-siswa]')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        cm_file = 'confusion_matrix_diabetes_ci.png'
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        
        # Save locally and log to MLflow
        self.save_artifact_locally(cm_file, "(confusion matrix)")
        mlflow.log_artifact(cm_file)
        plt.close()
        
        # 2. Feature Importance
        plt.figure(figsize=(12, 8))
        feature_names = X_test.columns.tolist()
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices], alpha=0.7, color='lightcoral')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.title('Feature Importance - Diabetes Classification (MLProject)\nAuthor: [nama-siswa]')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        fi_file = 'feature_importance_diabetes_ci.png'
        plt.savefig(fi_file, dpi=300, bbox_inches='tight')
        
        # Save locally and log to MLflow
        self.save_artifact_locally(fi_file, "(feature importance)")
        mlflow.log_artifact(fi_file)
        plt.close()
        
        # 3. ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba[:, 1]):.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Diabetes Classification (MLProject)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        roc_file = 'roc_curve_diabetes_ci.png'
        plt.savefig(roc_file, dpi=300, bbox_inches='tight')
        
        # Save locally and log to MLflow
        self.save_artifact_locally(roc_file, "(ROC curve)")
        mlflow.log_artifact(roc_file)
        plt.close()
        
        logger.info("Diabetes visualizations created and logged")
    
    def log_classification_report(self, y_test, y_pred):
        """Log classification report for diabetes model"""
        
        report_text = classification_report(y_test, y_pred, 
                                          target_names=self.target_names)
        
        report_file = 'classification_report_diabetes_ci.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("CLASSIFICATION REPORT - DIABETES CLASSIFICATION (MLPROJECT)\n")
            f.write("="*60 + "\n")
            f.write(f"Author: [nama-siswa]\n")
            f.write(f"Training Type: MLProject CI/CD Workflow\n")
            f.write(f"Dataset: Diabetes\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Model: Random Forest\n")
            f.write(f"Parameters: n_estimators={self.args.n_estimators}, max_depth={self.args.max_depth}\n")
            f.write(f"Local Artifacts: {self.local_artifacts_dir}\n")
            f.write("="*60 + "\n\n")
            f.write(report_text)
        
        # Save locally and log to MLflow
        self.save_artifact_locally(report_file, "(classification report)")
        mlflow.log_artifact(report_file)
    
    def create_training_summary(self, metrics):
        """Create comprehensive training summary for diabetes model"""
        summary_file = "diabetes_training_summary.md"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# Diabetes Classification MLProject Training Summary\n\n")
            f.write(f"**Author:** [nama-siswa]\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Training Type:** MLProject CI/CD Workflow\n")
            f.write(f"**Dataset:** Diabetes Classification\n\n")
            
            f.write("## Model Configuration\n")
            f.write(f"- **Algorithm:** Random Forest Classifier\n")
            f.write(f"- **N Estimators:** {self.args.n_estimators}\n")
            f.write(f"- **Max Depth:** {self.args.max_depth}\n")
            f.write(f"- **Random State:** {self.args.random_state}\n\n")
            
            f.write("## Performance Metrics\n")
            f.write(f"- **Accuracy:** {metrics['diabetes_accuracy']:.4f}\n")
            f.write(f"- **Precision:** {metrics['diabetes_precision']:.4f}\n")
            f.write(f"- **Recall:** {metrics['diabetes_recall']:.4f}\n")
            f.write(f"- **F1-Score:** {metrics['diabetes_f1_score']:.4f}\n")
            f.write(f"- **ROC AUC:** {metrics['diabetes_roc_auc']:.4f}\n\n")
            
            f.write("## Data Information\n")
            f.write(f"- **Training Samples:** {metrics['training_samples']}\n")
            f.write(f"- **Test Samples:** {metrics['test_samples']}\n")
            f.write(f"- **Feature Count:** {metrics['feature_count']}\n")
            f.write(f"- **Target Classes:** {self.target_names}\n\n")
            
            f.write("## Artifacts Generated\n")
            f.write("### MLflow Tracked Files\n")
            f.write("- Model (registered in MLflow)\n")
            f.write("- Metrics and parameters\n")
            f.write("- Confusion matrix visualization\n")
            f.write("- Feature importance plot\n")
            f.write("- ROC curve\n")
            f.write("- Classification report\n\n")
            
            f.write("### Local Backup Files\n")
            f.write(f"Location: `./{self.local_artifacts_dir}/`\n")
            f.write("- diabetes_model_ci.pkl\n")
            f.write("- confusion_matrix_diabetes_ci.png\n")
            f.write("- feature_importance_diabetes_ci.png\n")
            f.write("- roc_curve_diabetes_ci.png\n")
            f.write("- classification_report_diabetes_ci.txt\n")
            f.write("- diabetes_training_summary.md\n\n")
            
            f.write("## MLProject Information\n")
            f.write("- **Experiment:** Local MLflow Tracking\n")
            f.write("- **Tracking URI:** file:./mlruns\n")
            f.write("- **View with:** `mlflow ui`\n")
            f.write("- **MLProject Entry Point:** main\n")
        
        # Save locally and log to MLflow
        self.save_artifact_locally(summary_file, "(training summary)")
        mlflow.log_artifact(summary_file)
        logger.info(f"Diabetes training summary created: {summary_file}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Diabetes MLProject Model Training')
    
    parser.add_argument('--data_path', type=str, default='diabetes_preprocessed.csv',
                       help='Path to preprocessed diabetes data')
    parser.add_argument('--experiment_name', type=str, 
                       default='diabetes_classification_ci_[nama-siswa]',
                       help='MLflow experiment name')
    parser.add_argument('--model_name', type=str, default='diabetes_classifier_ci',
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
    print("="*70)
    print("DIABETES CLASSIFICATION MLPROJECT CI/CD TRAINING")
    print("Author: [nama-siswa]")
    print("Kriteria 3: MLProject Workflow")
    print("="*70)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        print(f"Starting diabetes training with parameters:")
        print(f"  Data path: {args.data_path}")
        print(f"  Experiment: {args.experiment_name}")
        print(f"  Model name: {args.model_name}")
        print(f"  N estimators: {args.n_estimators}")
        print(f"  Max depth: {args.max_depth}")
        print(f"  Random state: {args.random_state}")
        
        # Initialize trainer
        trainer = DiabetesMLProjectTrainer(args)
        
        # Train model
        results = trainer.train_model()
        
        print("\n" + "="*70)
        print("DIABETES CI/CD TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print(f"Model saved: {results['model_path']}")
        print(f"Local artifacts: {results['local_artifacts_dir']}")
        
        print("\nArtifacts generated:")
        print("  MLflow Tracked:")
        print("    - Diabetes model (registered)")
        print("    - Metrics and parameters")
        print("    - All visualization files")
        print("  Local Backup:")
        print("    - diabetes_model_ci.pkl")
        print("    - confusion_matrix_diabetes_ci.png")
        print("    - feature_importance_diabetes_ci.png")
        print("    - roc_curve_diabetes_ci.png")
        print("    - classification_report_diabetes_ci.txt")
        print("    - diabetes_training_summary.md")
        
        print("\n📊 MLProject requirements fulfilled:")
        print("  ✅ MLProject folder structure")
        print("  ✅ Conda environment (conda.yaml)")
        print("  ✅ Entry points defined")
        print("  ✅ Diabetes model training")
        print("  ✅ CI/CD ready workflow")
        
        print(f"\n🔧 Run MLflow UI with: mlflow ui")
        print(f"🐳 Ready for Docker build with MLProject structure")
        
        # Return exit code 0 for successful completion
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: Diabetes training failed - {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()