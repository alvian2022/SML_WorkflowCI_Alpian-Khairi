"""
Optimized Diabetes Model Training with Efficient Hyperparameter Tuning
Author: ALPIAN KHAIRI
Description: Fast version with reduced hyperparameter search space
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix, roc_auc_score,
                           precision_recall_curve, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import logging
import time
import joblib
from datetime import datetime
import dagshub

# Import config for DagsHub
from config_dagshub import setup_dagshub_tracking, setup_dagshub_auth

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedDiabetesModelTrainer:
    """Optimized diabetes model trainer with efficient hyperparameter tuning"""
    
    def __init__(self, data_path: str = "diabetes_preprocessed.csv"):
        """
        Initialize the trainer
        
        Args:
            data_path: Path to preprocessed diabetes data
        """
        self.data_path = data_path
        self.model = None
        self.best_model = None
        self.experiment_name = "diabetes_classification_optimized_Alpian-Khairi"
        self.search = None
        self.target_names = ['No Diabetes', 'Diabetes']
        
    def load_data(self) -> tuple:
        """Load preprocessed diabetes data"""
        try:
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            
            # Separate features and target
            X = df.drop('diabetes', axis=1)
            y = df['diabetes']
            
            logger.info(f"Dataset loaded successfully")
            logger.info(f"Features shape: {X.shape}")
            logger.info(f"Target shape: {y.shape}")
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def setup_mlflow_dagshub(self):
        """Setup MLflow with DagsHub tracking"""
        try:
            # Setup DagsHub authentication
            if not setup_dagshub_auth():
                raise Exception("DagsHub authentication failed")
            
            # Setup DagsHub tracking
            setup_dagshub_tracking()
            
            # Set experiment
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment set: {self.experiment_name}")
            
        except Exception as e:
            logger.error(f"Error setting up MLflow with DagsHub: {str(e)}")
            raise
    
    def efficient_hyperparameter_tuning(self, X_train, y_train) -> dict:
        """
        Efficient hyperparameter tuning using RandomizedSearchCV
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            dict: Tuning results
        """
        logger.info("Starting EFFICIENT hyperparameter tuning...")
        
        # OPTIMIZED parameter distributions (reduced search space)
        param_distributions = {
            'n_estimators': [50, 100, 150],  # Reduced from [50, 100, 200]
            'max_depth': [5, 10, None],      # Reduced from [3, 5, 7, 10, None]
            'min_samples_split': [2, 5],     # Reduced from [2, 5, 10]
            'min_samples_leaf': [1, 2],      # Reduced from [1, 2, 4]
            'max_features': ['sqrt'],        # Fixed to 'sqrt' (most common best choice)
            'bootstrap': [True]              # Fixed to True (usually better)
        }
        
        # Calculate total combinations
        total_combinations = 3 * 3 * 2 * 2 * 1 * 1  # 36 combinations
        logger.info(f"Total parameter combinations: {total_combinations}")
        
        # Use RandomizedSearchCV for faster execution
        # Sample only 20 combinations instead of all 36
        n_iter = min(20, total_combinations)
        
        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Randomized search with reduced CV folds
        self.search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=3,  # Reduced from 5 to 3 for faster execution
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        start_time = time.time()
        self.search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        self.best_model = self.search.best_estimator_
        
        logger.info(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
        logger.info(f"Best parameters: {self.search.best_params_}")
        logger.info(f"Best CV ROC AUC: {self.search.best_score_:.4f}")
        
        return {
            'best_params': self.search.best_params_,
            'best_score': self.search.best_score_,
            'tuning_time': tuning_time,
            'cv_results': self.search.cv_results_,
            'n_iter': n_iter
        }
    
    def quick_baseline_training(self, X_train, y_train) -> dict:
        """
        Quick baseline training without hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            dict: Training results
        """
        logger.info("Training quick baseline model...")
        
        # Use default parameters with some optimization
        baseline_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        start_time = time.time()
        baseline_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Quick cross-validation
        cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=3, scoring='roc_auc')
        
        logger.info(f"Baseline training completed in {training_time:.2f} seconds")
        logger.info(f"CV ROC AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return {
            'model': baseline_model,
            'training_time': training_time,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores)
        }
    
    def train_model(self, use_tuning: bool = True) -> dict:
        """
        Train model with option for hyperparameter tuning
        
        Args:
            use_tuning: Whether to use hyperparameter tuning
            
        Returns:
            dict: Training results
        """
        try:
            logger.info("Starting optimized diabetes model training...")
            
            # Load data
            X_train, X_test, y_train, y_test = self.load_data()
            
            # Setup MLflow with DagsHub
            self.setup_mlflow_dagshub()
            
            # Ensure any existing run is ended
            if mlflow.active_run():
                mlflow.end_run()
            
            try:
                run_name = "optimized_diabetes_tuned" if use_tuning else "optimized_diabetes_baseline"
                
                with mlflow.start_run(run_name=run_name) as run:
                    logger.info(f"Started MLflow run: {run_name}")
                    
                    # Disable autolog for manual logging
                    mlflow.sklearn.autolog(disable=True)
                    
                    # Log basic information
                    mlflow.log_param("author", "Alpian-Khairi")
                    mlflow.log_param("model_type", "optimized_diabetes")
                    mlflow.log_param("dataset", "diabetes")
                    mlflow.log_param("use_tuning", use_tuning)
                    mlflow.log_param("timestamp", datetime.now().isoformat())
                    
                    if use_tuning:
                        # Efficient hyperparameter tuning
                        tuning_results = self.efficient_hyperparameter_tuning(X_train, y_train)
                        
                        # Log tuning parameters
                        mlflow.log_params(tuning_results['best_params'])
                        mlflow.log_metric("best_cv_roc_auc", tuning_results['best_score'])
                        mlflow.log_metric("tuning_time_seconds", tuning_results['tuning_time'])
                        mlflow.log_metric("n_iter_searched", tuning_results['n_iter'])
                        
                        final_model = self.best_model
                        
                    else:
                        # Quick baseline training
                        baseline_results = self.quick_baseline_training(X_train, y_train)
                        
                        # Log baseline parameters
                        mlflow.log_param("n_estimators", 100)
                        mlflow.log_param("max_depth", 10)
                        mlflow.log_param("min_samples_split", 5)
                        mlflow.log_param("min_samples_leaf", 2)
                        mlflow.log_param("max_features", "sqrt")
                        mlflow.log_param("bootstrap", True)
                        
                        mlflow.log_metric("baseline_cv_roc_auc", baseline_results['cv_mean'])
                        mlflow.log_metric("baseline_cv_std", baseline_results['cv_std'])
                        mlflow.log_metric("baseline_training_time", baseline_results['training_time'])
                        
                        final_model = baseline_results['model']
                        tuning_results = None
                    
                    # Make predictions
                    start_time = time.time()
                    y_pred = final_model.predict(X_test)
                    y_pred_proba = final_model.predict_proba(X_test)
                    prediction_time = time.time() - start_time
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                    
                    # Log all metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                    
                    mlflow.log_metric("prediction_time_seconds", prediction_time)
                    mlflow.log_metric("total_features", X_train.shape[1])
                    mlflow.log_metric("train_samples", X_train.shape[0])
                    mlflow.log_metric("test_samples", X_test.shape[0])
                    
                    # Create essential visualizations
                    self._create_essential_visualizations(X_test, y_test, y_pred, y_pred_proba, final_model)
                    
                    # Log model
                    model_name = "diabetes_classifier_optimized_Alpian-Khairi"
                    mlflow.sklearn.log_model(
                        final_model, 
                        "optimized_diabetes_model",
                        registered_model_name=model_name
                    )
                    
                    # Prepare results
                    results = {
                        'model': final_model,
                        'metrics': metrics,
                        'tuning_results': tuning_results,
                        'predictions': y_pred,
                        'prediction_probabilities': y_pred_proba,
                        'test_features': X_test,
                        'test_target': y_test,
                        'use_tuning': use_tuning
                    }
                    
                    logger.info("Optimized diabetes model training completed successfully!")
                    self._log_results_summary(metrics, use_tuning)
                    
                    return results
                    
            except Exception as e:
                logger.error(f"Error during MLflow run: {str(e)}")
                if mlflow.active_run():
                    mlflow.end_run(status="FAILED")
                raise
                    
        except Exception as e:
            logger.error(f"Error during optimized diabetes model training: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_test, y_pred, y_pred_proba) -> dict:
        """Calculate essential metrics"""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        metrics['pr_auc'] = auc(recall, precision)
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['sensitivity'] = tp / (tp + fn)  # Recall for positive class
            metrics['specificity'] = tn / (tn + fp)  # Recall for negative class
        
        return metrics
    
    def _create_essential_visualizations(self, X_test, y_test, y_pred, y_pred_proba, model):
        """Create essential visualizations"""
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.target_names,
                   yticklabels=self.target_names)
        plt.title('Confusion Matrix - Optimized Diabetes Model\nAuthor: Alpian-Khairi')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix_optimized.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('confusion_matrix_optimized.png')
        plt.close()
        
        # 2. Feature Importance
        plt.figure(figsize=(10, 6))
        feature_names = X_test.columns.tolist()
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices], alpha=0.7, color='lightcoral')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.title('Feature Importance - Optimized Diabetes Model\nAuthor: Alpian-Khairi')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance_optimized.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('feature_importance_optimized.png')
        plt.close()
        
        # 3. ROC and PR Curves
        plt.figure(figsize=(12, 5))
        
        # ROC Curve
        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        # Precision-Recall Curve
        plt.subplot(1, 2, 2)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig('roc_pr_curves_optimized.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('roc_pr_curves_optimized.png')
        plt.close()
        
        logger.info("Essential visualizations created and logged")
    
    def _log_results_summary(self, metrics: dict, use_tuning: bool):
        """Log results summary"""
        
        logger.info("="*50)
        logger.info("OPTIMIZED DIABETES MODEL RESULTS")
        logger.info("="*50)
        logger.info(f"Hyperparameter Tuning: {'Yes' if use_tuning else 'No'}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
        logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
        logger.info(f"Specificity: {metrics['specificity']:.4f}")
        logger.info("="*50)

def main():
    """Main function with option to choose training mode"""
    
    print("="*60)
    print("OPTIMIZED DIABETES MODEL TRAINING")
    print("Author: Alpian-Khairi")
    print("="*60)
    
    try:
        # Check if preprocessed data exists
        data_file = "diabetes_preprocessed.csv"
        if not os.path.exists(data_file):
            print(f"ERROR: Preprocessed data file '{data_file}' not found!")
            return
        
        # Initialize trainer
        trainer = OptimizedDiabetesModelTrainer(data_file)
        
        # Ask user for training mode
        print("\nChoose training mode:")
        print("1. Quick Baseline (No tuning - ~30 seconds)")
        print("2. Efficient Tuning (Reduced search space - ~2-3 minutes)")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            use_tuning = False
            print("\n🚀 Starting QUICK BASELINE training...")
        elif choice == "2":
            use_tuning = True
            print("\n🔧 Starting EFFICIENT HYPERPARAMETER TUNING...")
        else:
            print("Invalid choice. Using quick baseline mode...")
            use_tuning = False
        
        # Ensure clean state
        if mlflow.active_run():
            mlflow.end_run()
        
        # Train model
        start_time = time.time()
        results = trainer.train_model(use_tuning=use_tuning)
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Mode: {'Hyperparameter Tuning' if use_tuning else 'Quick Baseline'}")
        print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"Precision: {results['metrics']['precision']:.4f}")
        print(f"Recall: {results['metrics']['recall']:.4f}")
        print(f"F1-Score: {results['metrics']['f1_score']:.4f}")
        print(f"ROC AUC: {results['metrics']['roc_auc']:.4f}")
        print(f"PR AUC: {results['metrics']['pr_auc']:.4f}")
        
        if use_tuning and results['tuning_results']:
            print(f"Best CV ROC AUC: {results['tuning_results']['best_score']:.4f}")
            print(f"Tuning Time: {results['tuning_results']['tuning_time']:.2f} seconds")
            print("\nBest Parameters:")
            for param, value in results['tuning_results']['best_params'].items():
                print(f"  {param}: {value}")
        
        print("\nFiles generated:")
        print("  - confusion_matrix_optimized.png")
        print("  - feature_importance_optimized.png")
        print("  - roc_pr_curves_optimized.png")
        
        print("\n🔗 View results at: https://dagshub.com/alvian2022/diabetes-classification")
        
    except Exception as e:
        print(f"ERROR: Training failed - {str(e)}")
        logger.error(f"Training failed: {str(e)}")
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise

if __name__ == "__main__":
    main()