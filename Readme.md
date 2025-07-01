# 🎯 MLflow CI/CD Training & Docker Summary

- **Timestamp**: Tue Jul  1 07:20:34 UTC 2025
- **Parameters**: n_estimators=100, max_depth=10
- **Experiment**: iris_classification_ci_alpian_khairi
- **Remote Tracking**: false
- **Docker Build**: true
- **Author**: alpian_khairi_C1BO
## 🐳 Docker Information
- **Docker Hub Repository**: https://hub.docker.com/r/alvian2023/iris-classification-mlflow
- **Latest Tag**: `***/iris-classification-mlflow:latest`
- **Build Tag**: `***/iris-classification-mlflow:build-15`
- **Pull Command**: `docker pull ***/iris-classification-mlflow:latest`
- **Run Command**: `docker run -p 8080:8080 ***/iris-classification-mlflow:latest`
## 📁 Generated Files
```
Workflow-CI/
├── .github/workflows/mlflow-training.yml
├── MLProject/
│   ├── MLproject ✓
│   ├── conda.yaml ✓
│   ├── modelling.py ✓
│   ├── iris_preprocessing.csv ✓
│   └── Dockerfile
├── mlruns/ ✓ (hasil eksekusi)
└── README.md
```
## 📊 Training Results
# MLProject Training Summary

**Author:** alpian_khairi_C1BO
**Date:** 2025-07-01 07:18:42
**Training Type:** CI/CD MLProject Workflow
**Tracking:** Local MLflow

## Model Configuration
- **Algorithm:** Random Forest Classifier
- **N Estimators:** 100
- **Max Depth:** 10
- **Random State:** 42

## Performance Metrics
- **Accuracy:** 0.9333
- **Precision:** 0.9333
- **Recall:** 0.9333
- **F1-Score:** 0.9333

## Artifacts Generated
### MLflow Tracked Files
- Model (registered in MLflow)
- Metrics and parameters
- Confusion matrix visualization
- Feature importance plot
- Classification report

### Local Backup Files
Location: `./model_artifacts/`
- trained_model_ci.pkl
- confusion_matrix_ci.png
- feature_importance_ci.png
- classification_report_ci.txt
- training_summary.md

