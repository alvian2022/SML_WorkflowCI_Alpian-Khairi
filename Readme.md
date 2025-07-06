# 🎯 MLflow CI/CD Training & Docker Summary

- **Timestamp**: Tue Jul  1 07:20:34 UTC 2025
- **Parameters**: n_estimators=100, max_depth=10
- **Experiment**:diabetes-classification_ci_alpian_khairi
- **Remote Tracking**: false
- **Docker Build**: true
- **Author**: alpian_khairi_C1BO
## 🐳 Docker Information
- **Docker Hub Repository**: https://hub.docker.com/r/alvian2023/diabetes-classification-mlflow
- **Latest Tag**: `***/diabetes-classification-mlflow:latest`
- **Build Tag**: `***/diabetes-classification-mlflow:build-15`
- **Pull Command**: `docker pull ***/diabetes-classification-mlflow:latest`
- **Run Command**: `docker run -p 8080:8080 ***/diabetes-classification-mlflow:latest`


## 📁 Generated Files
```
Workflow-CI/
├── .github/workflows/mlflow-training.yml
├── MLProject/
│   ├── MLproject ✓
│   ├── conda.yaml ✓
│   ├── modelling.py ✓
│   ├── diabetes_preprocessed.csv ✓
│   └── Dockerfile
├── mlruns/ ✓ (hasil eksekusi)
└── README.md
```

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

