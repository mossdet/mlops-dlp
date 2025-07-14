# Start MLflow server
- Go to the directory where the backend database will be created, e.g.:
```bash
cd ~/mlops-dlp/w2_Experiment_Tracking/running-mlflow-examples/mlflow_cases/
```

- Start the mlflow tracking/backend server
```bash
mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root ./artifacts_local
```

