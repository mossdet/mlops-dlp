# Activate venv
source .venv/bin/activate

# Start MLflow
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

