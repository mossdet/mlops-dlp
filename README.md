# MLOps

## 📚 Index

1. [Setup EC2 Development Environment](#01-setup-development-environment)<br>
2. [Experiment Tracking](#02-experiment-tracking)<br>

## 1. Setup EC2 Development Environment
### Activate venv
source .venv/bin/activate

### Start MLflow
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

### Monitor CPU
sar -u 5

## 2. Experiment Tracking
![Experiment-TRacking-Visual-Summary](week02/Images/W2-Experiment-Tracking_v2.png)
