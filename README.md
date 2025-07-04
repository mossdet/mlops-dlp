# MLOps

## 📚 Index

1. [Setup EC2 Development Environment](/week01/)
2. [Experiment Tracking](/week02/)

## 1. Setup EC2 Development Environment <a name="01-setup-development-environment"></a>
- Create EC2 Instance
- Setup ssh access to EC2 instance
- Install [uv-python ](https://docs.astral.sh/uv/getting-started/installation/) to EC2 instance:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- Install needed packages
```bash
uv add mlflow numpy pandas xgboost scikit-learn jupyter lab
```
- Activate venv
source .venv/bin/activate

- Start MLflow
```bash
cd /mlflow/diretcory/
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```
- Monitor CPU
```bash
sar -u 5
```


## 2. Experiment Tracking <a name="01-setup-development-environment"></a>
![Experiment-TRacking-Visual-Summary](/Visual_Summaries/W2-Experiment-Tracking_v2.png)
