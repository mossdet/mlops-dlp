# Experiment Tracking

## 📚 Index

1. [Track Model Development](#01-track-model-development)<br>
2. [Model Registry](#02-model-registry)<br>
3. [Remote Experiment Tracking](#03-setup-remote-mlflow)<br>
    3.1 [Setup Remote Data Lake](#031-step-1-setup-remote-data-lake)  
    3.2 [Setup Remote Database](#032-step-2-setup-remote-database)  
    3.3 [Setup Remote Tracking Server](#033-step-3-setup-remote-tracking-server)  

# Visual Summary
![Kiku](Images/W2-Experiment-Tracking_v2.png)

### 1. Track model development <a name="01-track-model-development"></a>
### Use Class [NYCTaxiDurationPrediction](duration_prediction.py)
- Setup mlflow backend store / tracking server
- Pre-process data, train models and optimize hyperparameters with optuna
- Log the models' parameters to mlflow
- Log the model to mlflow



### 2. Model Registry <a name="02-model-registry"></a>
### Use Class [NYCTaxiDurationPrediction](model_registry.py)
- Search runs in an experiment and show each run's performance
- Register a model
- Change a registered model's Tags and description
- Retrieve a model and other artifacts
- Test retrieved models
    - Compare each registered model's size, training time, run time and performance on the test-set
- Elevate a model to Production status (i.e. set Tags)



### 3. Remote Experiment Tracking <a name="03-setup-remote-mlflow"></a>
#### 3.1. Create an S3 bucket to store the artifacts <a name="031-step-1-setup-remote-data-lake"></a>
#### 3.2. Create a PostgreSQL database in RDS to use as the mlflow database containing all metdadata <a name="032-step-2-setup-remote-database"></a>
#### 3.3. Setup Remote Tracking Server <a name="033-step-3-setup-remote-tracking-server"></a>
**Create an EC2 instance to use as the remote tracking-server**
- Configure ssh access with key-pair file
- Update packages:
```bash
sudo apt-get update
```
- Install uv python and create virtualenv with packages
```bash
uv init tracking_server
uv add mlflow boto3 psycopg2-binary
```

- Install AWS cli
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

- On both the local machine (i.e. the one used for training models) as well as the mlflow-tracking-server, AWS cli must be setup.
Setup AWS ***Access key ID*** and ***Secret access key***. If you want to configure the default profile, simply run aws configure. If you want to configure a named profile, use the --profile option: aws configure --profile <profile_name>
```bash
aws configure
```
```bash
aws configure --profile mlops_zc
```

Test access to bucket:
```bash
aws s3 ls
```

- Start mlflow server in AWS:
```bash
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/DB_NAME --default-artifact-root s3://S3_BUCKET_NAME
```
- Open MLflow on local browser:
```bash
Public DNS:5000/
```





