# Experiment tracking

### 1. Track model development
- Setup mlflow backend store / tracking server
- Pre-process data, train models and optimize hyperparameters with optuna
- Log the models' parameters to mlflow
- Log the model to mlflow

### 2. Model Registry
- Search runs in an experiment and show each run's performance
- Register a model
- Change a registered model's Tags and description

### 3. Retrieve a model and other artifacts

### 4. Test retrieved models
- Compare each registered model's size, training time, run time and performance on the test-set

### 4. Elevate a model to Production status (i.e. set Tags)

<br><br>
# Remote experiment tracking
#### 1. Create an EC2 instance to use as the remote tracking-server
#### 2. Create an S3 bucket to store the artifacts
#### 3. Create a PostgreSQL database in RDS to use as the mlflow database containing all metdadata

### **Tracking Server** ###
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





