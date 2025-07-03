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


