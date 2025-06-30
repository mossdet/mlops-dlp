import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import mlflow
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope


mlflow.set_tracking_uri("sqlite:///mlflow_db/mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

def read_dataframe(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)

        df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
        df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df



# Load the dataset
training_data_path = '~/Data/green_tripdata_2021-01.parquet'
validation_data_path = '~/Data/green_tripdata_2021-02.parquet'
df_train = read_dataframe(training_data_path)
df_val = read_dataframe(validation_data_path)

df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']


categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
numerical = ['trip_distance']

# Convert categorical features to string type
df_train[categorical] = df_train[categorical].astype(str)
df_val[categorical] = df_val[categorical].astype(str)


# Transform the mixed data types into a format suitable for machine learning using DictVectorizer
# DictVectorizer converts a list of dictionaries into a matrix of features
# Each dictionary corresponds to a row in the DataFrame, with keys as feature names and values as feature values
# Numerical features are already in a suitable format, while categorical features need to be one-hot encoded
# The DictVectorizer will handle this automatically by creating binary columns for each category
# in the categorical features
# The resulting matrix will have one column for each unique value in the categorical features and one column for each numerical feature
# The DictVectorizer will also handle the conversion of the categorical features to one-hot encoded features,
# which means that each unique value in the categorical features will be represented by a separate binary column in the resulting matrix
# The resulting matrix will be sparse, meaning it will only store non-zero values to save memory
# The DictVectorizer will also handle missing values by treating them as a separate category
dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

# Select the target variable
# The target variable is the duration of the trip in minutes
target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values

# Fit a linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on the validation set
y_pred = lr.predict(X_val)
rmse_val = root_mean_squared_error(y_val, y_pred)
r2_val = r2_score(y_val, y_pred)
print(f'RMSE: {rmse_val:.3f}')
print(f'R2: {r2_val:.3f}')

# # Save the model and DictVectorizer
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
model_fpath = os.path.join(model_dir, 'lin_reg.bin')
with open(model_fpath, 'wb') as f_out:
    pickle.dump((dv, lr), f_out)

# Log the model and metrics using MLflow
with mlflow.start_run():

    mlflow.set_tag("developer", "dlp")

    mlflow.log_param("train-data-path", training_data_path)
    mlflow.log_param("valid-data-path", validation_data_path)

    alpha = 0.01
    mlflow.log_param("alpha", alpha)
    lr = Lasso(alpha)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)

    mlflow.log_artifact(local_path=model_fpath, artifact_path="models_pickle")

pass


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_parquet('~/Data/green_tripdata_2021-01.parquet')