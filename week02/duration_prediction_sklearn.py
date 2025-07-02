import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import mlflow
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance, plot_tree
import optuna

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR



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

# Create a new feature that combines pickup and dropoff locations
df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']

# Select features by data type
categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
numerical = ['trip_distance']

# Convert categorical features to string type
df_train[categorical] = df_train[categorical].astype(str)
df_val[categorical] = df_val[categorical].astype(str)


# Transform the mixed data types into a format suitable for machine learning using DictVectorizer
dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_valid = dv.transform(val_dicts)

# Select the target variable
# The target variable is the duration of the trip in minutes
target = 'duration'
y_train = df_train[target].values
y_valid = df_val[target].values

X_examples = dv.transform(val_dicts[0:10])

# X_train = X_train.toarray()
# X_valid = X_valid.toarray()

mlflow.sklearn.autolog()

for model_class in (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, LinearSVR):

    with mlflow.start_run():

        print(f"Training model: {model_class.__name__}")

        mlflow.log_param("train-data-path", training_data_path)
        mlflow.log_param("valid-data-path", validation_data_path)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlmodel = model_class()
        mlmodel.fit(X_train, y_train)

        y_pred = mlmodel.predict(X_valid)
        rmse = root_mean_squared_error(y_valid, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(mlmodel, artifact_path="models_mlflow", input_example=X_examples)

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_parquet('~/Data/green_tripdata_2021-01.parquet')