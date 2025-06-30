import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import mlflow
import xgboost as xgb
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

def objective_XGB(trial, data_train, data_valid, y_train, y_valid):


    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 100, step=1),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 10, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10, step=1),
        "subsample": trial.suggest_float("subsample", 0.5, 1, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1, step=0.1),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1, step=0.1),
        #"scale_pos_weight": trial.suggest_float("scale_pos_weight", scale_pos_weight_range[0], scale_pos_weight_range[1], log=True),
        # Constants
        #"device": trial.suggest_categorical("device", ["cuda"]),
        "objective": trial.suggest_categorical("objective", ["reg:squarederror"]),
        "tree_method": trial.suggest_categorical("tree_method", ["hist"]),
        #"seed": trial.suggest_categorical("seed", [42]),
        # "num_boost_round": trial.suggest_categorical("num_boost_round", [200]),
        # "early_stopping_rounds": trial.suggest_categorical("early_stopping_rounds", [50]),
    }

    # params['num_boost_round'] = 200
    # params['early_stopping_rounds'] = 50
    # params['seed'] = 42  # Fixed seed for reproducibility

    model = xgb.train(
        params=params,
        dtrain=data_train,
        evals=[(data_valid, "valid")],
        verbose_eval=50,
        num_boost_round=200,
        early_stopping_rounds=50,
    )

    y_pred = model.predict(data_train)
    if np.isnan(y_pred).any():
        return float("inf")  # Return a high value if predictions are NaN
    rmse_train = root_mean_squared_error(y_train, y_pred)

    y_pred = model.predict(data_valid)
    if np.isnan(y_pred).any():
        return float("inf")  # Return a high value if predictions are NaN

    # Calculate RMSE for validation set
    rmse_valid = root_mean_squared_error(y_valid, y_pred)

    with mlflow.start_run():
        mlflow.set_tag("developer", "dlp")
        mlflow.xgboost.autolog(disable=True)
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        mlflow.log_metric("train_rmse", rmse_train)

        mlflow.log_metric("valid_rmse", rmse_valid)

    return rmse_valid


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
y_valid = df_val[target].values

# Convert the training and validation sets to DMatrix format for XGBoost
X_train = xgb.DMatrix(X_train, label=y_train)
X_valid = xgb.DMatrix(X_val, label=y_valid)

# Create an example DMatrix for logging with MLflow
data_example = xgb.DMatrix(X_val.toarray()[0:10,:], y_valid[0:10])

# Start an MLflow run and disable autologging for XGBoost
# This is necessary to manually log parameters and metrics for hyperparameter tuning
# If autologging is enabled, it will interfere with the manual logging of parameters and metrics
# This is because autologging will automatically log parameters and metrics for each trial,
# which will result in duplicate entries in the MLflow run
# and will make it difficult to track the best trial
# Therefore, we disable autologging for the XGBoost model during hyperparameter tuning
# After hyperparameter tuning, we will enable autologging again to log the best model
# and its parameters and metrics
# This way, we can still use autologging for the final model, while manually logging
# parameters and metrics for hyperparameter tuning
# This approach allows us to have more control over the hyperparameter tuning process
# and to log only the best parameters and metrics for the final model
# This is a common practice in machine learning projects to ensure that the best model is logged
# and to avoid cluttering the MLflow run with unnecessary entries
# This is especially important when using Optuna for hyperparameter tuning,
# as it will create multiple trials with different parameters and metrics,
# and we want to log only the best trial's parameters and metrics for the final model
# This will also help in tracking the best model and its performance on the validation set
# It is also important to note that we will log the preprocessor as an artifact,
# which will allow us to use the same preprocessor for the final model
# and to ensure that the final model can be used for inference on new data

# Set up Optuna for hyperparameter tuning
func = lambda trial:objective_XGB(trial, X_train, X_valid, y_train, y_valid)
study = optuna.create_study(direction = "minimize")
study.optimize(func, n_trials = 3, timeout=None, n_jobs=-1)


best_trial = study.best_trial
print(f"Best Score on Validation Set: {best_trial.value}")
print("Number of finished trials: ", len(study.trials))
print(f"Best trial: {best_trial.number}")
print("Params: ")
for key, value in best_trial.params.items():
    #mlflow.log_param(key, value)
    print("    {}: {}".format(key, value))

# Retrain model using parameters from best run
best_model = xgb.train(
    params=best_trial.params,
    dtrain=X_train,
    evals=[(X_valid, "valid")],
    verbose_eval=50, # Every 50 rounds
    early_stopping_rounds=10,
    num_boost_round=100,
    )

y_pred = best_model.predict(X_train)
rmse_train = root_mean_squared_error(y_train, y_pred)

y_pred = best_model.predict(X_valid)
rmse_valid = root_mean_squared_error(y_valid, y_pred)

with open("models/preprocessor.b", "wb") as f_out:
    pickle.dump(dv, f_out)

with mlflow.start_run():
    mlflow.xgboost.autolog(disable=True)
    mlflow.set_tag("developer", "dlp")
    mlflow.log_param("train-data-path", training_data_path)
    mlflow.log_param("valid-data-path", validation_data_path)
    mlflow.set_tag("model", "xgboost")
    mlflow.log_params(best_trial.params)
    mlflow.log_metric("train_rmse", rmse_train)
    mlflow.log_metric("valid_rmse", rmse_valid)
    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
    mlflow.xgboost.log_model(best_model, artifact_path="models_mlflow")


# mlflow.sklearn.autolog()

# for model_class in (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, LinearSVR):

#     with mlflow.start_run():

#         mlflow.log_param("train-data-path", training_data_path)
#         mlflow.log_param("valid-data-path", validation_data_path)
#         mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

#         mlmodel = model_class()
#         mlmodel.fit(X_train, y_train)

#         y_pred = mlmodel.predict(X_val)
#         rmse = root_mean_squared_error(y_val, y_pred)
#         mlflow.log_metric("rmse", rmse)
# pass


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_parquet('~/Data/green_tripdata_2021-01.parquet')