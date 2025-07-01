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
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import Ridge



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
        "objective":"reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 10, 40, step=1),
        "max_depth": trial.suggest_int("max_depth", 2, 40, step=1),
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
        "eval_metric": trial.suggest_categorical("eval_metric", ["rmse"]),
        "tree_method": trial.suggest_categorical("tree_method", ["hist"]),
        "random_state": trial.suggest_categorical("random_state", [42]),
        "early_stopping_rounds": trial.suggest_categorical("early_stopping_rounds", [50]),
    }

    # params['num_boost_round'] = 200
    # params['early_stopping_rounds'] = 50
    # params['seed'] = 42  # Fixed seed for reproducibility

    model = XGBRegressor(**params)
    model = model.fit(data_train, y_train, eval_set=[(data_valid, y_valid)], verbose=False)
    
    # model = xgb.train(
    #     params=params,
    #     dtrain=data_train,
    #     evals=[(data_valid, "valid")],
    #     verbose_eval=50,
    #     num_boost_round=200,
    #     early_stopping_rounds=50,
    # )


    y_pred = model.predict(data_train)
    if np.isnan(y_pred).any():
        return float("inf")  # Return a high value if predictions are NaN
    rmse_train = root_mean_squared_error(y_train, y_pred)

    y_pred = model.predict(data_valid)
    if np.isnan(y_pred).any():
        return float("inf")  # Return a high value if predictions are NaN

    # Calculate RMSE for validation set
    rmse_valid = root_mean_squared_error(y_valid, y_pred)

    # with mlflow.start_run():
    #     mlflow.set_tag("developer", "dlp")
    #     mlflow.xgboost.autolog(disable=True)
    #     mlflow.set_tag("model", "xgboost")
    #     mlflow.log_params(params)
    #     mlflow.log_metric("train_rmse", rmse_train)

    #     mlflow.log_metric("valid_rmse", rmse_valid)

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
dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_valid = dv.transform(val_dicts)

os.makedirs("models", exist_ok=True)
with open("models/preprocessor.b", "wb") as f_out:
    pickle.dump(dv, f_out)

# Select the target variable
# The target variable is the duration of the trip in minutes
target = 'duration'
y_train = df_train[target].values
y_valid = df_val[target].values

# Create an example DMatrix for logging with MLflow
X_examples = dv.transform(val_dicts[0:10])

for repeats in range(10):
    print(f"Run {repeats + 1} of 10")

    # Set up Optuna for hyperparameter tuning
    func = lambda trial:objective_XGB(trial, X_train, X_valid, y_train, y_valid)
    study = optuna.create_study(direction = "minimize")
    study.optimize(func, n_trials = 10, timeout=None, n_jobs=-1)

    # Print the best trial and its parameters
    best_trial = study.best_trial
    print(f"Best Score on Validation Set: {best_trial.value}")
    print("Number of finished trials: ", len(study.trials))
    print(f"Best trial: {best_trial.number}")
    print("Params: ")
    for key, value in best_trial.params.items():
        #mlflow.log_param(key, value)
        print("    {}: {}".format(key, value))

    # Retrain model using parameters from best run
    best_model = XGBRegressor(**best_trial.params)
    best_model = best_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    y_pred = best_model.predict(X_train)
    rmse_train = root_mean_squared_error(y_train, y_pred)

    y_pred = best_model.predict(X_valid)
    rmse_valid = root_mean_squared_error(y_valid, y_pred)

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
        mlflow.xgboost.log_model(best_model, artifact_path="models_mlflow", input_example=X_examples)


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_parquet('~/Data/green_tripdata_2021-01.parquet')