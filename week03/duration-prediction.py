#!/usr/bin/env python
# coding: utf-8

# Standard library
import os
import pickle
import argparse
from pathlib import Path

# Third-party libraries
import pandas as pd
import xgboost as xgb
import mlflow
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

# Set the MLflow tracking URI
mlflow_df_path = "/home/ubuntu/mlops-dlp/mlflow/mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{mlflow_df_path}"
mlflow_experiment_name = "nyc-taxi-experiment"
models_dir=Path('/home/ubuntu/mlops-dlp/mlflow/models/')
images_dir=Path('/home/ubuntu/mlops-dlp/mlflow/images/')


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(mlflow_experiment_name)
models_dir.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(parents=True, exist_ok=True)

# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("nyc-taxi-experiment")



def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:squarederror',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        preprocessor_path = models_dir/"preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":

    # Set to False for actual runs, True for testing purposes
    # If testing is True, it will use a fixed year and month for training
    testing = True
    if testing:
        year = 2021
        month = 1
        run_id = run(year=year, month=month)
    else:
        parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
        parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
        parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
        args = parser.parse_args()

        run_id = run(year=args.year, month=args.month)

    run_id_fpath = Path(os.path.dirname(os.path.abspath(__file__))) / "run_id.txt"
    run_id_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(run_id_fpath, "w") as f:
        f.write(run_id)