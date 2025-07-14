#!/usr/bin/env python
"""
Duration Prediction with Experiment Tracking

ML pipeline for NYC taxi trip duration prediction with MLflow experiment tracking.

Author: Daniel Lachner-Piza
Email: dalapiz@proton.me
"""

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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score


class NYCTaxiDurationPrediction:
    def __init__(self, training_data_path, validation_data_path, test_data_path, mlflow_tracking_uri, mlflow_experiment_name, models_dir="models/", images_dir="images/"):
        self.training_data_path = training_data_path
        self.validation_data_path = validation_data_path
        self.test_data_path = test_data_path
        self.mlflow_tracking_uri=mlflow_tracking_uri
        self.mlflow_experiment_name=mlflow_experiment_name
        self.models_dir = models_dir
        self.images_dir = images_dir
        self.preprocess_path = os.path.join(self.models_dir, 'preprocessor.b')

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        pass

    
    def read_dataframe(self, filename):
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

    def save_preprocessor(self, dv):
        # Save the DictVectorizer to a file for later use
        try:
            with open(self.preprocess_path, 'wb') as f_out:
                pickle.dump(dv, f_out)
            print("Preprocessor saved to:", self.preprocess_path)
        except Exception as e:
            print(f"Error saving preprocessor: {e}")
            pass
    
    def load_preprocessor(self):
        # Load the DictVectorizer from a file
        try:
            with open(self.preprocess_path, 'rb') as f_in:
                dv = pickle.load(f_in)
            print("Preprocessor loaded from:", self.preprocess_path)
            return dv
        except Exception as e:
            print(f"Error loading preprocessor: {e}")
            return None

    def preprocess_data(self):
        # Load the dataset
        df_train = self.read_dataframe(self.training_data_path)
        df_val = self.read_dataframe(self.validation_data_path)
        df_test = self.read_dataframe(self.test_data_path)

        # Create a new feature that combines pickup and dropoff locations
        df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
        df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']
        df_test['PU_DO'] = df_test['PULocationID'] + '_' + df_test['DOLocationID']

        # Select features by data type
        categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
        numerical = ['trip_distance']

        # Convert categorical features to string type
        df_train[categorical] = df_train[categorical].astype(str)
        df_val[categorical] = df_val[categorical].astype(str)
        df_test[categorical] = df_test[categorical].astype(str)

        # Transform the mixed data types into a format suitable for machine learning using DictVectorizer
        # DictVectorizer converts a list of dictionaries into a matrix of features
        # Each dictionary corresponds to a row in the DataFrame, with keys as feature names and values as feature values
        # Numerical features are already in a suitable format, while categorical features need to be one-hot encoded
        dv = DictVectorizer()

        train_dicts = df_train[categorical + numerical].to_dict(orient='records')
        X_train = dv.fit_transform(train_dicts)

        val_dicts = df_val[categorical + numerical].to_dict(orient='records')
        X_valid = dv.transform(val_dicts)

        test_dicts = df_test[categorical + numerical].to_dict(orient='records')
        X_test = dv.transform(test_dicts)

        # Select the target variable
        # The target variable is the duration of the trip in minutes
        target = 'duration'
        y_train = df_train[target].values
        y_valid = df_val[target].values
        y_test = df_test[target].values

        # Create an example DMatrix for logging with MLflow
        X_examples = dv.transform(val_dicts[0:10])

        return X_train, X_valid, X_test, X_examples, y_train, y_valid, y_test, dv

    ### Objective function for Optuna hyperparameter tuning
    def objective_XGB(self, trial, data_train, data_valid, y_train, y_valid):
        
        # Define the hyperparameter search space
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


        model = XGBRegressor(**params)
        model = model.fit(data_train, y_train, eval_set=[(data_valid, y_valid)], verbose=False)
        
        y_pred = model.predict(data_train)
        if np.isnan(y_pred).any():
            return float("inf")  # Return a high value if predictions are NaN
        rmse_train = root_mean_squared_error(y_train, y_pred)

        y_pred = model.predict(data_valid)
        if np.isnan(y_pred).any():
            return float("inf")  # Return a high value if predictions are NaN

        # Calculate RMSE for validation set
        rmse_valid = root_mean_squared_error(y_valid, y_pred)

        return rmse_valid

    def train_xgb_regressor(self, nr_runs, X_train, y_train, X_valid, y_valid, X_examples):
        
        all_runs_best_model = None
        best_rmse = float("inf")
        for repeats in range(nr_runs):
            print(f"Run {repeats + 1} of {nr_runs}")

            # Set up Optuna for hyperparameter tuning
            func = lambda trial:self.objective_XGB(trial, X_train, X_valid, y_train, y_valid)
            study = optuna.create_study(direction = "minimize")
            study.optimize(func, n_trials = 3, timeout=None, n_jobs=1)

            # Print the best trial and its parameters
            best_trial = study.best_trial
            print(f"Best Score on Validation Set: {best_trial.value}")
            print("Number of finished trials: ", len(study.trials))
            print(f"Best trial: {best_trial.number}")
            print("Params: ")
            for key, value in best_trial.params.items():
                print("    {}: {}".format(key, value))

            # Retrain model using parameters from best run
            best_model = XGBRegressor(**best_trial.params)
            best_model = best_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

            y_pred = best_model.predict(X_train)
            rmse_train = root_mean_squared_error(y_train, y_pred)

            y_pred = best_model.predict(X_valid)
            rmse_valid = root_mean_squared_error(y_valid, y_pred)

            if rmse_valid < best_rmse:
                best_rmse = rmse_valid
                all_runs_best_model = best_model

            with mlflow.start_run():
                mlflow.xgboost.autolog(disable=True)
                mlflow.set_tag("developer", "dlp")
                mlflow.log_param("train-data-path", self.training_data_path)
                mlflow.log_param("valid-data-path", self.validation_data_path)
                mlflow.set_tag("model", "xgboost")
                mlflow.log_params(best_trial.params)
                mlflow.log_metric("rmse", rmse_valid)
                mlflow.log_artifact(self.preprocess_path, artifact_path="preprocessor")
                mlflow.xgboost.log_model(best_model, artifact_path="models", input_example=X_examples)

        return all_runs_best_model, best_rmse

    def train_sklearn_regressors(self, X_train, y_train, X_valid, y_valid, X_examples):
        # Enable MLflow autologging for scikit-learn models
        mlflow.sklearn.autolog(disable=False)
        for model_class in (LinearRegression, Lasso, Ridge, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, LinearSVR):
            with mlflow.start_run():
                print(f"Training model: {model_class.__name__}")

                mlflow.set_tag("developer", "dlp")
                mlflow.log_param("train-data-path", self.training_data_path)
                mlflow.log_param("valid-data-path", self.validation_data_path)
                mlflow.log_artifact(self.preprocess_path, artifact_path="preprocessor")

                mlmodel = model_class()
                mlmodel.fit(X_train, y_train)

                y_pred = mlmodel.predict(X_valid)
                rmse = root_mean_squared_error(y_valid, y_pred)
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(mlmodel, artifact_path="models", input_example=X_examples)
    
    def run(self):
        # Preprocess the data
        X_train, X_valid, X_test, X_examples, y_train, y_valid, y_test, dv = self.preprocess_data()

        # Save the preprocessor for later use
        self.save_preprocessor(dv)

        # Train the model with hyperparameter tuning
        best_model, best_rmse = self.train_xgb_regressor(nr_runs=5, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_examples=X_examples)
        try:
            # Optionally, visualize feature importance
            plot_importance(best_model, max_num_features=10)
            plt.savefig(os.path.join(self.images_dir, 'feature_importance.png'))
            plot_tree(best_model, num_trees=0)
            plt.savefig(os.path.join(self.images_dir, 'tree_plot.png'))
            plt.close()

        except Exception as e:
            print(f"Error during feature importance visualization: {e}")

        try:
            # Optionally, visualize the distribution of trip durations
            # create a subplot with two histograms
            plt.figure(figsize=(12, 6))
            sns.histplot(y_train, bins=50, kde=True, color='blue', label='Training Set')
            sns.histplot(y_valid, bins=50, kde=True, color='orange', label='Validation Set')
            plt.title("Distribution of Trip Durations")
            plt.xlabel("Duration (minutes)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.savefig(os.path.join(self.images_dir, 'trip_duration_distribution.png'))
            plt.close()
        except Exception as e:
            print(f"Error during distribution visualization: {e}")


        # Train scikit-learn models
        self.train_sklearn_regressors(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_examples=X_examples)




if __name__ == '__main__':

    # Set the MLflow tracking URI
    mlflow_df_path = "/home/ubuntu/mlops-dlp/mlflow/mlflow.db"
    MLFLOW_TRACKING_URI = f"sqlite:///{mlflow_df_path}"
    mlflow_experiment_name = "nyc-taxi-experiment"

    pipeline = NYCTaxiDurationPrediction(
        training_data_path='~/Data/green_tripdata_2021-01.parquet',
        validation_data_path='~/Data/green_tripdata_2021-02.parquet',
        test_data_path='~/Data/green_tripdata_2021-03.parquet',
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        mlflow_experiment_name=mlflow_experiment_name,
        models_dir='/home/ubuntu/mlops-dlp/mlflow/models/',
        images_dir='/home/ubuntu/mlops-dlp/mlflow/images/'
    )
    pipeline.run()
    