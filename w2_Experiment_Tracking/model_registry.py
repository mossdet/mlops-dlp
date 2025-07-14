#!/usr/bin/env python
"""
MLflow Model Registry Management

This script demonstrates model registry operations with MLflow including
model versioning, staging, and lifecycle management.

Author: Daniel Lachner-Piza
Email: dalapiz@proton.me
"""

import time
import os
import mlflow
import pandas as pd
import pickle
import duration_prediction
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from pathlib import Path
from datetime import datetime
from sklearn.metrics import root_mean_squared_error
from datetime import datetime

class MLflowModelRegistry:
    def __init__(self, tracking_uri):
        """
        Initialize the MLflowModelRegistry with a tracking URI.
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    
    # Exemplify use fo MlflowClient
    def search_experiments(self, name:str=None,):
        """
        List all experiments in the MLflow tracking server.
        """

        print(f"Searching for experiments with name: {name}")

        # If no name is provided, list all experiments
        filter_string=None
        if name is not None:
            # If a name is provided, filter experiments by name
            filter_string = f"name = '{name}'"

        experiments = self.client.search_experiments(view_type=ViewType.ALL, filter_string=filter_string)
        for exp in experiments:
            print(f"Experiment ID: {exp.experiment_id}, Name: {exp.name}, Artifact Location: {exp.artifact_location}")

        pass
        return experiments

    def create_experiment(self, name, artifact_location=None):
        """
        Create a new experiment in the MLflow tracking server.
        """

        print(f"Creating experiment with name: {name} and artifact location: {artifact_location}")
        
        # If artifact_location is not provided, use the default location
        try:
            experiment_id = self.client.create_experiment(name, artifact_location=artifact_location)
            print(f"Experiment created with ID: {experiment_id}")
            return experiment_id
        except Exception as e:
            print(f"Error creating experiment: {e}")
            return None
        
    def search_runs(self, experiment_id:str=None, filter_string:str=None, max_results:int=5, order_by:str=None):
        """
        Search for runs in a specific experiment.
        """
        
        print(f"Searching for runs in experiment ID: {experiment_id} with filter: {filter_string}")

        try:
            # If filter_string is not provided, search all runs in the experiment            
            runs = self.client.search_runs(
                experiment_ids=experiment_id,
                filter_string=filter_string,
                run_view_type=ViewType.ALL,
                max_results=max_results,
                order_by=[order_by]
            )
            for run in runs:
                start_time = datetime.fromtimestamp(run.info.start_time / 1000)
                end_time = datetime.fromtimestamp(run.info.end_time / 1000)
                duration = end_time - start_time
                duration_s = duration.total_seconds()

                # Extract model class from tags
                model_class = 'N/A'
                if 'model' in list(run.data.tags.keys()):
                    model_class = run.data.tags['model']
                elif 'estimator_class' in list(run.data.tags.keys()):
                    model_class = run.data.tags['estimator_class']
                
                print(f"Run ID: {run.info.run_id},\n \
                      Status: {run.info.status},\n \
                      Start Time: {start_time},\n \
                      Duration(s): {duration_s},\n \
                      Model: {model_class}\n \
                      RMSE: {run.data.metrics['rmse']:.4f} \
                    ")
                pass

            print(f"Total runs found: {len(runs)}")
            return runs
        except Exception as e:
            print(f"Error searching runs: {e}")
            return None
    
    def delete_experiment(self, experiment_id):
        """
        Delete an experiment by its ID.
        """
        
        print(f"Deleting experiment with ID: {experiment_id}")

        try:
            self.client.delete_experiment(experiment_id)
            print(f"Experiment with ID {experiment_id} deleted successfully.")
        except Exception as e:
            print(f"Error deleting experiment: {e}")
    
    def register_model(self, model_uri, name, tags=None, description=None):
        """
        Register a model in the MLflow Model Registry.
        """
        
        print(f"Registering model from URI: {model_uri} with name: {name}")

        try:
            model_details = mlflow.register_model(model_uri=model_uri, name=name)
            print(f"Model registered with name: {model_details.name}, version: {model_details.version}")
            return model_details
        except Exception as e:
            print(f"Error registering model: {e}")
            return None

    def get_latest_registered_model_versions(self, model_name):
        """
        Get the latest versions of a registered model.
        """
        print(f"Getting latest versions for model: {model_name}")

        try:
            #model_versions = self.client.get_latest_versions(name=model_name, stages=["Production", "Staging"])
            model_versions = self.client.get_latest_versions(name=model_name, stages=["None"])
            for version in model_versions:
                print(f"Model Name: {version.name}, Version: {version.version}, Stage: {version.current_stage}, Status: {version.status}")
            print(f"Total model versions found: {len(model_versions)}")
            if len(model_versions) == 0:
                print(f"No model versions found for model: {model_name}")
                return None
            return model_versions
        except Exception as e:
            print(f"Error getting latest model versions: {e}")
            return None
        
    def set_registered_model_alias(self, name, alias, version):
        """
        Set an alias for a registered model version.
        """
        
        print(f"Setting alias: {alias} for model: {name}, version: {version}")

        try:
            self.client.set_registered_model_alias(name=name, alias=alias, version=version)
            print(f"Alias {alias} set for model {name}, version {version}")
        except Exception as e:
            print(f"Error setting alias: {e}")
    
    def update_model_version(self, name, version, description):
        """
        Update metadata associated with a model version in backend.
        """
        
        print(f"Updating model version: {version} for model: {name} with description: {description}")

        try:
            self.client.update_model_version(name=name, version=version, description=description)
            print(f"Model version {version} for model {name} updated with description: {description}")
        except Exception as e:
            print(f"Error updating model version: {e}")

    def retrieve_artifact_from_run(self, run_id, path, dst_path):
        """
        Retrieve an artifact from a run.
        """

        print(f"Retrieving artifact from run: {run_id}")

        try:
            self.client.download_artifacts(run_id=run_id, path=path, dst_path=dst_path)
            print(f"Artifact downloaded to: {dst_path}")
            with open(dst_path, "rb") as f_in:
                dv = pickle.load(f_in)
            print(f"Artifact loaded successfully from: {dst_path}")
            # Return the dictionary vectorizer object
            return dv
        except Exception as e:
            print(f"Error retrieving artifact: {e}")

    def test_registered_model(self, experiment_name, model_name, model_version, X_test, y_test):
        """
        Test a registered model by name and alias.
        """
        
        print(f"\nTesting registered model: {model_name}, Version: {model_version}")

        try:

            # Get the model uri using model name and version
            model_uri = self.client.get_model_version_download_uri(model_name, model_version) + 's/'

            # Load the model from the URI
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Model loaded successfully from URI: {model_uri}")
            
            # Extract metadata from the model
            loader_module = model.metadata.flavors['python_function']['loader_module']
            model_size_mb = model.metadata._model_size_bytes/1000/1000
            model_run_id = model.metadata.run_id
            model_experiment_id = self.search_experiments(experiment_name)[0].experiment_id

            # Extract model class from tags
            filter_string = f"attributes.run_id='{model_run_id}'"
            runs = self.client.search_runs(experiment_ids=model_experiment_id, filter_string=filter_string)
            model_class = 'N/A'
            if 'model' in list(runs[0].data.tags.keys()):
                model_class = runs[0].data.tags['model']
            elif 'estimator_class' in list(runs[0].data.tags.keys()):
                model_class = runs[0].data.tags['estimator_class']
            
            print(f"Model run ID: {model_run_id}")
            print(f"Model experiment ID: {model_experiment_id}")
            print(f"Model-Source: {loader_module}, Model type: {model_class}, Model size: {model_size_mb:.2f} MB")

            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            end_time = time.time()
            duration = end_time - start_time
            print(f"Prediction completed in {duration:.2f} seconds")

            rmse = root_mean_squared_error(y_test, y_pred)
            print(f"RMSE for model {model_name} with version {model_version}: {rmse:.4f}")
            return rmse
        except Exception as e:
            print(f"Error testing registered model: {e}")   

        

def register_models():
    """
    Example function to register models to the MLflow Model Registry.
    """
    # Set the MLflow tracking URI
    mlflow_df_path = "/home/ubuntu/mlops-dlp/mlflow/mlflow.db"
    MLFLOW_TRACKING_URI = f"sqlite:///{mlflow_df_path}"

    registry = MLflowModelRegistry(MLFLOW_TRACKING_URI)
    registry.search_experiments('nyc-taxi-experiment')
    registry.create_experiment('test-experiment-1')
    registry.search_experiments()

    best_runs = registry.search_runs(experiment_id='1', filter_string="metrics.rmse < 7", max_results=10, order_by="metrics.rmse ASC")

    # #Example of registering a model to the model registry
    # run_id = "43f4a758f3434176a5f6e0d8eb078d05"  # Replace with your actual run ID
    # model_uri = f"runs:/{run_id}/model"
    # registry.register_model(model_uri=model_uri, 
    #                         name="nyc-taxi-regressor", 
    #                         tags={"Model": "xgb"}, 
    #                         description="XGB Regressor with best RMSE for NYC Taxi duration prediction")

    # run_id = "e5b767ad112c4fff9b46514e273d79e1"  # Replace with your actual run ID
    # model_uri = f"runs:/{run_id}/model"
    # registry.register_model(model_uri=model_uri, 
    #                         name="nyc-taxi-regressor", 
    #                         tags={"Model": "GradientBoostingRegressor"}, 
    #                         description="GradientBoostingRegressor with best RMSE for NYC Taxi duration prediction")

    # run_id = "c4cd7e6ceb374817ae2594f79305c1bf"  # Replace with your actual run ID
    # model_uri = f"runs:/{run_id}/model"
    # registry.register_model(model_uri=model_uri, 
    #                         name="nyc-taxi-regressor", 
    #                         tags={"Model": "RandomForestRegressor"}, 
    #                         description="RandomForestRegressor with best RMSE for NYC Taxi duration prediction")

    registry.get_latest_registered_model_versions("nyc-taxi-regressor")
    
    model_version = "15"
    model_type = "XGBRegressor"
    registry.set_registered_model_alias(name="nyc-taxi-regressor", alias=f"Testing{model_version}", version=model_version)
    registry.update_model_version(name="nyc-taxi-regressor", version=model_version, description=f"Updated description for version {model_version}. {model_type} with best RMSE for NYC Taxi duration prediction. RMSE = 6.53")

    model_version = "16"
    model_type = " GradientBoostingRegressor"
    registry.set_registered_model_alias(name="nyc-taxi-regressor", alias=f"Testing{model_version}", version=model_version)
    registry.update_model_version(name="nyc-taxi-regressor", version=model_version, description=f"Updated description for version {model_version}. {model_type} with best RMSE for NYC Taxi duration prediction. RMSE = 6.53")

    model_version = "17"
    model_type = "RandomForestRegressor"
    registry.set_registered_model_alias(name="nyc-taxi-regressor", alias=f"Testing{model_version}", version=model_version)
    registry.update_model_version(name="nyc-taxi-regressor", version=model_version, description=f"Updated description for version {model_version}. {model_type} with best RMSE for NYC Taxi duration prediction. RMSE = 6.53")

    pass

def retrieve_registered_models():
    """
    Example function to retrieve registered models from the MLflow Model Registry.
    """

    # Set the MLflow tracking URI
    mlflow_df_path = "/home/ubuntu/mlops-dlp/mlflow/mlflow.db"
    MLFLOW_TRACKING_URI = f"sqlite:///{mlflow_df_path}"
    mlflow_experiment_name = 'nyc-taxi-experiment'

    # Initialize the data pipeline for NYC Taxi duration prediction
    pipeline = duration_prediction.NYCTaxiDurationPrediction(
        training_data_path='~/Data/green_tripdata_2021-01.parquet',
        validation_data_path='~/Data/green_tripdata_2021-02.parquet',
        test_data_path='~/Data/green_tripdata_2021-03.parquet',
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        mlflow_experiment_name=mlflow_experiment_name,
        models_dir='/home/ubuntu/mlops-dlp/mlflow/models/',
        images_dir='/home/ubuntu/mlops-dlp/mlflow/images/'
    )

    # Initialize the MLflow Model Registry
    registry = MLflowModelRegistry(MLFLOW_TRACKING_URI)

    # Get test data    
    X_train, X_valid, X_test, X_examples, y_train, y_valid, y_test, dv = pipeline.preprocess_data()

    # Retrieve dictionary vectorizer from the model registry
    run_id = "43f4a758f3434176a5f6e0d8eb078d05"
    path = "preprocessor"
    dst_path = '.'
    dv = registry.retrieve_artifact_from_run(run_id, path, dst_path=dst_path)

    # Example of searching for runs in an experiment using their run ID
    experiment = registry.search_experiments(mlflow_experiment_name)
    experiment_id = experiment[0].experiment_id  # Replace with your actual experiment ID
    run_id = '43f4a758f3434176a5f6e0d8eb078d05'
    filter_string = f"attributes.run_id='{run_id}'"
    registry.search_runs(experiment_id=experiment_id, filter_string=filter_string)
    
    model_name = "nyc-taxi-regressor"
    model_version = "15"
    registry.test_registered_model(mlflow_experiment_name, model_name, model_version, X_test, y_test)

    model_name = "nyc-taxi-regressor"
    model_version = "16"
    registry.test_registered_model(mlflow_experiment_name, model_name, model_version, X_test, y_test)

    model_name = "nyc-taxi-regressor"
    model_version = "17"
    registry.test_registered_model(mlflow_experiment_name, model_name, model_version, X_test, y_test)

    model_version = "15"
    model_type = "XGBRegressor"
    registry.set_registered_model_alias(name="nyc-taxi-regressor", alias="Production", version=model_version)
    registry.update_model_version(name="nyc-taxi-regressor", version=model_version, description=f"Updated description for version {model_version}. {model_type} with best RMSE for NYC Taxi duration prediction. RMSE = 6.53")

    pass

if __name__ == "__main__":

    register_models()
    retrieve_registered_models()