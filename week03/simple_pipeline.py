#!/usr/bin/env python
"""
Simple ML Pipeline Orchestration Example

This script demonstrates a basic ML pipeline orchestration approach
without external orchestration tools, showing the fundamental concepts.
"""

import os
import pickle
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLPipeline:
    """Simple ML Pipeline class for orchestrating ML workflows"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        os.environ["AWS_PROFILE"] = self.config['mlflow']['aws_profile']
        tracking_server_host = self.config['mlflow']['tracking_server_host']
        tracking_uri = f"http://{tracking_server_host}:5000"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    
    def read_dataframe(self, year: int, month: int) -> pd.DataFrame:
        """
        Data extraction and basic transformation (matching reference script)
        
        Args:
            year: Year of the data
            month: Month of the data
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Reading dataframe for {year}-{month:02d}")
        
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
        
        try:
            df = pd.read_parquet(url)
            
            # Calculate duration
            df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
            df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
            
            # Filter outliers
            df = df[(df.duration >= 1) & (df.duration <= 60)]
            
            # Feature engineering (matching reference script)
            categorical = ['PULocationID', 'DOLocationID']
            df[categorical] = df[categorical].astype(str)
            df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
            
            logger.info(f"Successfully loaded and processed {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Failed to extract data: {e}")
            raise
    
    def create_X(self, df: pd.DataFrame, dv=None):
        """
        Create feature matrix (matching reference script)
        
        Args:
            df: DataFrame with features
            dv: Optional pre-fitted DictVectorizer
            
        Returns:
            Feature matrix and DictVectorizer
        """
        categorical = ['PU_DO']  # Only PU_DO as categorical, matching reference
        numerical = ['trip_distance']
        dicts = df[categorical + numerical].to_dict(orient='records')

        if dv is None:
            dv = DictVectorizer(sparse=True)  # sparse=True matching reference
            X = dv.fit_transform(dicts)
        else:
            X = dv.transform(dicts)

        return X, dv
    
    def train_model(self, X_train, y_train, X_val, y_val, dv):
        """
        Model training step (matching reference script)
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            dv: DictVectorizer for saving
            
        Returns:
            MLflow run ID
        """
        logger.info("Training model")
        
        with mlflow.start_run() as run:
            # Create DMatrix objects for XGBoost native API
            train = xgb.DMatrix(X_train, label=y_train)
            valid = xgb.DMatrix(X_val, label=y_val)

            # Model parameters (matching reference script)
            best_params = self.config['model']['params']

            # Log parameters
            mlflow.log_params(best_params)

            # Train model using XGBoost native API
            booster = xgb.train(
                params=best_params,
                dtrain=train,
                num_boost_round=30,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )

            # Make predictions on validation set
            y_pred = booster.predict(valid)
            rmse = root_mean_squared_error(y_val, y_pred)
            
            # Log metrics
            mlflow.log_metric("rmse", rmse)

            # Save preprocessor (matching reference script)
            models_dir = Path('/home/ubuntu/mlops-dlp/week03/mlflow/models/')
            models_dir.mkdir(parents=True, exist_ok=True)
            
            preprocessor_path = models_dir / "preprocessor.b"
            with open(preprocessor_path, "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")

            # Log model
            mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
            
            logger.info(f"Model trained successfully. RMSE: {rmse:.4f}")
            
            return run.info.run_id
    
    def run(self, year: int, month: int) -> str:
        """
        Run the complete ML pipeline (matching reference script structure)
        
        Args:
            year: Year of the data
            month: Month of the data
            
        Returns:
            MLflow run ID
        """
        logger.info(f"Starting ML pipeline for {year}-{month:02d}")
        
        try:
            # Extract training data
            df_train = self.read_dataframe(year=year, month=month)

            # Extract validation data (next month)
            next_year = year if month < 12 else year + 1
            next_month = month + 1 if month < 12 else 1
            df_val = self.read_dataframe(year=next_year, month=next_month)

            # Prepare features
            X_train, dv = self.create_X(df_train)
            X_val, _ = self.create_X(df_val, dv)

            # Get target values
            target = 'duration'
            y_train = df_train[target].values
            y_val = df_val[target].values

            # Train model
            run_id = self.train_model(X_train, y_train, X_val, y_val, dv)
            
            logger.info(f"Pipeline completed successfully. Run ID: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main execution function"""
    
    # Set to False for actual runs, True for testing purposes
    testing = True
    
    # Default configuration
    tracking_server_host = "ec2-18-223-115-201.us-east-2.compute.amazonaws.com"
    aws_profile = "mlops_zc"
    
    # Pipeline configuration
    config = {
        'mlflow': {
            'tracking_server_host': tracking_server_host,
            'aws_profile': aws_profile,
            'experiment_name': 'nyc-taxi-experiment'  # Match reference script
        },
        'model': {
            'params': {
                'learning_rate': 0.09585355369315604,
                'max_depth': 30,
                'min_child_weight': 1.060597050922164,
                'objective': 'reg:squarederror',
                'reg_alpha': 0.018060244040060163,
                'reg_lambda': 0.011658731377413597,
                'seed': 42
            }
        },
        'artifacts': {
            'models_dir': '/home/ubuntu/mlops-dlp/mlflow/models'
        }
    }
    
    # Initialize pipeline
    pipeline = MLPipeline(config)
    
    if testing:
        # Use fixed year and month for testing (matching reference script)
        year = 2021
        month = 1
        run_id = pipeline.run(year=year, month=month)
        print(f"MLflow run_id: {run_id}")
    else:
        # Use command line arguments for production
        parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
        parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
        parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
        args = parser.parse_args()

        run_id = pipeline.run(year=args.year, month=args.month)
        print(f"MLflow run_id: {run_id}")

    # Save run ID to file (matching reference script)
    run_id_fpath = Path(os.path.dirname(os.path.abspath(__file__))) / "run_id.txt"
    run_id_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(run_id_fpath, "w") as f:
        f.write(run_id)


if __name__ == "__main__":
    main()
