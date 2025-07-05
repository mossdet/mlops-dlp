#!/usr/bin/env python
"""
Simple ML Pipeline Orchestration Example

This script demonstrates a basic ML pipeline orchestration approach
without external orchestration tools, showing the fundamental concepts.
"""

import os
import pickle
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
        mlflow_db_path = self.config['mlflow']['db_path']
        tracking_uri = f"sqlite:///{mlflow_db_path}"
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    
    def extract_data(self, year: int, month: int) -> pd.DataFrame:
        """
        Data extraction step
        
        Args:
            year: Year of the data
            month: Month of the data
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Extracting data for {year}-{month:02d}")
        
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
        
        try:
            df = pd.read_parquet(url)
            logger.info(f"Successfully loaded {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Failed to extract data: {e}")
            raise
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Data transformation step
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Transforming data")
        
        # Calculate duration
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
        
        # Filter outliers
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        
        # Feature engineering
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
        
        logger.info(f"Data transformed. Final shape: {df.shape}")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[Dict, Dict, DictVectorizer]:
        """
        Feature preparation step
        
        Args:
            df: Transformed DataFrame
            
        Returns:
            Training features, target values, and vectorizer
        """
        logger.info("Preparing features")
        
        categorical = ['PULocationID', 'DOLocationID', 'PU_DO']
        numerical = ['trip_distance']
        
        dv = DictVectorizer()
        
        train_dicts = df[categorical + numerical].to_dict(orient='records')
        X_train = dv.fit_transform(train_dicts)
        y_train = df.duration.values
        
        logger.info(f"Features prepared. Shape: {X_train.shape}")
        
        return X_train, y_train, dv
    
    def train_model(self, X_train, y_train) -> xgb.XGBRegressor:
        """
        Model training step
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        logger.info("Training model")
        
        with mlflow.start_run():
            # Model parameters
            params = self.config['model']['params']
            
            # Train model
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_train)
            rmse = root_mean_squared_error(y_train, y_pred)
            
            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
            mlflow.xgboost.log_model(model, "model")
            
            logger.info(f"Model trained. RMSE: {rmse:.4f}")
            
            return model
    
    def save_artifacts(self, model, vectorizer, run_id: str):
        """
        Save model artifacts
        
        Args:
            model: Trained model
            vectorizer: Feature vectorizer
            run_id: MLflow run ID
        """
        logger.info("Saving artifacts")
        
        models_dir = Path(self.config['artifacts']['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save vectorizer
        vectorizer_path = models_dir / f"vectorizer_{run_id}.pkl"
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        logger.info(f"Artifacts saved to {models_dir}")
    
    def run_pipeline(self, year: int, month: int) -> str:
        """
        Run the complete ML pipeline
        
        Args:
            year: Year of the data
            month: Month of the data
            
        Returns:
            MLflow run ID
        """
        logger.info(f"Starting ML pipeline for {year}-{month:02d}")
        
        try:
            # Extract data
            raw_data = self.extract_data(year, month)
            
            # Transform data
            processed_data = self.transform_data(raw_data)
            
            # Prepare features
            X_train, y_train, vectorizer = self.prepare_features(processed_data)
            
            # Train model
            model = self.train_model(X_train, y_train)
            
            # Get current run ID
            run_id = mlflow.active_run().info.run_id
            
            # Save artifacts
            self.save_artifacts(model, vectorizer, run_id)
            
            logger.info(f"Pipeline completed successfully. Run ID: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main execution function"""
    
    # Pipeline configuration
    config = {
        'mlflow': {
            'db_path': '/home/ubuntu/mlops-dlp/mlflow/mlflow.db',
            'experiment_name': 'orchestration-pipeline-simple'
        },
        'model': {
            'params': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42
            }
        },
        'artifacts': {
            'models_dir': '/home/ubuntu/mlops-dlp/mlflow/models'
        }
    }
    
    # Initialize pipeline
    pipeline = MLPipeline(config)
    
    # Run pipeline for January 2023
    run_id = pipeline.run_pipeline(year=2023, month=1)
    
    print(f"Pipeline completed with run ID: {run_id}")


if __name__ == "__main__":
    main()
