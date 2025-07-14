#!/usr/bin/env python
"""
Prefect Workflow Example for ML Pipeline Orchestration

This script demonstrates how to create a Prefect workflow for orchestrating
ML pipelines with better error handling, retries, and observability.
"""
import time
import pickle
import logging
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

# Import centralized configuration
from config import get_config

# Prefect imports (would be available in a Prefect environment)
try:
    from prefect import flow, task, get_run_logger
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    print("Prefect not installed. This is a template showing workflow structure.")
    
    # Mock decorators for standalone execution
    def flow(func):
        return func
    
    def task(func):
        return func
    
    def get_run_logger():
        return logging.getLogger(__name__)

# Load centralized configuration
CONFIG = get_config().get_script_config('prefect')

@task(retries=3, retry_delay_seconds=30)
def setup_mlflow_task():
    """Setup MLflow tracking"""
    logger = get_run_logger()
    
    os.environ["AWS_PROFILE"] = CONFIG['mlflow']['aws_profile']
    tracking_server_host = CONFIG['mlflow']['tracking_server_host']
    tracking_uri = f"http://{tracking_server_host}:5000"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(CONFIG['mlflow']['experiment_name'])
    
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    return tracking_uri

@task(retries=2, retry_delay_seconds=60)
def extract_data_task(year: int, month: int) -> str:
    """
    Extract data from source with retry logic
    
    Args:
        year: Year of the data
        month: Month of the data
        
    Returns:
        Path to saved raw data file
    """
    logger = get_run_logger()
    
    logger.info(f"Extracting data for {year}-{month:02d}")
    
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    
    try:
        df = pd.read_parquet(url)
        
        # Save raw data
        data_dir = Path(CONFIG['artifacts']['data_dir'])
        data_dir.mkdir(parents=True, exist_ok=True)
        
        raw_data_path = data_dir / f"raw_data_{year}_{month:02d}.parquet"
        df.to_parquet(raw_data_path)
        
        logger.info(f"Successfully extracted {len(df)} records to {raw_data_path}")
        
        return str(raw_data_path)
        
    except Exception as e:
        logger.error(f"Failed to extract data: {e}")
        raise

@task(retries=2, retry_delay_seconds=60)
def extract_validation_data_task(year: int, month: int) -> str:
    """
    Extract validation data from next month
    
    Args:
        year: Year of the training data
        month: Month of the training data
        
    Returns:
        Path to saved validation data file
    """
    logger = get_run_logger()
    
    # Calculate next month for validation
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    
    logger.info(f"Extracting validation data for {next_year}-{next_month:02d}")
    
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{next_year}-{next_month:02d}.parquet'
    
    try:
        df = pd.read_parquet(url)
        
        # Save raw validation data
        data_dir = Path(CONFIG['artifacts']['data_dir'])
        data_dir.mkdir(parents=True, exist_ok=True)
        
        raw_val_data_path = data_dir / f"raw_val_data_{next_year}_{next_month:02d}.parquet"
        df.to_parquet(raw_val_data_path)
        
        logger.info(f"Successfully extracted {len(df)} validation records to {raw_val_data_path}")
        
        return str(raw_val_data_path)
        
    except Exception as e:
        logger.error(f"Failed to extract validation data: {e}")
        raise

@task
def transform_validation_data_task(raw_val_data_path: str, year: int, month: int) -> str:
    """
    Transform and clean validation data
    
    Args:
        raw_val_data_path: Path to raw validation data file
        year: Training year for naming
        month: Training month for naming
        
    Returns:
        Path to processed validation data file
    """
    logger = get_run_logger()
    
    logger.info(f"Transforming validation data from {raw_val_data_path}")
    
    # Load raw validation data
    df = pd.read_parquet(raw_val_data_path)
    original_size = len(df)
    
    # Calculate duration
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    
    # Filter outliers
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    filtered_size = len(df)
    
    # Feature engineering
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    # Save processed validation data
    data_dir = Path(CONFIG['artifacts']['data_dir'])
    processed_val_data_path = data_dir / f"processed_val_data_{year}_{month:02d}.parquet"
    df.to_parquet(processed_val_data_path)
    
    logger.info(f"Validation data transformed. Original: {original_size}, Filtered: {filtered_size} ({filtered_size/original_size:.1%})")
    logger.info(f"Processed validation data saved to {processed_val_data_path}")
    
    return str(processed_val_data_path)

@task
def validate_data_task(raw_data_path: str) -> bool:
    """
    Validate extracted data quality
    
    Args:
        raw_data_path: Path to raw data file
        
    Returns:
        True if data passes validation
    """
    logger = get_run_logger()
    
    logger.info(f"Validating data from {raw_data_path}")
    
    df = pd.read_parquet(raw_data_path)
    
    # Data quality checks
    checks = {
        'non_empty': len(df) > 0,
        'has_required_columns': all(col in df.columns for col in 
                                  ['lpep_pickup_datetime', 'lpep_dropoff_datetime', 
                                   'PULocationID', 'DOLocationID', 'trip_distance']),
        'no_all_null_rows': not df.isnull().all(axis=1).any(),
        'reasonable_size': 1000 <= len(df) <= 10000000  # Between 1K and 10M records
    }
    
    passed_checks = sum(checks.values())
    total_checks = len(checks)
    
    logger.info(f"Data validation: {passed_checks}/{total_checks} checks passed")
    
    for check_name, result in checks.items():
        logger.info(f"  {check_name}: {'✅' if result else '❌'}")
    
    if passed_checks != total_checks:
        raise ValueError(f"Data validation failed: {passed_checks}/{total_checks} checks passed")
    
    return True

@task
def transform_data_task(raw_data_path: str) -> str:
    """
    Transform and clean data
    
    Args:
        raw_data_path: Path to raw data file
        
    Returns:
        Path to processed data file
    """
    logger = get_run_logger()
    
    logger.info(f"Transforming data from {raw_data_path}")
    
    # Load raw data
    df = pd.read_parquet(raw_data_path)
    original_size = len(df)
    
    # Calculate duration
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    
    # Filter outliers
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    filtered_size = len(df)
    
    # Feature engineering
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    # Save processed data
    data_dir = Path(CONFIG['artifacts']['data_dir'])
    processed_data_path = data_dir / f"processed_data_{CONFIG['data']['year']}_{CONFIG['data']['month']:02d}.parquet"
    df.to_parquet(processed_data_path)
    
    logger.info(f"Data transformed. Original: {original_size}, Filtered: {filtered_size} ({filtered_size/original_size:.1%})")
    logger.info(f"Processed data saved to {processed_data_path}")
    
    return str(processed_data_path)

@task
def prepare_features_task(train_data_path: str, val_data_path: str) -> Dict[str, str]:
    """
    Prepare features for training and validation
    
    Args:
        train_data_path: Path to training data file
        val_data_path: Path to validation data file
        
    Returns:
        Dictionary of file paths for features, targets, and vectorizer
    """
    logger = get_run_logger()
    
    logger.info(f"Preparing features from {train_data_path} and {val_data_path}")
    
    # Load processed data
    df_train = pd.read_parquet(train_data_path)
    df_val = pd.read_parquet(val_data_path)
    
    # Use only PU_DO as categorical feature (not separate PULocationID and DOLocationID)
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    
    # Create DictVectorizer with sparse=True to match reference script
    dv = DictVectorizer(sparse=True)
    
    # Prepare training features
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    y_train = df_train.duration.values
    
    # Prepare validation features using the same vectorizer
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_val = df_val.duration.values
    
    # Save features and vectorizer
    data_dir = Path(CONFIG['artifacts']['data_dir'])
    
    features_train_path = data_dir / f"features_train_{CONFIG['data']['year']}_{CONFIG['data']['month']:02d}.pkl"
    targets_train_path = data_dir / f"targets_train_{CONFIG['data']['year']}_{CONFIG['data']['month']:02d}.pkl"
    features_val_path = data_dir / f"features_val_{CONFIG['data']['year']}_{CONFIG['data']['month']:02d}.pkl"
    targets_val_path = data_dir / f"targets_val_{CONFIG['data']['year']}_{CONFIG['data']['month']:02d}.pkl"
    vectorizer_path = data_dir / f"vectorizer_{CONFIG['data']['year']}_{CONFIG['data']['month']:02d}.pkl"
    
    # Save using pickle
    with open(features_train_path, 'wb') as f:
        pickle.dump(X_train, f)
    
    with open(targets_train_path, 'wb') as f:
        pickle.dump(y_train, f)
    
    with open(features_val_path, 'wb') as f:
        pickle.dump(X_val, f)
    
    with open(targets_val_path, 'wb') as f:
        pickle.dump(y_val, f)
        
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(dv, f)
    
    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Validation features shape: {X_val.shape}")
    logger.info(f"Feature density: {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.3f}")
    
    return {
        'features_train_path': str(features_train_path),
        'targets_train_path': str(targets_train_path),
        'features_val_path': str(features_val_path),
        'targets_val_path': str(targets_val_path),
        'vectorizer_path': str(vectorizer_path)
    }

@task
def train_model_task(feature_paths: Dict[str, str], tracking_uri: str) -> Dict[str, Any]:
    """
    Train ML model with MLflow tracking using XGBoost native API
    
    Args:
        feature_paths: Dictionary of file paths
        tracking_uri: MLflow tracking URI
        
    Returns:
        Dictionary with training results
    """
    logger = get_run_logger()
    
    logger.info("Training model")
    
    # Load features and targets
    with open(feature_paths['features_train_path'], 'rb') as f:
        X_train = pickle.load(f)
    
    with open(feature_paths['targets_train_path'], 'rb') as f:
        y_train = pickle.load(f)
    
    with open(feature_paths['features_val_path'], 'rb') as f:
        X_val = pickle.load(f)
    
    with open(feature_paths['targets_val_path'], 'rb') as f:
        y_val = pickle.load(f)
    
    with open(feature_paths['vectorizer_path'], 'rb') as f:
        dv = pickle.load(f)
    
    with mlflow.start_run() as run:

        start_time = time.time()
        # Create DMatrix objects for XGBoost native API
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        
        # Model parameters (matching reference script)
        best_params = CONFIG['model']['params']
        
        # Log parameters
        mlflow.log_params(best_params)
        
        # Train model using XGBoost native API
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=CONFIG['model']['num_boost_round'],
            evals=[(valid, 'validation')],
            early_stopping_rounds=CONFIG['model']['early_stopping_rounds']
        )
        
        # Make predictions on validation set
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        
        # Save preprocessor
        models_dir = Path('/home/ubuntu/mlops-dlp/week03/mlflow/models/')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        preprocessor_path = models_dir / "preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        
        # Log artifacts with error handling
        try:
            mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")
            logger.info("✅ Preprocessor artifact logged successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log preprocessor artifact: {e}")
        
        # Log model with error handling  
        try:
            mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
            logger.info("✅ Model logged successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log model: {e}")
        
        # Get run ID
        run_id = run.info.run_id
        
        logger.info(f"Model trained successfully")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  Run ID: {run_id}")

        duration = time.time() - start_time
        
        return {
            'run_id': run_id,
            'rmse': rmse,            
            'booster': booster,
            'vectorizer_path': feature_paths['vectorizer_path']
        }

@task
def validate_model_task(training_results: Dict[str, Any]) -> bool:
    """
    Validate trained model meets quality thresholds
    
    Args:
        training_results: Results from model training
        
    Returns:
        True if model passes validation
    """
    logger = get_run_logger()
    
    logger.info("Validating model quality")
    
    rmse = training_results['rmse']
    
    # Quality thresholds
    max_rmse = 10.0  # Maximum acceptable RMSE
    min_rmse = 0.1   # Minimum reasonable RMSE (should not be suspiciously low)
    
    checks = {
        'rmse_threshold': rmse <= max_rmse,
        'reasonable_rmse': rmse >= min_rmse
    }
    
    passed_checks = sum(checks.values())
    total_checks = len(checks)
    
    logger.info(f"Model validation: {passed_checks}/{total_checks} checks passed")
    
    for check_name, result in checks.items():
        logger.info(f"  {check_name}: {'✅' if result else '❌'}")
    
    if passed_checks != total_checks:
        logger.warning(f"Model validation failed: {passed_checks}/{total_checks} checks passed")
        return False
    
    logger.info("Model passed all quality checks")
    return True

@task
def save_model_artifacts_task(training_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save model artifacts and metadata
    
    Args:
        training_results: Results from model training
        
    Returns:
        Metadata about saved artifacts
    """
    logger = get_run_logger()
    
    logger.info(f"Saving model artifacts for run {training_results['run_id']}")
    
    models_dir = Path(CONFIG['artifacts']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy vectorizer to models directory
    import shutil
    src_vectorizer = training_results['vectorizer_path']
    dst_vectorizer = models_dir / f"vectorizer_{training_results['run_id']}.pkl"
    shutil.copy2(src_vectorizer, dst_vectorizer)
    
    # Save run metadata
    metadata = {
        'run_id': training_results['run_id'],
        'rmse': training_results['rmse'],
        'timestamp': datetime.now().isoformat(),
        'vectorizer_path': str(dst_vectorizer),
        'model_config': CONFIG['model']
    }
    
    metadata_path = models_dir / f"metadata_{training_results['run_id']}.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"Model artifacts saved to {models_dir}")
    logger.info(f"  Vectorizer: {dst_vectorizer}")
    logger.info(f"  Metadata: {metadata_path}")
    
    return metadata

@task
def cleanup_task():
    """Cleanup temporary files"""
    logger = get_run_logger()
    
    data_dir = Path(CONFIG['artifacts']['data_dir'])
    
    # Remove temporary data files
    cleanup_patterns = ['raw_data_*.parquet', 'raw_val_data_*.parquet', 
                       'processed_data_*.parquet', 'processed_val_data_*.parquet',
                       'features_*.pkl', 'targets_*.pkl', 'vectorizer_*.pkl']
    
    cleaned_files = 0
    for pattern in cleanup_patterns:
        for file_path in data_dir.glob(pattern):
            if file_path.exists():
                file_path.unlink()
                cleaned_files += 1
    
    logger.info(f"Cleaned up {cleaned_files} temporary files")

@flow(name="ML Pipeline with Prefect")
def ml_pipeline_flow(year: Optional[int] = None, month: Optional[int] = None, tracking_server_host: Optional[str] = None, aws_profile: Optional[str] = None):
    """
    Main ML pipeline flow
    
    Args:
        year: Year of the data to process (overrides config)
        month: Month of the data to process (overrides config)
        tracking_server_host: MLflow tracking server host (overrides config)
        aws_profile: AWS profile (overrides config)
    """
    logger = get_run_logger()
    
    # Get configuration manager and update if parameters provided
    config_manager = get_config()
    
    if tracking_server_host or aws_profile:
        config_manager.update_mlflow_settings(
            tracking_server_host=tracking_server_host,
            aws_profile=aws_profile
        )
    
    if year or month:
        config_manager.update_data_settings(year=year, month=month)
    
    # Get updated configuration
    global CONFIG
    CONFIG = config_manager.get_script_config('prefect')
    
    actual_year = CONFIG['data']['year']
    actual_month = CONFIG['data']['month']
    
    logger.info(f"Starting ML pipeline for {actual_year}-{actual_month:02d}")
    logger.info(f"Using MLflow server: {CONFIG['mlflow']['tracking_server_host']}")
    logger.info(f"Using AWS profile: {CONFIG['mlflow']['aws_profile']}")
    
    # Setup
    tracking_uri = setup_mlflow_task()
    
    # Data pipeline
    raw_data_path = extract_data_task(actual_year, actual_month)
    raw_val_data_path = extract_validation_data_task(actual_year, actual_month)
    
    data_valid = validate_data_task(raw_data_path)
    
    if not data_valid:
        raise ValueError("Data validation failed")
    
    processed_data_path = transform_data_task(raw_data_path)
    processed_val_data_path = transform_validation_data_task(raw_val_data_path, actual_year, actual_month)
    feature_paths = prepare_features_task(processed_data_path, processed_val_data_path)
    
    # Model pipeline
    training_results = train_model_task(feature_paths, tracking_uri)
    model_valid = validate_model_task(training_results)
    
    if not model_valid:
        logger.warning("Model validation failed, but continuing with artifact saving")
    
    # Artifact management
    metadata = save_model_artifacts_task(training_results)
    run_id_path = save_run_id_task(training_results['run_id'])
    cleanup_task()
    
    logger.info("Pipeline completed successfully")
    logger.info(f"  Run ID: {training_results['run_id']}")
    logger.info(f"  RMSE: {training_results['rmse']:.4f}")
    logger.info(f"  Model validation: {'✅' if model_valid else '❌'}")
    
    return {
        'success': True,
        'run_id': training_results['run_id'],
        'rmse': training_results['rmse'],
        'model_valid': model_valid,
        'metadata': metadata
    }

@task
def save_run_id_task(run_id: str):
    """Save run ID to file like in reference script"""
    logger = get_run_logger()
    
    run_id_fpath = Path(os.path.dirname(os.path.abspath(__file__))) / "run_id.txt"
    run_id_fpath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(run_id_fpath, "w") as f:
        f.write(run_id)
    
    logger.info(f"Run ID saved to {run_id_fpath}")
    
    return str(run_id_fpath)

# Standalone execution
def run_standalone():
    """Run the pipeline standalone for testing"""
    print("Running Prefect pipeline standalone...")
    
    try:
        result = ml_pipeline_flow(year=2021, month=1)
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"   Run ID: {result['run_id']}")
        print(f"   RMSE: {result['rmse']:.4f}")
        print(f"   Model validation: {'✅' if result['model_valid'] else '❌'}")
        
        return result
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    # Set to False for actual runs, True for testing purposes
    testing = False
    
    if PREFECT_AVAILABLE:
        print("Prefect is available. You can:")
        print("1. Run standalone: python prefect_pipeline.py")
        print("2. Deploy to Prefect server: prefect deployment build-from-flow prefect_pipeline.py:ml_pipeline_flow")
    else:
        print("Prefect not installed. Running standalone version...")
        print("To install Prefect: pip install prefect")
    
    if testing:
        # Use configuration defaults for testing
        config_manager = get_config()
        config = config_manager.get_config()
        result = ml_pipeline_flow(
            year=config.get('data', {}).get('year'),
            month=config.get('data', {}).get('month'),
            tracking_server_host=config.get('mlflow', {}).get('tracking_server_host'),
            aws_profile=config.get('mlflow', {}).get('aws_profile')
        )
        print(f"MLflow run_id: {result['run_id']}")
    else:
        # Use command line arguments for production
        parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
        parser.add_argument('--year', type=int, help='Year of the data to train on (overrides config)')
        parser.add_argument('--month', type=int, help='Month of the data to train on (overrides config)')
        parser.add_argument('--tracking-server-host', type=str, help='Tracking server hostname (overrides config)')
        parser.add_argument('--aws-profile', type=str, help='AWS profile name (overrides config)')
        args = parser.parse_args()

        result = ml_pipeline_flow(
            year=args.year, 
            month=args.month, 
            tracking_server_host=args.tracking_server_host, 
            aws_profile=args.aws_profile
        )
        print(f"MLflow run_id: {result['run_id']}")
