#!/usr/bin/env python
"""
Prefect Workflow Example for ML Pipeline Orchestration

This script demonstrates how to create a Prefect workflow for orchestrating
ML pipelines with better error handling, retries, and observability.
"""

import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

# Prefect imports (would be available in a Prefect environment)
try:
    from prefect import flow, task, get_run_logger
    from prefect.task_runners import SequentialTaskRunner
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

# Configuration
CONFIG = {
    'mlflow': {
        'db_path': '/home/ubuntu/mlops-dlp/mlflow/mlflow.db',
        'experiment_name': 'orchestration-pipeline-prefect'
    },
    'data': {
        'year': 2023,
        'month': 1
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
        'models_dir': '/home/ubuntu/mlops-dlp/mlflow/models',
        'data_dir': '/home/ubuntu/mlops-dlp/data'
    }
}

@task(retries=3, retry_delay_seconds=30)
def setup_mlflow_task():
    """Setup MLflow tracking"""
    logger = get_run_logger()
    
    mlflow_db_path = CONFIG['mlflow']['db_path']
    tracking_uri = f"sqlite:///{mlflow_db_path}"
    
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
def prepare_features_task(processed_data_path: str) -> Dict[str, str]:
    """
    Prepare features for training
    
    Args:
        processed_data_path: Path to processed data file
        
    Returns:
        Dictionary of file paths for features, targets, and vectorizer
    """
    logger = get_run_logger()
    
    logger.info(f"Preparing features from {processed_data_path}")
    
    # Load processed data
    df = pd.read_parquet(processed_data_path)
    
    categorical = ['PULocationID', 'DOLocationID', 'PU_DO']
    numerical = ['trip_distance']
    
    dv = DictVectorizer()
    
    train_dicts = df[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values
    
    # Save features and vectorizer
    data_dir = Path(CONFIG['artifacts']['data_dir'])
    
    features_path = data_dir / f"features_{CONFIG['data']['year']}_{CONFIG['data']['month']:02d}.pkl"
    targets_path = data_dir / f"targets_{CONFIG['data']['year']}_{CONFIG['data']['month']:02d}.pkl"
    vectorizer_path = data_dir / f"vectorizer_{CONFIG['data']['year']}_{CONFIG['data']['month']:02d}.pkl"
    
    # Save using pickle
    with open(features_path, 'wb') as f:
        pickle.dump(X_train, f)
    
    with open(targets_path, 'wb') as f:
        pickle.dump(y_train, f)
        
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(dv, f)
    
    logger.info(f"Features prepared. Shape: {X_train.shape}")
    logger.info(f"Feature density: {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.3f}")
    
    return {
        'features_path': str(features_path),
        'targets_path': str(targets_path),
        'vectorizer_path': str(vectorizer_path)
    }

@task
def train_model_task(feature_paths: Dict[str, str], tracking_uri: str) -> Dict[str, Any]:
    """
    Train ML model with MLflow tracking
    
    Args:
        feature_paths: Dictionary of file paths
        tracking_uri: MLflow tracking URI
        
    Returns:
        Dictionary with training results
    """
    logger = get_run_logger()
    
    logger.info("Training model")
    
    # Load features and targets
    with open(feature_paths['features_path'], 'rb') as f:
        X_train = pickle.load(f)
    
    with open(feature_paths['targets_path'], 'rb') as f:
        y_train = pickle.load(f)
    
    with mlflow.start_run():
        # Model parameters
        params = CONFIG['model']['params']
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_train)
        rmse = root_mean_squared_error(y_train, y_pred)
        
        # Additional metrics
        mean_duration = y_train.mean()
        std_duration = y_train.std()
        
        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics({
            "rmse": rmse,
            "mean_duration": mean_duration,
            "std_duration": std_duration,
            "relative_rmse": rmse / mean_duration
        })
        mlflow.xgboost.log_model(model, "model")
        
        # Get run ID
        run_id = mlflow.active_run().info.run_id
        
        logger.info(f"Model trained successfully")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  Mean duration: {mean_duration:.2f} minutes")
        logger.info(f"  Relative RMSE: {rmse/mean_duration:.3f}")
        logger.info(f"  Run ID: {run_id}")
        
        return {
            'run_id': run_id,
            'rmse': rmse,
            'mean_duration': mean_duration,
            'relative_rmse': rmse / mean_duration,
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
    relative_rmse = training_results['relative_rmse']
    
    # Quality thresholds
    max_rmse = 10.0  # Maximum acceptable RMSE
    max_relative_rmse = 0.5  # Maximum relative RMSE (50% of mean)
    
    checks = {
        'rmse_threshold': rmse <= max_rmse,
        'relative_rmse_threshold': relative_rmse <= max_relative_rmse,
        'reasonable_rmse': rmse > 0.1  # Should be reasonable but not suspiciously low
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
        'mean_duration': training_results['mean_duration'],
        'relative_rmse': training_results['relative_rmse'],
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
    cleanup_patterns = ['raw_data_*.parquet', 'processed_data_*.parquet', 
                       'features_*.pkl', 'targets_*.pkl', 'vectorizer_*.pkl']
    
    cleaned_files = 0
    for pattern in cleanup_patterns:
        for file_path in data_dir.glob(pattern):
            if file_path.exists():
                file_path.unlink()
                cleaned_files += 1
    
    logger.info(f"Cleaned up {cleaned_files} temporary files")

@flow(name="ML Pipeline with Prefect", 
      task_runner=SequentialTaskRunner() if PREFECT_AVAILABLE else None)
def ml_pipeline_flow(year: int = 2023, month: int = 1):
    """
    Main ML pipeline flow
    
    Args:
        year: Year of the data to process
        month: Month of the data to process
    """
    logger = get_run_logger()
    
    logger.info(f"Starting ML pipeline for {year}-{month:02d}")
    
    # Update config with parameters
    CONFIG['data']['year'] = year
    CONFIG['data']['month'] = month
    
    # Setup
    tracking_uri = setup_mlflow_task()
    
    # Data pipeline
    raw_data_path = extract_data_task(year, month)
    data_valid = validate_data_task(raw_data_path)
    
    if not data_valid:
        raise ValueError("Data validation failed")
    
    processed_data_path = transform_data_task(raw_data_path)
    feature_paths = prepare_features_task(processed_data_path)
    
    # Model pipeline
    training_results = train_model_task(feature_paths, tracking_uri)
    model_valid = validate_model_task(training_results)
    
    if not model_valid:
        logger.warning("Model validation failed, but continuing with artifact saving")
    
    # Artifact management
    metadata = save_model_artifacts_task(training_results)
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

# Standalone execution
def run_standalone():
    """Run the pipeline standalone for testing"""
    print("Running Prefect pipeline standalone...")
    
    try:
        result = ml_pipeline_flow(year=2023, month=1)
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"   Run ID: {result['run_id']}")
        print(f"   RMSE: {result['rmse']:.4f}")
        print(f"   Model validation: {'✅' if result['model_valid'] else '❌'}")
        
        return result
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    if PREFECT_AVAILABLE:
        print("Prefect is available. You can:")
        print("1. Run standalone: python prefect_pipeline.py")
        print("2. Deploy to Prefect server: prefect deployment build-from-flow prefect_pipeline.py:ml_pipeline_flow")
    else:
        print("Prefect not installed. Running standalone version...")
        print("To install Prefect: pip install prefect")
    
    run_standalone()
