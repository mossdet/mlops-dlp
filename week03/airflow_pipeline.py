#!/usr/bin/env python
"""
Apache Airflow DAG Example for ML Pipeline Orchestration

This script demonstrates how to create an Airflow DAG for orchestrating
ML workflows with proper task dependencies and error handling.
"""

from datetime import datetime, timedelta
from pathlib import Path
import pickle
import logging
import os

import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

# Airflow imports (would be available in an Airflow environment)
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    try:
        import pendulum
        start_date = pendulum.today('UTC').add(days=-1)
    except ImportError:
        from airflow.utils.dates import days_ago
        start_date = days_ago(1)
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    start_date = datetime(2023, 1, 1)
    print("Airflow not installed. This is a template showing DAG structure.")

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': start_date,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Global configuration
CONFIG = {
    'mlflow': {
        'tracking_server_host': 'ec2-18-223-115-201.us-east-2.compute.amazonaws.com',  # EC2 MLflow server
        'aws_profile': 'mlops_zc',  # AWS profile for authentication
        'experiment_name': 'nyc-taxi-experiment'  # Match reference script
    },
    'data': {
        'year': 2021,  # Updated default to match reference script
        'month': 1
    },
    'model': {
        'params': {
            # Optimized hyperparameters from reference script
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:squarederror',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        },
        'num_boost_round': 30,
        'early_stopping_rounds': 50
    },
    'artifacts': {
        'models_dir': '/home/ubuntu/mlops-dlp/mlflow/models',
        'data_dir': '/home/ubuntu/mlops-dlp/data'
    }
}

def setup_mlflow():
    """Setup MLflow tracking"""
    os.environ["AWS_PROFILE"] = CONFIG['mlflow']['aws_profile']
    tracking_server_host = CONFIG['mlflow']['tracking_server_host']
    tracking_uri = f"http://{tracking_server_host}:5000"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(CONFIG['mlflow']['experiment_name'])
    logging.info(f"MLflow tracking URI set to: {tracking_uri}")

def read_dataframe(year: int, month: int) -> pd.DataFrame:
    """
    Data extraction and basic transformation (matching reference script)
    """
    logging.info(f"Reading dataframe for {year}-{month:02d}")
    
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    
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
    
    return df

def create_X(df: pd.DataFrame, dv=None):
    """
    Create feature matrix (matching reference script)
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

def extract_data_task(**context):
    """
    Airflow task: Extract training and validation data
    """
    year = CONFIG['data']['year']
    month = CONFIG['data']['month']
    
    logging.info(f"Extracting training data for {year}-{month:02d}")
    
    try:
        # Extract training data
        df_train = read_dataframe(year, month)
        
        # Extract validation data (next month)
        next_year = year if month < 12 else year + 1
        next_month = month + 1 if month < 12 else 1
        logging.info(f"Extracting validation data for {next_year}-{next_month:02d}")
        df_val = read_dataframe(next_year, next_month)
        
        # Save data
        data_dir = Path(CONFIG['artifacts']['data_dir'])
        data_dir.mkdir(parents=True, exist_ok=True)
        
        train_data_path = data_dir / f"train_data_{year}_{month:02d}.parquet"
        val_data_path = data_dir / f"val_data_{next_year}_{next_month:02d}.parquet"
        
        df_train.to_parquet(train_data_path)
        df_val.to_parquet(val_data_path)
        
        logging.info(f"Successfully extracted {len(df_train)} training and {len(df_val)} validation records")
        
        # Return both paths
        return {
            'train_data_path': str(train_data_path),
            'val_data_path': str(val_data_path)
        }
        
    except Exception as e:
        logging.error(f"Failed to extract data: {e}")
        raise

def prepare_features_task(**context):
    """
    Airflow task: Prepare features for training and validation
    """
    # Get file paths from previous task
    ti = context['ti']
    data_paths = ti.xcom_pull(task_ids='extract_data')
    
    train_data_path = data_paths['train_data_path']
    val_data_path = data_paths['val_data_path']
    
    logging.info(f"Preparing features from {train_data_path} and {val_data_path}")
    
    # Load data
    df_train = pd.read_parquet(train_data_path)
    df_val = pd.read_parquet(val_data_path)
    
    # Prepare features (matching reference script)
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)
    
    # Get target values
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    
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
    
    logging.info(f"Training features shape: {X_train.shape}")
    logging.info(f"Validation features shape: {X_val.shape}")
    
    return {
        'features_train_path': str(features_train_path),
        'targets_train_path': str(targets_train_path),
        'features_val_path': str(features_val_path),
        'targets_val_path': str(targets_val_path),
        'vectorizer_path': str(vectorizer_path)
    }

def train_model_task(**context):
    """
    Airflow task: Train ML model (matching reference script)
    """
    setup_mlflow()
    
    # Get file paths from previous task
    ti = context['ti']
    paths = ti.xcom_pull(task_ids='prepare_features')
    
    logging.info("Training model")
    
    # Load features and targets
    with open(paths['features_train_path'], 'rb') as f:
        X_train = pickle.load(f)
    
    with open(paths['targets_train_path'], 'rb') as f:
        y_train = pickle.load(f)
    
    with open(paths['features_val_path'], 'rb') as f:
        X_val = pickle.load(f)
    
    with open(paths['targets_val_path'], 'rb') as f:
        y_val = pickle.load(f)
    
    with open(paths['vectorizer_path'], 'rb') as f:
        dv = pickle.load(f)
    
    with mlflow.start_run() as run:
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

        # Save preprocessor (matching reference script)
        models_dir = Path('/home/ubuntu/mlops-dlp/week03/mlflow/models/')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        preprocessor_path = models_dir / "preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")

        # Log model
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        
        run_id = run.info.run_id
        
        logging.info(f"Model trained successfully. RMSE: {rmse:.4f}")
        logging.info(f"Run ID: {run_id}")
        
        return run_id

def save_run_id_task(**context):
    """
    Airflow task: Save run ID to file (matching reference script)
    """
    # Get run_id from previous task
    ti = context['ti']
    run_id = ti.xcom_pull(task_ids='train_model')
    
    logging.info(f"Saving run ID: {run_id}")
    
    # Save run ID to file (matching reference script)
    run_id_fpath = Path(os.path.dirname(os.path.abspath(__file__))) / "run_id.txt"
    run_id_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(run_id_fpath, "w") as f:
        f.write(run_id)
    
    logging.info(f"Run ID saved to {run_id_fpath}")
    return str(run_id_fpath)

def cleanup_task(**context):
    """
    Airflow task: Cleanup temporary files
    """
    import shutil
    
    data_dir = Path(CONFIG['artifacts']['data_dir'])
    
    if not data_dir.exists():
        logging.info("Data directory does not exist, no cleanup needed")
        return
    
    # Remove temporary data files
    year = CONFIG['data']['year']
    month = CONFIG['data']['month']
    
    file_patterns = [
        f'train_data_{year}_{month:02d}.parquet',
        f'val_data_*_{month:02d}.parquet',
        f'features_train_{year}_{month:02d}.pkl',
        f'targets_train_{year}_{month:02d}.pkl',
        f'features_val_{year}_{month:02d}.pkl',
        f'targets_val_{year}_{month:02d}.pkl',
        f'vectorizer_{year}_{month:02d}.pkl'
    ]
    
    cleaned_files = 0
    for pattern in file_patterns:
        for file_path in data_dir.glob(pattern):
            if file_path.exists():
                file_path.unlink()
                logging.info(f"Cleaned up {file_path}")
                cleaned_files += 1
    
    logging.info(f"Cleanup completed: {cleaned_files} files removed")

# Create the DAG (only if Airflow is available)
if AIRFLOW_AVAILABLE:
    dag = DAG(
        'ml_pipeline_orchestration',
        default_args=default_args,
        description='ML Pipeline Orchestration with Airflow',
        schedule=timedelta(days=1),  # Daily execution (updated from schedule_interval)
        catchup=False,
        tags=['mlops', 'machine-learning', 'orchestration'],
    )

    # Define tasks
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data_task,
        dag=dag,
    )

    features_task = PythonOperator(
        task_id='prepare_features',
        python_callable=prepare_features_task,
        dag=dag,
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task,
        dag=dag,
    )

    save_run_id_task_op = PythonOperator(
        task_id='save_run_id',
        python_callable=save_run_id_task,
        dag=dag,
    )

    cleanup_task_op = PythonOperator(
        task_id='cleanup',
        python_callable=cleanup_task,
        dag=dag,
        trigger_rule='all_done',  # Run even if upstream tasks fail
    )

    # Health check task
    health_check = BashOperator(
        task_id='health_check',
        bash_command='echo "Pipeline health check passed"',
        dag=dag,
    )

    # Define task dependencies
    extract_task >> features_task >> train_task >> save_run_id_task_op >> cleanup_task_op
    health_check >> extract_task

else:
    print("This is an Airflow DAG template. Install Apache Airflow to run this pipeline.")
    print("pip install apache-airflow")

# Standalone execution function for testing
def run_pipeline_standalone():
    """Run the pipeline standalone for testing"""
    import tempfile
    
    print("Running pipeline standalone (without Airflow)...")
    
    # Create temporary context
    context = {'ti': None}
    
    try:
        # Run tasks sequentially
        data_paths = extract_data_task(**context)
        print(f"✅ Data extraction completed: {data_paths}")
        
        # Mock XCom for standalone execution
        class MockTI:
            def __init__(self):
                self.data = {}
            
            def xcom_pull(self, task_ids):
                return self.data.get(task_ids)
            
            def xcom_push(self, key, value):
                self.data[key] = value
        
        mock_ti = MockTI()
        mock_ti.data['extract_data'] = data_paths
        context['ti'] = mock_ti
        
        feature_paths = prepare_features_task(**context)
        print(f"✅ Feature preparation completed")
        
        mock_ti.data['prepare_features'] = feature_paths
        
        run_id = train_model_task(**context)
        print(f"✅ Model training completed. Run ID: {run_id}")
        
        mock_ti.data['train_model'] = run_id
        
        run_id_path = save_run_id_task(**context)
        print(f"✅ Run ID saved to: {run_id_path}")
        
        cleanup_task(**context)
        print(f"✅ Cleanup completed")
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"   Run ID: {run_id}")
        print(f"   Run ID file: {run_id_path}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apache Airflow ML Pipeline Orchestration")
    parser.add_argument('--year', type=int, default=2021, help='Year for data processing (default: 2021)')
    parser.add_argument('--month', type=int, default=1, help='Month for data processing (default: 1)')
    parser.add_argument('--tracking-server-host', type=str, 
                       default='ec2-18-223-115-201.us-east-2.compute.amazonaws.com',
                       help='MLflow tracking server host (default: EC2 instance)')
    parser.add_argument('--aws-profile', type=str, default='mlops_zc',
                       help='AWS profile for authentication (default: mlops_zc)')
    
    args = parser.parse_args()
    
    # Update CONFIG with command line arguments
    CONFIG['data']['year'] = args.year
    CONFIG['data']['month'] = args.month
    CONFIG['mlflow']['tracking_server_host'] = args.tracking_server_host
    CONFIG['mlflow']['aws_profile'] = args.aws_profile
    
    print(f"Configuration:")
    print(f"  Year: {args.year}")
    print(f"  Month: {args.month}")
    print(f"  MLflow Tracking Server: {args.tracking_server_host}")
    print(f"  AWS Profile: {args.aws_profile}")
    print()
    
    run_pipeline_standalone()
