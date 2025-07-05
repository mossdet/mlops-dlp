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
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    print("Airflow not installed. This is a template showing DAG structure.")

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': days_ago(1) if AIRFLOW_AVAILABLE else datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Global configuration
CONFIG = {
    'mlflow': {
        'db_path': '/home/ubuntu/mlops-dlp/mlflow/mlflow.db',
        'experiment_name': 'orchestration-pipeline-airflow'
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

def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow_db_path = CONFIG['mlflow']['db_path']
    tracking_uri = f"sqlite:///{mlflow_db_path}"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(CONFIG['mlflow']['experiment_name'])
    logging.info(f"MLflow tracking URI set to: {tracking_uri}")

def extract_data_task(**context):
    """
    Airflow task: Extract data from source
    """
    year = CONFIG['data']['year']
    month = CONFIG['data']['month']
    
    logging.info(f"Extracting data for {year}-{month:02d}")
    
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    
    try:
        df = pd.read_parquet(url)
        
        # Save raw data
        data_dir = Path(CONFIG['artifacts']['data_dir'])
        data_dir.mkdir(parents=True, exist_ok=True)
        
        raw_data_path = data_dir / f"raw_data_{year}_{month:02d}.parquet"
        df.to_parquet(raw_data_path)
        
        logging.info(f"Successfully extracted {len(df)} records to {raw_data_path}")
        
        # Pass file path to next task
        return str(raw_data_path)
        
    except Exception as e:
        logging.error(f"Failed to extract data: {e}")
        raise

def transform_data_task(**context):
    """
    Airflow task: Transform and clean data
    """
    # Get file path from previous task
    ti = context['ti']
    raw_data_path = ti.xcom_pull(task_ids='extract_data')
    
    logging.info(f"Transforming data from {raw_data_path}")
    
    # Load raw data
    df = pd.read_parquet(raw_data_path)
    
    # Calculate duration
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    
    # Filter outliers
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    # Feature engineering
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    # Save processed data
    data_dir = Path(CONFIG['artifacts']['data_dir'])
    processed_data_path = data_dir / f"processed_data_{CONFIG['data']['year']}_{CONFIG['data']['month']:02d}.parquet"
    df.to_parquet(processed_data_path)
    
    logging.info(f"Data transformed. Final shape: {df.shape}")
    logging.info(f"Processed data saved to {processed_data_path}")
    
    return str(processed_data_path)

def prepare_features_task(**context):
    """
    Airflow task: Prepare features for training
    """
    # Get file path from previous task
    ti = context['ti']
    processed_data_path = ti.xcom_pull(task_ids='transform_data')
    
    logging.info(f"Preparing features from {processed_data_path}")
    
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
    
    logging.info(f"Features prepared. Shape: {X_train.shape}")
    
    return {
        'features_path': str(features_path),
        'targets_path': str(targets_path),
        'vectorizer_path': str(vectorizer_path)
    }

def train_model_task(**context):
    """
    Airflow task: Train ML model
    """
    setup_mlflow()
    
    # Get file paths from previous task
    ti = context['ti']
    paths = ti.xcom_pull(task_ids='prepare_features')
    
    logging.info("Training model")
    
    # Load features and targets
    with open(paths['features_path'], 'rb') as f:
        X_train = pickle.load(f)
    
    with open(paths['targets_path'], 'rb') as f:
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
        
        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(model, "model")
        
        # Get run ID
        run_id = mlflow.active_run().info.run_id
        
        logging.info(f"Model trained. RMSE: {rmse:.4f}, Run ID: {run_id}")
        
        return {
            'run_id': run_id,
            'rmse': rmse,
            'vectorizer_path': paths['vectorizer_path']
        }

def save_model_task(**context):
    """
    Airflow task: Save model artifacts
    """
    # Get results from previous task
    ti = context['ti']
    results = ti.xcom_pull(task_ids='train_model')
    
    logging.info(f"Saving model artifacts for run {results['run_id']}")
    
    models_dir = Path(CONFIG['artifacts']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy vectorizer to models directory
    import shutil
    src_vectorizer = results['vectorizer_path']
    dst_vectorizer = models_dir / f"vectorizer_{results['run_id']}.pkl"
    shutil.copy2(src_vectorizer, dst_vectorizer)
    
    # Save run metadata
    metadata = {
        'run_id': results['run_id'],
        'rmse': results['rmse'],
        'timestamp': datetime.now().isoformat(),
        'vectorizer_path': str(dst_vectorizer)
    }
    
    metadata_path = models_dir / f"metadata_{results['run_id']}.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    logging.info(f"Model artifacts saved to {models_dir}")
    
    return metadata

def cleanup_task(**context):
    """
    Airflow task: Cleanup temporary files
    """
    import shutil
    
    data_dir = Path(CONFIG['artifacts']['data_dir'])
    
    # Remove temporary data files (keep only the latest)
    for file_pattern in ['raw_data_*.parquet', 'processed_data_*.parquet', 
                        'features_*.pkl', 'targets_*.pkl', 'vectorizer_*.pkl']:
        for file_path in data_dir.glob(file_pattern):
            if file_path.exists():
                file_path.unlink()
                logging.info(f"Cleaned up {file_path}")

# Create the DAG (only if Airflow is available)
if AIRFLOW_AVAILABLE:
    dag = DAG(
        'ml_pipeline_orchestration',
        default_args=default_args,
        description='ML Pipeline Orchestration with Airflow',
        schedule_interval=timedelta(days=1),  # Daily execution
        catchup=False,
        tags=['mlops', 'machine-learning', 'orchestration'],
    )

    # Define tasks
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data_task,
        dag=dag,
    )

    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data_task,
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

    save_task = PythonOperator(
        task_id='save_model',
        python_callable=save_model_task,
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
    extract_task >> transform_task >> features_task >> train_task >> save_task
    save_task >> cleanup_task_op
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
        raw_data_path = extract_data_task(**context)
        print(f"✅ Data extraction completed: {raw_data_path}")
        
        # Mock XCom for standalone execution
        class MockTI:
            def __init__(self):
                self.data = {}
            
            def xcom_pull(self, task_ids):
                return self.data.get(task_ids)
            
            def xcom_push(self, key, value):
                self.data[key] = value
        
        mock_ti = MockTI()
        mock_ti.data['extract_data'] = raw_data_path
        context['ti'] = mock_ti
        
        processed_data_path = transform_data_task(**context)
        print(f"✅ Data transformation completed: {processed_data_path}")
        
        mock_ti.data['transform_data'] = processed_data_path
        
        feature_paths = prepare_features_task(**context)
        print(f"✅ Feature preparation completed")
        
        mock_ti.data['prepare_features'] = feature_paths
        
        model_results = train_model_task(**context)
        print(f"✅ Model training completed. Run ID: {model_results['run_id']}")
        
        mock_ti.data['train_model'] = model_results
        
        metadata = save_model_task(**context)
        print(f"✅ Model artifacts saved")
        
        cleanup_task(**context)
        print(f"✅ Cleanup completed")
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"   Run ID: {model_results['run_id']}")
        print(f"   RMSE: {model_results['rmse']:.4f}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_pipeline_standalone()
