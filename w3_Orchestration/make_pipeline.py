#!/usr/bin/env python
"""
Make.py - Simple Make-like Task Runner for ML Pipelines

This script provides a simple task runner similar to GNU Make,
allowing you to define and execute ML pipeline tasks with dependencies.
"""

import os
import sys
import time
import json
import pickle
import hashlib
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional

import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

# Import centralized configuration
from config import get_config

class TaskRunner:
    """Simple task runner with dependency management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tasks = {}
        self.cache_dir = Path(config.get('cache_dir', '.cache'))
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup MLflow
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        os.environ["AWS_PROFILE"] = self.config['mlflow']['aws_profile']
        tracking_server_host = self.config['mlflow']['tracking_server_host']
        tracking_uri = f"http://{tracking_server_host}:5000"
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        print(f"MLflow tracking URI: {tracking_uri}")
    
    def task(self, name: str, depends_on: List[str] = None, 
             inputs: List[str] = None, outputs: List[str] = None):
        """Decorator to register a task"""
        def decorator(func: Callable):
            self.tasks[name] = {
                'func': func,
                'depends_on': depends_on or [],
                'inputs': inputs or [],
                'outputs': outputs or [],
                'last_run': None,
                'cache_key': None
            }
            return func
        return decorator
    
    def get_file_hash(self, filepath: str) -> str:
        """Get hash of a file for change detection"""
        if not os.path.exists(filepath):
            return ""
        
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def get_task_cache_key(self, task_name: str) -> str:
        """Generate cache key for a task based on inputs and dependencies"""
        task = self.tasks[task_name]
        
        # Hash inputs
        input_hashes = []
        for inp in task['inputs']:
            if os.path.exists(inp):
                input_hashes.append(self.get_file_hash(inp))
        
        # Hash dependencies
        dep_hashes = []
        for dep in task['depends_on']:
            if dep in self.tasks:
                dep_hashes.append(self.tasks[dep].get('cache_key', ''))
        
        # Hash task configuration
        config_hash = hashlib.md5(str(task).encode()).hexdigest()
        
        combined = ''.join(input_hashes + dep_hashes + [config_hash])
        return hashlib.md5(combined.encode()).hexdigest()
    
    def should_run_task(self, task_name: str) -> bool:
        """Check if task should be run based on dependencies and cache"""
        task = self.tasks[task_name]
        
        # Check if outputs exist
        for output in task['outputs']:
            if not os.path.exists(output):
                print(f"Task '{task_name}': Output {output} missing")
                return True
        
        # Check if cache key changed
        current_cache_key = self.get_task_cache_key(task_name)
        if task['cache_key'] != current_cache_key:
            print(f"Task '{task_name}': Dependencies changed")
            return True
        
        print(f"Task '{task_name}': Up to date")
        return False
    
    def run_task(self, task_name: str, force: bool = False) -> Any:
        """Run a single task"""
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not found")
        
        task = self.tasks[task_name]
        
        # Run dependencies first
        for dep in task['depends_on']:
            self.run_task(dep, force)
        
        # Check if task needs to run
        if not force and not self.should_run_task(task_name):
            return None
        
        print(f"\n🚀 Running task: {task_name}")
        start_time = time.time()
        
        try:
            # Run the task function
            result = task['func']()
            
            # Update cache key
            task['cache_key'] = self.get_task_cache_key(task_name)
            task['last_run'] = datetime.now().isoformat()
            
            elapsed = time.time() - start_time
            print(f"✅ Task '{task_name}' completed in {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Task '{task_name}' failed after {elapsed:.2f}s: {e}")
            raise
    
    def run(self, target_task: str, force: bool = False):
        """Run a task and all its dependencies"""
        print(f"Building target: {target_task}")
        return self.run_task(target_task, force)
    
    def list_tasks(self):
        """List all available tasks"""
        print("Available tasks:")
        for name, task in self.tasks.items():
            deps = ', '.join(task['depends_on']) if task['depends_on'] else 'none'
            print(f"  {name}: depends on [{deps}]")
    
    def clean(self):
        """Clean cache and temporary files"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        print("Cache cleaned")

# Load centralized configuration
CONFIG = get_config().get_script_config('make')

# Ensure CONFIG has required paths for make pipeline
if 'paths' not in CONFIG:
    CONFIG['paths'] = {
        'data_dir': CONFIG['artifacts']['data_dir'],
        'models_dir': CONFIG['artifacts']['models_dir'],
        'train_data': f"{CONFIG['artifacts']['data_dir']}/train_data.parquet",
        'val_data': f"{CONFIG['artifacts']['data_dir']}/val_data.parquet",
        'features_train': f"{CONFIG['artifacts']['data_dir']}/features_train.pkl",
        'features_val': f"{CONFIG['artifacts']['data_dir']}/features_val.pkl",
        'targets_train': f"{CONFIG['artifacts']['data_dir']}/targets_train.pkl",
        'targets_val': f"{CONFIG['artifacts']['data_dir']}/targets_val.pkl",
        'vectorizer': f"{CONFIG['artifacts']['data_dir']}/vectorizer.pkl",
        'model_metadata': f"{CONFIG['artifacts']['models_dir']}/latest_metadata.json",
        'run_id_file': '/home/ubuntu/mlops-dlp/w3_Orchestration/run_id.txt'
    }

if 'cache_dir' not in CONFIG:
    CONFIG['cache_dir'] = '/home/ubuntu/mlops-dlp/.cache'

# Create task runner (basic config, will be updated in main())
runner = TaskRunner(CONFIG)

@runner.task('extract', 
             outputs=[CONFIG['paths']['train_data'], CONFIG['paths']['val_data']])
def extract_data():
    """Extract training and validation data (matching reference script)"""
    year = CONFIG['data']['year']
    month = CONFIG['data']['month']
    
    # Extract training data
    url_train = CONFIG['data']['url_template'].format(year=year, month=month)
    print(f"Downloading training data from {url_train}")
    df_train = pd.read_parquet(url_train)
    
    # Extract validation data (next month)
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    url_val = CONFIG['data']['url_template'].format(year=next_year, month=next_month)
    print(f"Downloading validation data from {url_val}")
    df_val = pd.read_parquet(url_val)
    
    # Ensure data directory exists
    Path(CONFIG['paths']['data_dir']).mkdir(parents=True, exist_ok=True)
    
    # Save raw data
    df_train.to_parquet(CONFIG['paths']['train_data'])
    df_val.to_parquet(CONFIG['paths']['val_data'])
    
    print(f"Extracted {len(df_train)} training and {len(df_val)} validation records")
    return {'train_size': len(df_train), 'val_size': len(df_val)}

def read_dataframe(filepath: str) -> pd.DataFrame:
    """
    Data transformation (matching reference script)
    """
    df = pd.read_parquet(filepath)
    
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

@runner.task('features',
             depends_on=['extract'],
             inputs=[CONFIG['paths']['train_data'], CONFIG['paths']['val_data']],
             outputs=[CONFIG['paths']['features_train'], 
                     CONFIG['paths']['features_val'],
                     CONFIG['paths']['targets_train'], 
                     CONFIG['paths']['targets_val'],
                     CONFIG['paths']['vectorizer']])
def prepare_features():
    """Prepare features for training (matching reference script)"""
    print(f"Preparing features from training and validation data")
    
    # Load and transform data
    df_train = read_dataframe(CONFIG['paths']['train_data'])
    df_val = read_dataframe(CONFIG['paths']['val_data'])
    
    # Prepare features (matching reference script)
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)
    
    # Get target values
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    
    # Save features, targets, and vectorizer
    with open(CONFIG['paths']['features_train'], 'wb') as f:
        pickle.dump(X_train, f)
    
    with open(CONFIG['paths']['features_val'], 'wb') as f:
        pickle.dump(X_val, f)
    
    with open(CONFIG['paths']['targets_train'], 'wb') as f:
        pickle.dump(y_train, f)
        
    with open(CONFIG['paths']['targets_val'], 'wb') as f:
        pickle.dump(y_val, f)
        
    with open(CONFIG['paths']['vectorizer'], 'wb') as f:
        pickle.dump(dv, f)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")
    
    return {
        'train_shape': X_train.shape,
        'val_shape': X_val.shape,
        'n_train_samples': len(y_train),
        'n_val_samples': len(y_val)
    }

@runner.task('train',
             depends_on=['features'],
             inputs=[CONFIG['paths']['features_train'], 
                    CONFIG['paths']['features_val'],
                    CONFIG['paths']['targets_train'],
                    CONFIG['paths']['targets_val'],
                    CONFIG['paths']['vectorizer']],
             outputs=[CONFIG['paths']['model_metadata'], CONFIG['paths']['run_id_file']])
def train_model():
    """Train the ML model (matching reference script)"""
    print("Training model")
    
    # Load features and targets
    with open(CONFIG['paths']['features_train'], 'rb') as f:
        X_train = pickle.load(f)
    
    with open(CONFIG['paths']['features_val'], 'rb') as f:
        X_val = pickle.load(f)
    
    with open(CONFIG['paths']['targets_train'], 'rb') as f:
        y_train = pickle.load(f)
        
    with open(CONFIG['paths']['targets_val'], 'rb') as f:
        y_val = pickle.load(f)
        
    with open(CONFIG['paths']['vectorizer'], 'rb') as f:
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
        models_dir = Path(CONFIG['paths']['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        preprocessor_path = models_dir / "preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")

        # Log model
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        
        run_id = run.info.run_id
        
        print(f"Model trained successfully. RMSE: {rmse:.4f}")
        print(f"Run ID: {run_id}")
        
        # Save metadata
        metadata = {
            'run_id': run_id,
            'rmse': rmse,
            'timestamp': datetime.now().isoformat(),
            'model_params': best_params,
            'training_samples': len(y_train),
            'validation_samples': len(y_val)
        }
        
        with open(CONFIG['paths']['model_metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save run ID to file (matching reference script)
        with open(CONFIG['paths']['run_id_file'], "w") as f:
            f.write(run_id)
        
        return metadata

@runner.task('validate',
             depends_on=['train'],
             inputs=[CONFIG['paths']['model_metadata']])
def validate_model():
    """Validate the trained model"""
    print("Validating model")
    
    # Load metadata
    with open(CONFIG['paths']['model_metadata'], 'r') as f:
        metadata = json.load(f)
    
    rmse = metadata['rmse']
    
    # Quality checks (more lenient for taxi duration prediction)
    max_rmse = 8.0  # Maximum acceptable RMSE for taxi duration
    
    if rmse > max_rmse:
        raise ValueError(f"Model validation failed: RMSE {rmse:.4f} > {max_rmse}")
    
    print(f"Model validation passed: RMSE {rmse:.4f} <= {max_rmse}")
    return True

@runner.task('deploy',
             depends_on=['validate'],
             inputs=[CONFIG['paths']['model_metadata']])
def deploy_model():
    """Deploy the validated model"""
    print("Deploying model (simulation)")
    
    # Load metadata
    with open(CONFIG['paths']['model_metadata'], 'r') as f:
        metadata = json.load(f)
    
    print(f"Deploying model {metadata['run_id']} with RMSE {metadata['rmse']:.4f}")
    
    # Create deployment record
    deployment_record = {
        'deployed_at': datetime.now().isoformat(),
        'model_run_id': metadata['run_id'],
        'model_rmse': metadata['rmse'],
        'deployment_status': 'success',
        'training_samples': metadata.get('training_samples', 0),
        'validation_samples': metadata.get('validation_samples', 0)
    }
    
    deployment_path = Path(CONFIG['paths']['models_dir']) / 'deployment_record.json'
    with open(deployment_path, 'w') as f:
        json.dump(deployment_record, f, indent=2)
    
    print(f"Model deployed successfully. Record saved to {deployment_path}")
    return deployment_record

@runner.task('cleanup',
             depends_on=['deploy'])
def cleanup_temp_files():
    """Clean up temporary files"""
    print("Cleaning up temporary files")
    
    temp_files = [
        CONFIG['paths']['train_data'],
        CONFIG['paths']['val_data'],
        CONFIG['paths']['features_train'],
        CONFIG['paths']['features_val'],
        CONFIG['paths']['targets_train'],
        CONFIG['paths']['targets_val'],
        CONFIG['paths']['vectorizer']
    ]
    
    cleaned_count = 0
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Removed {temp_file}")
            cleaned_count += 1
    
    print(f"Cleanup completed: {cleaned_count} files removed")
    return cleaned_count

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description='ML Pipeline Task Runner (Make-style)')
    parser.add_argument('target', nargs='?', default='deploy', 
                       help='Target task to run (default: deploy)')
    parser.add_argument('--list', action='store_true', help='List all tasks')
    parser.add_argument('--clean', action='store_true', help='Clean cache')
    parser.add_argument('--force', action='store_true', help='Force rebuild all tasks')
    parser.add_argument('--year', type=int, help='Data year (overrides config)')
    parser.add_argument('--month', type=int, help='Data month (overrides config)')
    parser.add_argument('--tracking-server-host', type=str,
                       help='MLflow tracking server host (overrides config)')
    parser.add_argument('--aws-profile', type=str,
                       help='AWS profile for authentication (overrides config)')
    
    args = parser.parse_args()
    
    # Get configuration manager and update if command line arguments provided
    config_manager = get_config()
    
    if args.tracking_server_host or args.aws_profile:
        config_manager.update_mlflow_settings(
            tracking_server_host=args.tracking_server_host,
            aws_profile=args.aws_profile
        )
    
    if args.year or args.month:
        config_manager.update_data_settings(year=args.year, month=args.month)
    
    # Reload configuration after updates
    global CONFIG
    CONFIG = config_manager.get_script_config('make')
    
    # Ensure CONFIG has required paths for make pipeline
    if 'paths' not in CONFIG:
        CONFIG['paths'] = {
            'data_dir': CONFIG['artifacts']['data_dir'],
            'models_dir': CONFIG['artifacts']['models_dir'],
            'train_data': f"{CONFIG['artifacts']['data_dir']}/train_data.parquet",
            'val_data': f"{CONFIG['artifacts']['data_dir']}/val_data.parquet",
            'features_train': f"{CONFIG['artifacts']['data_dir']}/features_train.pkl",
            'features_val': f"{CONFIG['artifacts']['data_dir']}/features_val.pkl",
            'targets_train': f"{CONFIG['artifacts']['data_dir']}/targets_train.pkl",
            'targets_val': f"{CONFIG['artifacts']['data_dir']}/targets_val.pkl",
            'vectorizer': f"{CONFIG['artifacts']['data_dir']}/vectorizer.pkl",
            'model_metadata': f"{CONFIG['artifacts']['models_dir']}/latest_metadata.json",
            'run_id_file': '/home/ubuntu/mlops-dlp/w3_Orchestration/run_id.txt'
        }

    if 'cache_dir' not in CONFIG:
        CONFIG['cache_dir'] = '/home/ubuntu/mlops-dlp/.cache'
    
    # Update existing runner configuration
    runner.config = CONFIG
    runner.setup_mlflow()
    
    if args.list:
        runner.list_tasks()
        return
    
    if args.clean:
        runner.clean()
        return
    
    try:
        # Run the target task
        print(f"🎯 Target: {args.target}")
        print(f"📅 Data: {CONFIG['data']['year']}-{CONFIG['data']['month']:02d}")
        print(f"🔄 Force rebuild: {args.force}")
        print(f"🌐 MLflow Server: {CONFIG['mlflow']['tracking_server_host']}")
        print(f"☁️ AWS Profile: {CONFIG['mlflow']['aws_profile']}")
        print("-" * 50)
        
        result = runner.run(args.target, force=args.force)
        
        print("\n" + "=" * 50)
        print("🎉 Pipeline completed successfully!")
        
        if args.target in ['deploy', 'train'] and os.path.exists(CONFIG['paths']['model_metadata']):
            with open(CONFIG['paths']['model_metadata'], 'r') as f:
                metadata = json.load(f)
            print(f"📊 Final model RMSE: {metadata['rmse']:.4f}")
            print(f"🆔 Model run ID: {metadata['run_id']}")
            
        if os.path.exists(CONFIG['paths']['run_id_file']):
            with open(CONFIG['paths']['run_id_file'], 'r') as f:
                run_id = f.read().strip()
            print(f"📄 Run ID saved to: {CONFIG['paths']['run_id_file']}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
