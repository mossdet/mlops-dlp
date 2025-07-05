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
        mlflow_db_path = self.config['mlflow']['db_path']
        tracking_uri = f"sqlite:///{mlflow_db_path}"
        
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

# Global configuration
CONFIG = {
    'mlflow': {
        'db_path': '/home/ubuntu/mlops-dlp/mlflow/mlflow.db',
        'experiment_name': 'orchestration-pipeline-make'
    },
    'data': {
        'year': 2023,
        'month': 1,
        'url_template': 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    },
    'model': {
        'params': {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
    },
    'paths': {
        'data_dir': '/home/ubuntu/mlops-dlp/data',
        'models_dir': '/home/ubuntu/mlops-dlp/mlflow/models',
        'raw_data': '/home/ubuntu/mlops-dlp/data/raw_data.parquet',
        'processed_data': '/home/ubuntu/mlops-dlp/data/processed_data.parquet',
        'features': '/home/ubuntu/mlops-dlp/data/features.pkl',
        'targets': '/home/ubuntu/mlops-dlp/data/targets.pkl',
        'vectorizer': '/home/ubuntu/mlops-dlp/data/vectorizer.pkl',
        'model_metadata': '/home/ubuntu/mlops-dlp/mlflow/models/latest_metadata.json'
    },
    'cache_dir': '/home/ubuntu/mlops-dlp/.cache'
}

# Create task runner
runner = TaskRunner(CONFIG)

@runner.task('extract', 
             outputs=[CONFIG['paths']['raw_data']])
def extract_data():
    """Extract raw data from source"""
    year = CONFIG['data']['year']
    month = CONFIG['data']['month']
    url = CONFIG['data']['url_template'].format(year=year, month=month)
    
    print(f"Downloading data from {url}")
    
    df = pd.read_parquet(url)
    
    # Ensure data directory exists
    Path(CONFIG['paths']['data_dir']).mkdir(parents=True, exist_ok=True)
    
    # Save raw data
    df.to_parquet(CONFIG['paths']['raw_data'])
    
    print(f"Extracted {len(df)} records to {CONFIG['paths']['raw_data']}")
    return CONFIG['paths']['raw_data']

@runner.task('transform', 
             depends_on=['extract'],
             inputs=[CONFIG['paths']['raw_data']],
             outputs=[CONFIG['paths']['processed_data']])
def transform_data():
    """Transform raw data"""
    print(f"Transforming data from {CONFIG['paths']['raw_data']}")
    
    # Load raw data
    df = pd.read_parquet(CONFIG['paths']['raw_data'])
    original_size = len(df)
    
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
    df.to_parquet(CONFIG['paths']['processed_data'])
    
    print(f"Transformed data: {original_size} -> {len(df)} records")
    return CONFIG['paths']['processed_data']

@runner.task('features',
             depends_on=['transform'],
             inputs=[CONFIG['paths']['processed_data']],
             outputs=[CONFIG['paths']['features'], 
                     CONFIG['paths']['targets'], 
                     CONFIG['paths']['vectorizer']])
def prepare_features():
    """Prepare features for training"""
    print(f"Preparing features from {CONFIG['paths']['processed_data']}")
    
    # Load processed data
    df = pd.read_parquet(CONFIG['paths']['processed_data'])
    
    categorical = ['PULocationID', 'DOLocationID', 'PU_DO']
    numerical = ['trip_distance']
    
    dv = DictVectorizer()
    
    train_dicts = df[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values
    
    # Save features, targets, and vectorizer
    with open(CONFIG['paths']['features'], 'wb') as f:
        pickle.dump(X_train, f)
    
    with open(CONFIG['paths']['targets'], 'wb') as f:
        pickle.dump(y_train, f)
        
    with open(CONFIG['paths']['vectorizer'], 'wb') as f:
        pickle.dump(dv, f)
    
    print(f"Features prepared. Shape: {X_train.shape}")
    
    return {
        'features_shape': X_train.shape,
        'n_samples': len(y_train)
    }

@runner.task('train',
             depends_on=['features'],
             inputs=[CONFIG['paths']['features'], 
                    CONFIG['paths']['targets'],
                    CONFIG['paths']['vectorizer']],
             outputs=[CONFIG['paths']['model_metadata']])
def train_model():
    """Train the ML model"""
    print("Training model")
    
    # Load features and targets
    with open(CONFIG['paths']['features'], 'rb') as f:
        X_train = pickle.load(f)
    
    with open(CONFIG['paths']['targets'], 'rb') as f:
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
        
        print(f"Model trained. RMSE: {rmse:.4f}, Run ID: {run_id}")
        
        # Save metadata
        metadata = {
            'run_id': run_id,
            'rmse': rmse,
            'timestamp': datetime.now().isoformat(),
            'model_params': params,
            'training_samples': len(y_train)
        }
        
        # Ensure models directory exists
        Path(CONFIG['paths']['models_dir']).mkdir(parents=True, exist_ok=True)
        
        with open(CONFIG['paths']['model_metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Copy vectorizer to models directory
        import shutil
        vectorizer_dest = Path(CONFIG['paths']['models_dir']) / f"vectorizer_{run_id}.pkl"
        shutil.copy2(CONFIG['paths']['vectorizer'], vectorizer_dest)
        
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
    
    # Quality checks
    max_rmse = 10.0  # Maximum acceptable RMSE
    
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
        'deployment_status': 'success'
    }
    
    deployment_path = Path(CONFIG['paths']['models_dir']) / 'deployment_record.json'
    with open(deployment_path, 'w') as f:
        json.dump(deployment_record, f, indent=2)
    
    print(f"Model deployed successfully. Record saved to {deployment_path}")
    return deployment_record

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description='ML Pipeline Task Runner')
    parser.add_argument('target', nargs='?', default='deploy', 
                       help='Target task to run (default: deploy)')
    parser.add_argument('--list', action='store_true', help='List all tasks')
    parser.add_argument('--clean', action='store_true', help='Clean cache')
    parser.add_argument('--force', action='store_true', help='Force rebuild all tasks')
    parser.add_argument('--year', type=int, default=2023, help='Data year')
    parser.add_argument('--month', type=int, default=1, help='Data month')
    
    args = parser.parse_args()
    
    # Update configuration with CLI arguments
    CONFIG['data']['year'] = args.year
    CONFIG['data']['month'] = args.month
    
    if args.list:
        runner.list_tasks()
        return
    
    if args.clean:
        runner.clean()
        return
    
    try:
        # Run the target task
        print(f"🎯 Target: {args.target}")
        print(f"📅 Data: {args.year}-{args.month:02d}")
        print(f"🔄 Force rebuild: {args.force}")
        print("-" * 50)
        
        result = runner.run(args.target, force=args.force)
        
        print("\n" + "=" * 50)
        print("🎉 Pipeline completed successfully!")
        
        if args.target == 'deploy' and os.path.exists(CONFIG['paths']['model_metadata']):
            with open(CONFIG['paths']['model_metadata'], 'r') as f:
                metadata = json.load(f)
            print(f"📊 Final model RMSE: {metadata['rmse']:.4f}")
            print(f"🆔 Model run ID: {metadata['run_id']}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
