#!/usr/bin/env python
"""
Centralized Configuration for Week 3 Orchestration Examples

This module provides centralized configuration management for all orchestration scripts.
All MLflow tracking server settings, AWS profiles, and other shared configurations
are managed here to ensure consistency across examples.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

class OrchestrationConfig:
    """Centralized configuration manager for orchestration examples"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Optional path to JSON config file
        """
        self.config_file = config_file or self._get_default_config_path()
        self._config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        return str(Path(__file__).parent / "orchestration_config.json")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                print(f"✅ Loaded configuration from {self.config_file}")
                return config
            except Exception as e:
                print(f"⚠️  Error loading config file: {e}")
                print("Using default configuration...")
        
        # Return default configuration
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "mlflow": {
                "tracking_server_host": "ec2-18-223-115-201.us-east-2.compute.amazonaws.com",
                "aws_profile": "mlops_zc",
                "experiment_name": "nyc-taxi-experiment"
            },
            "data": {
                "year": 2021,
                "month": 1,
                "url_template": "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
            },
            "model": {
                "params": {
                    "learning_rate": 0.09585355369315604,
                    "max_depth": 30,
                    "min_child_weight": 1.060597050922164,
                    "objective": "reg:squarederror",
                    "reg_alpha": 0.018060244040060163,
                    "reg_lambda": 0.011658731377413597,
                    "seed": 42
                },
                "num_boost_round": 30,
                "early_stopping_rounds": 50
            },
            "artifacts": {
                "models_dir": "/home/ubuntu/mlops-dlp/week03/mlflow/models",
                "data_dir": "/home/ubuntu/mlops-dlp/data"
            },
            "validation": {
                "max_rmse": 8.0,
                "min_rmse": 0.1
            }
        }
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get full configuration"""
        return self._config.copy()
    
    def get_mlflow_config(self) -> Dict[str, str]:
        """Get MLflow-specific configuration"""
        return self._config["mlflow"].copy()
    
    def get_tracking_uri(self) -> str:
        """Get complete MLflow tracking URI"""
        host = self._config["mlflow"]["tracking_server_host"]
        return f"http://{host}:5000"
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return self._config["model"].copy()
    
    def update_mlflow_settings(self, tracking_server_host: str = None, aws_profile: str = None):
        """
        Update MLflow settings
        
        Args:
            tracking_server_host: New tracking server host
            aws_profile: New AWS profile
        """
        if tracking_server_host:
            self._config["mlflow"]["tracking_server_host"] = tracking_server_host
            print(f"✅ Updated tracking server host: {tracking_server_host}")
        
        if aws_profile:
            self._config["mlflow"]["aws_profile"] = aws_profile
            print(f"✅ Updated AWS profile: {aws_profile}")
    
    def update_data_settings(self, year: int = None, month: int = None):
        """
        Update data settings
        
        Args:
            year: New default year
            month: New default month
        """
        if year:
            self._config["data"]["year"] = year
            print(f"✅ Updated default year: {year}")
        
        if month:
            self._config["data"]["month"] = month
            print(f"✅ Updated default month: {month}")
    
    def setup_mlflow(self):
        """Setup MLflow with current configuration"""
        import mlflow
        
        os.environ["AWS_PROFILE"] = self._config["mlflow"]["aws_profile"]
        tracking_uri = self.get_tracking_uri()
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self._config["mlflow"]["experiment_name"])
        
        print(f"✅ MLflow configured:")
        print(f"   Tracking URI: {tracking_uri}")
        print(f"   AWS Profile: {self._config['mlflow']['aws_profile']}")
        print(f"   Experiment: {self._config['mlflow']['experiment_name']}")
        
        return tracking_uri
    
    def get_script_config(self, script_type: str) -> Dict[str, Any]:
        """
        Get configuration customized for specific script type
        
        Args:
            script_type: Type of script ('simple', 'airflow', 'prefect', 'make')
            
        Returns:
            Configuration dictionary customized for the script type
        """
        config = self.get_config()
        
        # Customize experiment name based on script type
        if script_type in ['simple', 'airflow', 'prefect', 'make']:
            config["mlflow"]["experiment_name"] = f"nyc-taxi-experiment-{script_type}"
        
        return config
    
    def display_config(self):
        """Display current configuration in a readable format"""
        print("\n🔧 Current Orchestration Configuration")
        print("=" * 50)
        
        mlflow_config = self._config["mlflow"]
        print(f"🌐 MLflow:")
        print(f"   Server: {mlflow_config['tracking_server_host']}")
        print(f"   URI: {self.get_tracking_uri()}")
        print(f"   AWS Profile: {mlflow_config['aws_profile']}")
        print(f"   Experiment: {mlflow_config['experiment_name']}")
        
        data_config = self._config["data"]
        print(f"\n📅 Data:")
        print(f"   Default: {data_config['year']}-{data_config['month']:02d}")
        
        model_config = self._config["model"]
        print(f"\n🤖 Model:")
        print(f"   Algorithm: XGBoost (native API)")
        print(f"   Boost rounds: {model_config['num_boost_round']}")
        print(f"   Early stopping: {model_config['early_stopping_rounds']}")
        
        artifacts_config = self._config["artifacts"]
        print(f"\n📁 Artifacts:")
        print(f"   Models: {artifacts_config['models_dir']}")
        print(f"   Data: {artifacts_config['data_dir']}")
        
        print(f"\n💾 Config file: {self.config_file}")
        print("=" * 50)

# Global configuration instance
_global_config = None

def get_config(config_file: Optional[str] = None) -> OrchestrationConfig:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = OrchestrationConfig(config_file)
    return _global_config

def get_mlflow_config() -> Dict[str, str]:
    """Quick access to MLflow configuration"""
    return get_config().get_mlflow_config()

def get_tracking_uri() -> str:
    """Quick access to MLflow tracking URI"""
    return get_config().get_tracking_uri()

def setup_mlflow() -> str:
    """Quick MLflow setup with current configuration"""
    return get_config().setup_mlflow()

def update_mlflow_settings(tracking_server_host: str = None, aws_profile: str = None):
    """Quick update of MLflow settings"""
    get_config().update_mlflow_settings(tracking_server_host, aws_profile)

# Command-line interface for configuration management
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Orchestration Configuration Manager")
    parser.add_argument('--show', action='store_true', help='Show current configuration')
    parser.add_argument('--create-default', action='store_true', help='Create default configuration file')
    parser.add_argument('--update-mlflow-host', type=str, help='Update MLflow tracking server host')
    parser.add_argument('--update-aws-profile', type=str, help='Update AWS profile')
    parser.add_argument('--update-year', type=int, help='Update default year')
    parser.add_argument('--update-month', type=int, help='Update default month')
    parser.add_argument('--save', action='store_true', help='Save configuration to file')
    
    args = parser.parse_args()
    
    config = get_config()
    
    if args.show:
        config.display_config()
    
    if args.create_default:
        config.save_config()
        print("✅ Default configuration file created")
    
    if args.update_mlflow_host:
        config.update_mlflow_settings(tracking_server_host=args.update_mlflow_host)
    
    if args.update_aws_profile:
        config.update_mlflow_settings(aws_profile=args.update_aws_profile)
    
    if args.update_year:
        config.update_data_settings(year=args.update_year)
    
    if args.update_month:
        config.update_data_settings(month=args.update_month)
    
    if args.save:
        config.save_config()
    
    if not any(vars(args).values()):
        # No arguments provided, show help and current config
        parser.print_help()
        print()
        config.display_config()
