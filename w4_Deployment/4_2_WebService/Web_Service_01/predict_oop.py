#!/usr/bin/env python
"""
Taxi Duration Prediction Service

This script provides a Flask web service for predicting taxi ride duration
using XGBoost model and preprocessor.

The class `TaxiDurationPredictor` handles model loading, feature preparation,
and prediction in an object-oriented manner. 

The `TaxiDurationAPI` class wraps
the predictor in a Flask application, providing endpoints for prediction and health checks.

To run the service with gunicorn, use:
gunicorn -w 4 -k uvicorn.workers.UvicornWorker predict_oop
with the specified model directory.


Author: Daniel Lachner-Piza
Email: dalapiz@proton.me
"""

# Standard library
import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

# Third-party libraries
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from flask import Flask, request, jsonify


class TaxiDurationPredictor:
    """
    A class for predicting taxi ride duration using XGBoost model.
    
    This class handles model loading, feature preparation, and prediction
    in a structured object-oriented manner.
    """
    
    def __init__(self, models_dir: str):
        """
        Initialize the predictor with model directory.
        
        Args:
            models_dir (str): Path to directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.dv = None
        self.model = None
        self._load_model_and_preprocessor()
    
    def _load_model_and_preprocessor(self) -> None:
        """
        Load the preprocessor and model from the specified directories.
        
        Raises:
            FileNotFoundError: If model files are not found
            Exception: If model loading fails
        """
        preprocessor_path = self.models_dir / 'preprocessor.b'
        
        # Try different model formats
        model_extensions = ['.json', '.ubj', '.bin', '']
        model_path = None
        
        for ext in model_extensions:
            potential_path = self.models_dir / f'booster{ext}'
            if potential_path.exists():
                model_path = potential_path
                break
        
        if model_path is None:
            model_path = self.models_dir / 'booster'  # fallback
        
        try:
            # Load preprocessor
            with open(preprocessor_path, "rb") as f_in:
                self.dv = pickle.load(f_in)
            print(f"✅ Preprocessor loaded from {preprocessor_path}")
            
            # Load XGBoost model
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            print(f"✅ Model loaded from {model_path}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found: {e}")
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def prepare_features(self, ride: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare features from ride data for prediction.
        
        Args:
            ride (Dict): Dictionary containing ride information
            
        Returns:
            Dict: Prepared features for model input
        """
        features = {}
        features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
        features['trip_distance'] = ride['trip_distance']
        return features
    
    def predict(self, features: Dict[str, Any]) -> float:
        """
        Predict taxi ride duration using the loaded model.
        
        Args:
            features (Dict): Prepared features for prediction
            
        Returns:
            float: Predicted duration in minutes
            
        Raises:
            ValueError: If model is not loaded
        """
        if self.dv is None or self.model is None:
            raise ValueError("Model and preprocessor must be loaded before prediction")
        
        # Transform features using the DictVectorizer
        X = self.dv.transform([features])  # Note: wrap features in list
        X_dmatrix = xgb.DMatrix(X)
        
        # Make prediction using the loaded model
        prediction = self.model.predict(X_dmatrix)
        
        return float(prediction[0])
    
    def predict_from_ride(self, ride: Dict[str, Any]) -> float:
        """
        End-to-end prediction from raw ride data.
        
        Args:
            ride (Dict): Raw ride data
            
        Returns:
            float: Predicted duration in minutes
        """
        features = self.prepare_features(ride)
        return self.predict(features)


class TaxiDurationAPI:
    """
    Flask API wrapper for the TaxiDurationPredictor.
    
    This class provides a web service interface for the prediction model.
    """
    
    def __init__(self, models_dir: str, app_name: str = 'taxi_duration_prediction'):
        """
        Initialize the API with predictor and Flask app.

        To run the API, use:
        gunicorn -w 4 -k uvicorn.workers.UvicornWorker predict_oop:TaxiDurationAPI
        
        Args:
            models_dir (str): Path to model directory
            app_name (str): Name for the Flask application
        """
        self.predictor = TaxiDurationPredictor(models_dir)
        self.app = Flask(app_name)
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Set up Flask routes for the API."""
        # Register the prediction endpoint
        # Parameters explained:
        # 1. '/predict' - URL path (what users access)
        # 2. 'predict' - endpoint name (internal Flask identifier)
        # 3. self.predict_endpoint - method to call when route is accessed
        # 4. methods=['POST'] - only accept POST requests
        self.app.add_url_rule('/predict', 'predict', self.predict_endpoint, methods=['POST'])
        
        # Register health check endpoint (GET request)
        self.app.add_url_rule('/health', 'health', self.health_check, methods=['GET'])
        
        # Alternative decorator approach would be:
        # @self.app.route('/predict', methods=['POST'])
        # def predict_endpoint(self): ...
        # But this doesn't work well with class methods
    
    def predict_endpoint(self):
        """
        Flask endpoint for predicting taxi ride duration.
        
        Returns:
            JSON response with prediction result
        """
        try:
            ride = request.get_json()
            
            # Validate input
            required_fields = ['PULocationID', 'DOLocationID', 'trip_distance']
            for field in required_fields:
                if field not in ride:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            # Make prediction
            prediction = self.predictor.predict_from_ride(ride)
            
            result = {
                'duration': prediction,
                'status': 'success'
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500
    
    def health_check(self):
        """
        Health check endpoint.
        
        Returns:
            JSON response indicating service health
        """
        return jsonify({
            'status': 'healthy',
            'model_loaded': self.predictor.model is not None,
            'preprocessor_loaded': self.predictor.dv is not None
        })
    
    def run(self, debug: bool = True, host: str = '0.0.0.0', port: int = 9696):
        """
        Run the Flask application.
        
        Args:
            debug (bool): Enable debug mode
            host (str): Host address
            port (int): Port number
        """
        print(f"🚀 Starting Taxi Duration Prediction API on {host}:{port}")
        self.app.run(debug=debug, host=host, port=port)


def main():
    """
    Main function to run the application.

    To run with gunicorn, use:
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker predict_oop:main
    with the specified model directory.
    """
    # Configuration
    models_dir = '/home/ubuntu/mlops-dlp/w4_Deployment/Web_Service_01/models/'

    # Create and run API
    api = TaxiDurationAPI(models_dir)
    api.run()


if __name__ == "__main__":
    main()
