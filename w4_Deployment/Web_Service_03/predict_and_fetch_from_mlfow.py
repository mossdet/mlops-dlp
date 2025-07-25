"""
- To run this script:

Setup MLflow tracking server:
    - Start EC2 instance hosting the MLflow trackings erver
    - Start MLflow tracking server with:
        mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/DB_NAME --default-artifact-root s3://S3_BUCKET_NAME
    - Update the mlflow_config with the correct:
        - tracking_server_host (Public DNS from EC2 instance)
        -  Run id from which the model and preprocessor will be fetched

Run Web Service with gunicorn:
    gunicorn --bind=0.0.0.0:9696 predict_and_fetch_from_mlfow:app

Run Docker container:
    CHeck if processes are running on port 9696 of the host machine:
        lsof -i :9696
    If processes are running, stop them before running the Docker container:
        kill PID PID 

    docker build --no-cache -t ride-duration-prediction-service:v1 .
    docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
"""
# Standard library
import os
import pickle
import argparse
from pathlib import Path


# Third-party libraries
import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.feature_extraction import DictVectorizer
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Get script path and set models directory relative to it
script_path = Path(__file__).parent
models_dir = script_path / 'models'

mlflow_config = {
    "tracking_server_host": "ec2-18-219-71-162.us-east-2.compute.amazonaws.com",
    "aws_profile": "mlops_zc",
    "experiment_name": "nyc-taxi-experiment"
}

def load_preprocessor_and_model_from_mlflow(mlflow_config):

    """Setup MLflow tracking"""
    os.environ["AWS_PROFILE"] = mlflow_config['aws_profile']
    tracking_server_host = mlflow_config['tracking_server_host']
    tracking_uri = f"http://{tracking_server_host}:5000"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow_client = MlflowClient(tracking_uri=tracking_uri)

    # retrive model and dict vectorizer from a specific run
    run_id = "590a65419dde4151b5a36de464ed7eef"  # Replace with your actual run ID

    # Load the model from the specified run
    logged_model = f"runs:/{run_id}/models_mlflow"
    try:
        #model = mlflow.pyfunc.load_model(model_uri=logged_model)
        model = mlflow.xgboost.load_model(model_uri=logged_model)
        print(f"Model loaded from run: {run_id}, path: {logged_model}")
    except Exception as e:
        print(f"Error retrieving run: {e}")
        return

    # Retrieve the dict vectorizer artifact from a specific run
    path = "preprocessor"  # Path to the artifact in MLflow
    dst_path = models_dir  # Destination path to save the artifact
    print(f"Retrieving artifact from run: {run_id}")
    try:
        mlflow_client.download_artifacts(run_id=run_id, path=path, dst_path=dst_path)
        print(f"Artifact downloaded to: {dst_path}")
        artifact_path = dst_path / 'preprocessor/preprocessor.b'
        with open(artifact_path, "rb") as f_in:
            dv = pickle.load(f_in)
        print(f"Artifact loaded successfully from: {artifact_path}")
    except Exception as e:
        print(f"Error retrieving artifact: {e}")

    return dv,model

def prepare_features(ride):

    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(dv, model, features):
    """
    Predict the taxi ride duration using the loaded model and features.
    Args:
        dv: The DictVectorizer used for feature transformation.
        model: The loaded XGBoost model.
        features: A dictionary containing the features for prediction.
    Returns:
        prediction: The predicted duration of the taxi ride.
    """

    try:
        # Transform features using the DictVectorizer
        X = dv.transform(features)
        X = xgb.DMatrix(X)
        print("Features transformed successfully")
    except Exception as e:
        print(f"Error transforming features: {e}")
        return None

    try:
        # Make prediction using the loaded model
        preds = model.predict(X)
        print("Prediction made successfully")
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

    # Return the predicted duration
    return float(preds[0])

# Load preprocessor and model
dv, model = load_preprocessor_and_model_from_mlflow(mlflow_config)

# Setup web service
app = Flask('taxi_duration_prediction')
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Flask endpoint for predicting taxi ride duration.
    """

    ride = request.get_json()
    features = prepare_features(ride)
    prediction = predict(dv, model, features)
    result = {
        'duration': float(prediction)
    }

    return jsonify(result)

if __name__ == "__main__":
    dv, model = load_preprocessor_and_model_from_mlflow(mlflow_config)
    app.run(debug=True, host='0.0.0.0', port=9696)