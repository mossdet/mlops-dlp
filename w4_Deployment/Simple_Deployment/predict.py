# Standard library
import os
import pickle
import argparse
from pathlib import Path

# Third-party libraries
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from flask import Flask, request, jsonify

# Get script path and set models directory relative to it
script_path = Path(__file__).parent
models_dir = script_path / 'models'

def load_preprocessor_and_model(models_dir):
    """
    Load the preprocessor and model from the specified directories.
    
    Returns:
        dv: The DictVectorizer used for feature transformation.
        booster_loaded: The loaded XGBoost model.
    """
    preprocessor_path = models_dir / 'preprocessor.b'
    model_path = models_dir / 'booster.json'

    # Load preprocessor
    with open(preprocessor_path, "rb") as f_in:
        dv = pickle.load(f_in)

    # Load saved model
    booster_loaded = xgb.Booster()
    booster_loaded.load_model(str(model_path))

    return dv, booster_loaded

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

    # Transform features using the DictVectorizer
    X = dv.transform(features)
    X = xgb.DMatrix(X)

    # Make prediction using the loaded model
    preds = model.predict(X)

    # Return the predicted duration
    return float(preds[0])

# Load preprocessor and model
dv, model = load_preprocessor_and_model(models_dir)

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
    app.run(debug=True, host='0.0.0.0', port=9696)