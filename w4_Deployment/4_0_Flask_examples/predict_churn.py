#!/usr/bin/env python
"""
Churn Prediction Flask Service

Flask web service for customer churn prediction.

Author: Daniel Lachner-Piza
Email: dalapiz@proton.me
"""

import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = '/home/ubuntu/mlops-dlp/w4_Deployment/Flask_examples/model_C=1.0.bin'
#model_file = '/w4_Deployment/Flask_examples/model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)