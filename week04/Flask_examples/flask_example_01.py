# SImple example using flask
from flask import Flask, jsonify, request
import time
import random


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Simulate a prediction delay
    time.sleep(random.uniform(0.5, 2.0))
    # Mock prediction logic
    prediction = {"prediction": "mocked_value", "input": data}
    return jsonify(prediction)

if __name__ == '__main__':
    # Run the Flask app
    # The app will be accessible at http://localhost:5000/predict
    # You can send a POST request to this endpoint with JSON data to get a mocked prediction response
    # For example, using curl:
    # curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"input": "test data"}'
    # This will return a mocked prediction response
    # The app will run on all available IP addresses on port 5000
    # This allows you to access the app from any device on the same network
    # If you want to run it on a specific IP address, you can change '0.0.0.0' to your desired IP address
    # The port can also be changed to any available port
    app.run(host='0.0.0.0', port=5000)
