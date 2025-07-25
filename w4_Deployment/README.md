
[back to main ](../README.md)
# MLOps - Week 4: Model Deployment

This section covers the fundamentals of machine learning model deployment, including different deployment patterns, strategies, and tools used in production environments.

## Web Service Deployment for Ride Duration Prediction
This section describes the deployment of a web service for ride duration prediction, which includes the retrieval of models from a model registry.

### Implementation Details
- **Model Retrieval**: The web service retrieves the latest model version from the MLflow model registry.
- **Containerization**: The service is containerized using Docker for easy deployment and scalability.
- **API Endpoints**: RESTful API endpoints are created for model inference.
- **Environment Setup**: The service is set up to run in a Python environment with necessary dependencies.

![Experiment-Tracking-Visual-Summary](/Visual_Summaries/Deployment_1.png)