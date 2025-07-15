# Web Service Deployment for Ride Duration Prediction, WITHOUT(!) retrieval of models from model registry

# Build Docker Image
1. In the virtual environment used to develop the application, generate the requirements.txt file:
```bash
pip freeze > requirements.txt
```
2. Copy the requirements.txt file and the application code (predict.py and models directory) to the same directory as the Dockerfile.
This directory should contain:
```
├── Dockerfile
├── requirements.txt
├── predict.py
└── models/
    ├── booster.json
    └── preprocessor.b
```


3. Define the Dockerfile and make sure that the paths in the application code match the paths in the Dockerfile.

4. Navigate to the directory containing the Dockerfile and the application code:
```bash
cd path/to/Docker/file/directory
```

5. Build the Docker image using the Dockerfile:
The --no-cache option ensures that the image is built from scratch, without using any cached layers.
```bash
docker build --no-cache -t ride-duration-prediction-service:v1 .
```
6. Verify that the image is built successfully by listing the Docker images:
```bash
docker images
```

7. Run the Docker Container in interactive mode (this allows canceling the process with Ctrl+C):
```bash
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```