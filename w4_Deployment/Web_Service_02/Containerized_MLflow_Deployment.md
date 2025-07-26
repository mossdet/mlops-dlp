# Guide to deployment of ML models using:
- Flask and Gunicorn for web service
- Fetching the model from MLflow model registry
- Containerization with Docker


Now that we have a model to predict the taxi ride duration and a preprocessor (dictionary vectorizer), we can deploy these two to a service.

1. First, we need to know which xgboost and scikit-learn version we used to train the regression-model and the preprocessor. This is important because the model and preprocessor may not work with a different version of xgboost or scikit-learn. Assuming that we were already working in a virtual environment created with [uv](https://docs.astral.sh/uv/getting-started/installation/), we can check the versions with the following command:

```bash
uv pip freeze | grep -E "scikit*|xgboost"
```
This will output something like:

```bash
scikit-learn==1.7.0
xgboost==3.0.2
```

2. Next, we need to:

    - Create a virtual environment with a specific Python version
    ```bash
        uv venv --python 3.13 deployment_env
    ```

    - Activate the virtual environment
    ```bash
        source deployment_env/bin/activate
    ```
    - Verify that the virtual environment is activated by checking the prompt or running, also check the Python version:
    ```bash
        which python
        python --version
    ```
    - Install the required packages, including scikit-learn and xgboost with the specific version we found in the previous step.
    ```bash
        uv pip install scikit-learn==1.7.0 xgboost==3.0.2 flask numpy pandas
    ```


# Start process with gunicorn
The warning:
```bash
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
```
indicates that the Flask development server is not suitable for production use. Instead, we will use `gunicorn`, a production-ready WSGI server.

```bash
gunicorn --bind=0.0.0.0:9696 predict:app
```

# Test the deployment
- activate the virtual environment if not already activated:
```bash
source ~/deployment_env/bin/activate
```
add request library as a development dependency:
```bash
uv pip install requests
```
- Run the test script to send a request to the deployed service:
```bash
python test_prediction.py 
```


# Kill processes on port 9696
- Find the process ID (PID) of the process running on port 9696:
```bash
lsof -i :9696
```
- Kill the processes using their PID(e.g. 50390 and 51262):
```bash
kill 50390 51262
```

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
The . at the end of the command specifies that the Dockerfile is in the current directory.
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

8. How To Remove Docker Images, Containers, and Volumes

Purging All Unused or Dangling Images, Containers, Volumes, and Networks
Docker provides a single command that will clean up any resources — images, containers, volumes, and networks — that are dangling (not tagged or associated with a container):
```bash
docker system prune
```
To remove all unused images, containers, and volumes, you can use:
```bash
docker system prune -a --volumes
```


## Removing Docker Images
source:https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes
Remove one or more specific images
Use the docker images command with the -a flag to locate the ID of the images you want to remove. This will show you every image, including intermediate image layers. When you’ve located the images you want to delete, you can pass their ID or tag to docker rmi:
```bash
docker images -a
```
Remove a specific image by its ID or tag:
```bash
docker rmi <image-name-or-id>
```
 For example, to remove the ride-duration-prediction-service:v1 image:
```bash
docker rmi ride-duration-prediction-service:v1
```