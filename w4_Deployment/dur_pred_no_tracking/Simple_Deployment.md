
# Simple Deployment Guide

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
        uv pip install scikit-learn==1.7.0 xgboost==3.0.2
    ```
