# 1. Base Image
FROM python:3.13.4-slim

# 2. Metadata
LABEL author="Daniel Lachner-Piza"
LABEL version="1.0"
LABEL description="Dockerfile for deploying a taxi-ride-prediction-app with Python 3.13.4-slim base image."

# 3. System Setup
RUN apt-get update && apt-get install -y \
build-essential \
&& rm -rf /var/lib/apt/lists/*


RUN pip install -U pip

# 4. Application Setup
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# 5. Application Code
# The local folders structure is:
# ├──models/
#       ├──booster.json
#       ├──preprocessor.b
# └── predict.py
# The local folders structure is maintained in the container.:
COPY ["predict_and_fetch_from_mlfow.py", "./"]
#COPY ["models/", "./models/"]


# 6. Runtime Configuration
EXPOSE 9696
ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:9696", "predict_and_fetch_from_mlfow:app" ]