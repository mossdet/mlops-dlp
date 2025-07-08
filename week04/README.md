
# MLOps - Week 4: Model Deployment

This week covers the fundamentals of machine learning model deployment, including different deployment patterns, strategies, and tools used in production environments.

## 📚 Table of Contents
- [1. Batch Processing](#1-batch-processing)
- [2. Online Processing](#2-online-processing)
  - [2.1 Web Service](#21-web-service)
      - [2.1.1 What is HTTP?](#211-http)
  - [2.2 Streaming](#22-streaming)
  - [2.3 Differences between Web Service and Streaming](#23-differences-between-web-service-and-streaming)
- [3. Deployment Strategies](#3-deployment-strategies)
- [4. Deployment Tools](#4-deployment-tools)
  - [4.1 Model Management and Experiment Tracking](#41-model-management-and-experiment-tracking)
  - [4.2 General Orchestration Tools](#42-general-orchestration-tools)
  - [4.3 Containerization and Orchestration Tools](#43-containerization-and-orchestration-tools)
  - [4.4 Model Deployment and Serving](#44-model-deployment-and-serving)
  - [4.5 Data Storage and Management](#45-data-storage-and-management)
  - [4.6 Data Processing and Streaming](#46-data-processing-and-streaming)
  - [4.7 Monitoring and Visualization](#47-monitoring-and-visualization)
  - [4.8 Model Monitoring and Management](#48-model-monitoring-and-management)
  - [4.9 Data Versioning and Management](#49-data-versioning-and-management)
- [5. Further Reading](#5-further-reading)

# Deployment Overview

```mermaid
graph TD
    Deployment[Deployment] --> |Runs Periodically| Batch[Batch Offline]
    Deployment --> |Runs Continuously| Online[Online]
    Online --> WebService[Web Service]
    Online --> Streaming[Streaming]
```

## 1. Batch Processing

**Key Characteristics:**
- Run the model periodically (hourly, daily, monthly)
- Usually, a ***scoring job*** performs the following steps:
    - Pull data from database
    - Run model on the data
    - Write prediction results to another database
    - Another script pulls from results database and shows dashboards 📊 📈 💰 

**Example Use Cases:**
- Marketing data:
    > ▶️ Predict users about to churn on a daily basis<br>
    > ▶️ Send attractive offers to avoid churn
- Financial services:
    > ▶️ Daily credit risk assessment<br>
    > ▶️ Monthly fraud detection analysis
- Recommendation systems:
    > ▶️ Weekly product recommendations<br>
    > ▶️ Daily content personalization updates

## 2. Online Processing
### 2.1 Web Service
- A web service is a way to expose a model as an API endpoint
- It allows other applications to interact with the model over HTTP
- The web service can be built using frameworks like Flask or FastAPI
- This service needs to be online always
- In the case of a TaxiApp, it should be available at all times, providing the user with information like Price and Trip Duration
- The connection between the ***Backend Service*** and the ***Ride Duration Service*** is kept alive until the ***Ride Duration Service*** sends a response to the ***Backend Service***
- The web service can handle multiple requests concurrently, allowing for real-time predictions and interactions
```mermaid
graph LR
    classDef sub opacity:0
    classDef note fill:#ffd, stroke:#ccb

    User[👩User]--> |Request Taxi Service| Backend[Backend]
    Backend--> |"{UserID, PickUp, DropOff, ...}"| Model["Model"]
    Model--> |"Trip Duration"| Backend
    Model--> WebService[Webservice]
    Model--> Streaming[Streaming]

    subgraph subBackendService ["Backend Service "]
        Backend
    end

    subgraph subRideDurationService ["Ride Duration <br/> Prediction Service"]
        Model
    end
```
#### 2.1.1 What is HTTP?
The Hypertext Transfer Protocol (HTTP) is designed to enable communications between clients and servers. HTTP works as a request-response protocol between a client and server.
- **Client**: The client is the application that sends a request to the server.
- **Server**: The server is the application that receives the request and sends a response.
- **Request**: The request is the message sent by the client to the server. It contains the information needed to process the request, such as the URL, headers, and body.
- **Response**: The response is the message sent by the server to the client. It contains the information needed to process the response, such as the status code, headers, and body.
#### 2.1.2 HTTP Methods
- **GET**: Retrieve data from the server. It is used to fetch resources without modifying them.
- **POST**: Send data to the server. It is used to create new resources or submit data for processing.
- **PUT**: Update existing data on the server. It is used to replace the entire resource with the new data.
- **PATCH**: Partially update existing data on the server. It is used to modify specific fields of a resource.
- **DELETE**: Remove data from the server. It is used to delete resources.


### 2.2 Streaming
- Streaming is a way to process data in real-time
- It allows us to process data as it arrives, rather than waiting for a batch of data
- In streaming, we have one (or more) producer and several consumers
- The producer pushes events to the consumers and then this producer doesn't expect a response from the consumers.
```mermaid
graph LR
    Producer[🤖Producer]--> |Pushed Events| Consumer1[Consumer1🤖] 
    Producer[🤖Producer]--> |Pushed Events| Consumer2[Consumer2🤖]
    Producer[🤖Producer]--> |Pushed Events| Consumer3[Consumer3🤖]
    Producer[🤖Producer]--> |Pushed Events| Consumer4[Consumer4🤖]
```

#### Streaming Example: TaxiApp
- In the TaxiApp, the ***Backend Service*** pushes events to the ***Dynamic Ride Duration Service***, ***Tip Prediction Service***, and ***Cost Prediction Service***:
    - The ***Dynamic Ride Duration Service*** is a streaming service that predicts the trip duration based on the initial prediction and the real-time data
    - The ***Tip Prediction Service*** predicts the tip amount based on the trip duration and other factors
    - The ***Cost Prediction Service*** predicts the cost of the trip based on the trip duration and other factors
- The ***Backend Service*** is the producer that pushes events to the consumers
- The consumers are the ***Dynamic Ride Duration Service***, ***Tip Prediction Service***, and ***Cost Prediction Service***
- The consumers send the results directly to the ***User***. The consumers could also send the results back to the ***Backend Service***, which then sends the results back to the ***User***

```mermaid
graph LR
    User[👩User]--> |Request Taxi Service| Backend[Backend]
    Backend--> |Trip Duration| User

    subgraph subBackend [" "]
        Backend--> RideDurationService["Initial Duration Prediction"]--> Backend
    end

    Backend--> Events[/Pushed Events/]
    Events --> Consumer1[Consumer1🤖] & Consumer2[Consumer2🤖] & Consumer3[Consumer3🤖]

    subgraph noteConsumer1 ["Dynamic Trip Duration"]
        Consumer1
    end
    subgraph noteConsumer2 ["Tip Prediction"]
        Consumer2
    end
    subgraph noteConsumer3 ["Cost Prediction"]
        Consumer3
    end

    noteConsumer1--> |"Corrected Trip Duration"| User
    noteConsumer2--> |"Tip Amount"| User
    noteConsumer3--> |"Cost $"| User

```

### 2.3 Differences between Web Service and Streaming

| Aspect | Web Service | Streaming |
|--------|-------------|-----------|
| **Communication Pattern** | Request-Response (Synchronous) | Event-driven (Asynchronous) |
| **Connection** | Connection kept alive until response | Fire-and-forget, no response expected |
| **Response Time** | Immediate response required | No immediate response required |
| **Scalability** | Limited by synchronous nature | Highly scalable, parallel processing |
| **Use Cases** | Real-time predictions, user interactions | Event processing, notifications, analytics |
| **Examples** | Price estimation, trip duration | Dynamic updates, monitoring, alerts |

**Key Differences:**
- In a web service, the connection between the ***Backend Service*** and the ***Model Service*** is kept alive until the service sends a response to the ***Backend***
- In streaming, the ***Backend Service*** pushes events to the consumers and doesn't expect an immediate response
- The consumers process the events and can send results back to the ***Backend Service*** or directly to other services
- Streaming consumers can be run in parallel and on different machines, enabling independent scaling



## 3. Deployment Strategies
- **Blue/Green Deployment**: Two identical environments (blue and green) are maintained.
    - One environment is live (blue) while the other (green) is idle.
    - When a new version is ready, it is deployed to the idle environment (green).
    - After testing, traffic is switched to the green environment.
- **Canary Deployment**: A new version is rolled out to a small subset of users.
    - If successful, it is gradually rolled out to more users.
- **Rolling Deployment**: The new version is gradually rolled out to all instances.
- **Shadow Deployment**: The new version runs alongside the old version, receiving real traffic but not affecting the user experience.
    - It allows for testing the new version in a production environment without affecting users.
- **A/B Testing**: Two versions (A and B) are deployed to different user segments.
    - The performance of both versions is compared to determine which one is better.



## 4. Deployment Tools

### 4.1 Model Management and Experiment Tracking
- **MLflow**: Open-source platform for managing ML lifecycle, experiment tracking, and model deployment
- **Weights & Biases**: Comprehensive ML platform for experiment tracking, dataset versioning, and model management
- **Kubeflow Pipelines**: Kubernetes-native platform for building and deploying portable ML workflows

### 4.2 General Orchestration Tools
- **Apache Airflow**: Platform for developing, scheduling, and monitoring workflows
- **Prefect**: Modern workflow orchestration tool with improved UI and developer experience
- **Dagster**: Data orchestrator for machine learning, analytics, and ETL

### 4.3 Containerization and Orchestration Tools
- **Docker**: Platform for developing, shipping, and running applications in containers
- **Kubernetes**: Container orchestration platform for automating deployment, scaling, and management
- **Docker Compose**: Tool for defining and running multi-container Docker applications

### 4.4 Model Deployment and Serving
- **TensorFlow Serving**: Flexible, high-performance serving system for ML models
- **TorchServe**: PyTorch serving framework for deploying models at scale
- **BentoML**: Framework for building, shipping, and scaling AI applications
- **Seldon Core**: Platform for deploying ML models on Kubernetes

### 4.5 Data Storage and Management
- **Amazon S3**: Object storage service with high availability and durability
- **Google Cloud Storage**: Unified object storage for developers and enterprises
- **PostgreSQL**: Advanced open-source relational database
- **MongoDB**: Document database with high performance and availability

### 4.6 Data Processing and Streaming
- **Apache Spark**: Unified analytics engine for large-scale data processing
- **Apache Kafka**: Distributed event streaming platform for high-performance data pipelines
- **Apache Flink**: Framework for stateful computations over data streams
- **Redis**: In-memory data structure store for caching and real-time analytics

### 4.7 Monitoring and Visualization
- **Grafana**: Open-source platform for monitoring and observability
- **Prometheus**: Open-source monitoring system with time series database
- **Kibana**: Data visualization dashboard for Elasticsearch
- **DataDog**: Cloud monitoring service for servers, databases, and services

### 4.8 Model Monitoring and Management
- **Evidently AI**: Open-source tool for ML model monitoring and testing
- **Fiddler**: Enterprise platform for ML model performance management
- **Arize**: ML observability platform for model monitoring and explainability
- **WhyLabs**: AI observability platform for monitoring data and ML models

### 4.9 Data Versioning and Management
- **DVC**: Version control system for machine learning projects
- **LakeFS**: Data version control for data lakes with Git-like operations
- **Delta Lake**: Open-source storage layer that brings reliability to data lakes
- **Pachyderm**: Data science platform with version control for data pipelines



## 5. Further Reading
- [MLOps: End-to-End Machine Learning Workflow](https://ml-ops.org/content/end-to-end-ml-workflow#model-deployment)
- [MLOps: MLOps Principles](https://ml-ops.org/content/mlops-principles)
