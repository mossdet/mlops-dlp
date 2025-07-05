# Deployment

```mermaid
graph TD
    Deployment[Deployment]--> |Runs Periodically| Batch[Batch Offline]
    Deployment--> |Runs Continously| Online[Online]
    Online-->WebService[Webservice]
    Online-->Streaming[Streaming]
```

## 1. Batch Processing
- Run the model periodically (hourly, daily, monthly)
- Usually, a ***scoring job*** performs the following steps:
    - Pull data from database
    - Run model on the data
    - Write prediction results to another database
    - Another script pulls from results database and shows dashboards 📊 📈 💰 
- Example use cases:
    - Marketing data:
        >▶️ predict users about to churn on a daily basis<br>
        >▶️ send attractive offers to avoid churn

## 2. Online Prcoessing
### 2.1 Web Service
```mermaid
graph LR
    User[👩User]--> |Request Taxi Service| Backend[Backend]
    Backend--> |Request| RideDurationService["Ride Duration Service\n(Model)"]
    RideDurationService-->WebService[Webservice]
    RideDurationService-->Streaming[Streaming]
```


### 2.2 Streaming


