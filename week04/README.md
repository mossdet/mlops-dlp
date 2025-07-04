# Deployment

```mermaid
graph TD
    Deployment[Deployment]--> |Runs Periodically| Batch[Batch Offline]
    Deployment--> |Runs Continously| Online[Online]
    Online-->WebService[Webservice]
    Online-->Streaming[Streaming]
```

## 1. Batch Processing


## 2. Online


### 2.1 Web Service


### 2.2 Streaming


