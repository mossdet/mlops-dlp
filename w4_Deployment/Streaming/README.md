# Deployment of Streaming Data Processing
This example uses AWS Kinesis and AWS Lambda to process streaming data.

- AWS Kinesis is an event stream processing service (similar to Kafka) that can continuously ingest and process large streams of data records in real-time.

# AWS Kinesis Streaming Architecture (Merlin Diagram)
```mermaid
flowchart LR
    A[Data Producers<br>(Apps, Devices, Logs)] -->|PutRecord| B[AWS Kinesis Stream]
    B --> C[Shards]
    C --> D[Data Consumers<br>(EC2, Lambda, Kinesis Data Analytics)]
    D --> E[Storage<br>(S3, Redshift, DynamoDB)]
    B -.->|Retention| F[Replay/Analytics]
```

- **Data Producers** send records to the **Kinesis Stream**.
- The stream is divided into **Shards** for parallelism and scalability.
- **Data Consumers** (like AWS Lambda, EC2, or Kinesis Data Analytics) process the records in real time.
- Processed data can be stored in **S3**, **Redshift**, or **DynamoDB**.
- Kinesis supports **retention** and replay for analytics or recovery.

# Merlin Diagramm
![Merlin Diagram](https://raw.githubusercontent.com/merlin-ml/merlin/main/examples/w4_Deployment/Streaming/merlin_diagram.png)
- AWS Lambda is a serverless compute service that runs code in response to events and automatically manages the compute resources required by that code.

