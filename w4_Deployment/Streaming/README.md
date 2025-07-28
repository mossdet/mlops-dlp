# Deployment of Streaming Data Processing
This example uses AWS Kinesis and AWS Lambda to process streaming data.

- AWS Kinesis is an event stream processing service (similar to Kafka) that can continuously ingest and process large streams of data records in real-time.
# AWS Kinesis Streaming (Mermaid Diagram)
```mermaid
flowchart LR
    A[Data Producers] -->|Send Data| B[Kinesis Stream]
    B --> C[Shards]
    C --> D[Consumers (Lambda, EC2, Analytics)]
    D --> E[Storage (S3, Redshift, etc.)]
```



- AWS Lambda is a serverless compute service that runs code in response to events and automatically manages the compute resources required by that code.

