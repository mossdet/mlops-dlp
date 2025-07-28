# Deployment of Streaming Data Processing
This example uses AWS Kinesis and AWS Lambda to process streaming data.

**AWS Kinesis is an event stream processing service (similar to Kafka) that can continuously ingest and process large streams of data records in real-time.**
```mermaid
flowchart LR
    A[Data Producers] -->|Send Data| B[AWS Kinesis Stream]
    B -->|Process Records| C[AWS Lambda / Consumers]
    C -->|Store/Analyze| D[Data Storage / Analytics]
```

**AWS Lambda is a serverless compute service that runs code in response to events and automatically manages the compute resources required by that code.**
```mermaid
flowchart LR
    A[Event Source] -->|Trigger| B[AWS Lambda Function]
    B -->|Process| C[Output]
```


## 1. AWS Lambda Setup:
[AWS Lambda with Kinesis tutorial](https://docs.aws.amazon.com/lambda/latest/dg/with-kinesis-example.html)
- Create a role with permissions to access Kinesis and Lambda.
- Create a lambda function

