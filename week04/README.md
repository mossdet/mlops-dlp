# Deployment

```mermaid
graph TD
    A[Deployment]-->B[Batch Offline];
    Note right of B: Run Discontinously
    A-->C[Streaming\nOnline];
    Note right of C: Run Continously

```