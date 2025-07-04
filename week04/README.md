# Deployment

```mermaid
graph TD
    classDef sub opacity:0
    classDef note fill:#ffd, stroke:#ccb

    A[Deployment]-->B[Batch Offline];
    subgraph subB [" "]
        B
        noteB[Discontinous]
    end

    A-->C[Online];
    subgraph subC [" "]
        C
        noteC[Continous]
    end

    class subA,subB,subC sub
    class noteA,noteB,noteC note
```