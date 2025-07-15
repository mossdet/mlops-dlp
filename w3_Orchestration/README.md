[back to main ](../README.md)
# Week 3: ML Pipeline Orchestration Examples

This directory contains comprehensive examples of ML pipeline orchestration approaches, from simple to enterprise-grade solutions. Each example demonstrates different patterns and tools used in the MLOps industry for managing complex machine learning workflows.

## 🔗 Reference Alignment Features

## 📚 Table of Contents
- [Files Overview](#-files-overview)
- [Quick Start](#-quick-start)
- [Feature Comparison](#-feature-comparison)
- [Configuration](#-configuration)
- [Architecture Patterns](#️-architecture-patterns)
- [Expected Output](#-expected-output)
- [Additional Resources](#-additional-resources)
- [Tips](#tips)


## 📁 Files Overview

### 1. `simple_pipeline.py` - Basic Pipeline Class
A straightforward Python class-based approach to orchestrating ML workflows, **FULLY ALIGNED** with the reference `duration-prediction.py` script.

**Features:**
- Object-oriented pipeline design with **complete reference script alignment**
- Step-by-step execution with comprehensive logging
- Next month validation data (proper train/val split)
- XGBoost native API with DMatrix (matching reference implementation)
- MLflow tracking with EC2 server integration
- Sparse feature vectorization for memory efficiency
- Command-line argument support for production use
- Run ID saved to file for downstream processing
- **Expected RMSE: ~6.60 (identical to reference)**

**Expected Runtime:** ~45-60 seconds

**Usage:**
```bash
# Run in testing mode (default)
python simple_pipeline.py

# Run with command line arguments (set testing=False in script)
python simple_pipeline.py --year=2021 --month=1
```

### 2. `airflow_pipeline.py` - Apache Airflow DAG
Professional-grade workflow orchestration using Apache Airflow, **FULLY ALIGNED** with the reference script.

**Features:**
- DAG-based task definition with **complete reference alignment**
- Task dependencies and XCom communication
- Retry logic and error handling
- Scalable and production-ready
- **Expected RMSE: ~6.60 (identical to reference)**

**Expected Runtime:** ~50-70 seconds (standalone mode)

**Usage:**
```bash
# Install Airflow (optional for this example)
pip install apache-airflow

# Run standalone (without Airflow server)
python airflow_pipeline.py

# Or deploy to Airflow (requires Airflow setup)
# Copy to $AIRFLOW_HOME/dags/ directory
```
**Workaround to install Airflow:**
```bash
AIRFLOW_VERSION=2.9.1  # or your desired version
PYTHON_VERSION=3.12    # match your current python version (e.g., 3.9, 3.10)
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
uv pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Verify installation
airflow version
```

### 3. `prefect_pipeline.py` - Prefect Workflow
Modern workflow orchestration with Prefect 2.0, **FULLY ALIGNED** with the reference `duration-prediction.py` script.

**Features:**
- Flow and task decorators (no deprecated SequentialTaskRunner)
- Advanced retry strategies with configurable delays
- Data validation steps with quality checks
- Next month validation data (train/val split like reference script)
- XGBoost native API with DMatrix (matching reference implementation)
- MLflow tracking with EC2 server integration
- Command-line argument support for production use
- Sparse feature vectorization and optimized feature engineering
- **Expected RMSE: ~6.60 (identical to reference)**

**Expected Runtime:** ~55-75 seconds (with validation steps)

**Usage:**
```bash
# Install Prefect (optional for this example)
pip install prefect

# Run with command line arguments (production mode)
python prefect_pipeline.py --year=2021 --month=1 --tracking-server-host=ec2-18-223-115-201.us-east-2.compute.amazonaws.com --aws-profile=mlops_zc

# Run in testing mode (edit script to set testing=True)
python prefect_pipeline.py

# Or deploy to Prefect server
# prefect deployment build-from-flow prefect_pipeline.py:ml_pipeline_flow
```

### 4. `make_pipeline.py` - Make-like Task Runner
Simple dependency-based task runner inspired by GNU Make, **FULLY ALIGNED** with the reference script.

**Features:**
- Dependency tracking with smart rebuilds
- Incremental builds (only run what's needed)
- File-based caching for efficiency
- CLI interface with comprehensive options
- **Complete alignment with reference script**
- **Expected RMSE: ~6.60 (identical to reference)**

**Expected Runtime:** ~30-45 seconds (with caching), ~60 seconds (full rebuild)

**Usage:**
```bash
# Run full pipeline
python make_pipeline.py deploy

# List available tasks
python make_pipeline.py --list

# Force rebuild all tasks
python make_pipeline.py deploy --force

# Clean cache
python make_pipeline.py --clean

# Run with different data
python make_pipeline.py deploy --year 2021 --month 1 --tracking-server-host ec2-18-223-115-201.us-east-2.compute.amazonaws.com
```

### 5. `run_examples.sh` - Interactive Runner
**Enhanced** bash script that provides an interactive menu to run and manage all examples.

**Features:**
- Interactive menu interface with **FULL ALIGNMENT status indicators**
- Dependency checking and installation with version reporting
- Cache cleanup functionality with selective file removal
- Progress tracking and error reporting
- Configuration display showing current MLflow and model settings
- Component testing to verify environment readiness
- **Performance tracking**: Shows expected RMSE ~6.60 for all examples
- Comprehensive status reporting for all examples

**Usage:**
```bash
# Make executable (if needed)
chmod +x run_examples.sh

# Run interactive menu
./run_examples.sh

# Available options:
# 1-4) Run individual examples (ALL FULLY ALIGNED)
# 5) Run all examples sequentially
# 6) Install optional dependencies (Airflow, Prefect)
# 7) Clean cache and temporary files
# 8) Show configuration details
# 9) Test individual components
# 0) Exit
```

### 6. `duration-prediction.py` - Original Implementation
The original ML pipeline implementation used as a baseline for comparison.

**Features:**
- Basic ML workflow
- MLflow integration
- Command-line arguments

### 7. `duration-prediction.ipynb` - Jupyter Notebook
Interactive notebook version for experimentation and learning.


## 🚀 Quick Start

### Option 1: Interactive Menu (Recommended)

**Method 1: Direct execution**
```bash
cd w3_Orchestration
./run_examples.sh
```

**Method 2: Using bash explicitly**
```bash
cd w3_Orchestration
bash run_examples.sh
```

**Method 3: Make executable first (if needed)**
```bash
cd w3_Orchestration
chmod +x run_examples.sh
./run_examples.sh
```

The interactive menu will show:
- Alignment status for each script (✅ Fully Aligned)
- Configuration details (MLflow server, AWS profile, etc.)
- Component testing options
- Enhanced cleanup and management tools

### Option 2: Direct Execution
```bash
# All scripts now use consistent defaults (year=2021, month=1)
uv run python simple_pipeline.py       # ✅ FULLY ALIGNED - RMSE ~6.60
uv run python airflow_pipeline.py      # ✅ FULLY ALIGNED - RMSE ~6.60  
uv run python prefect_pipeline.py      # ✅ FULLY ALIGNED - RMSE ~6.60
uv run python make_pipeline.py         # ✅ FULLY ALIGNED - RMSE ~6.60

# All scripts support command-line arguments:
uv run python simple_pipeline.py --year 2021 --month 1 --tracking-server-host ec2-18-223-115-201.us-east-2.compute.amazonaws.com --aws-profile mlops_zc

# Expected output for ALL scripts:
# Training features shape: (73908, 13221)
# Validation features shape: (61921, 13221)  
# Model trained successfully. RMSE: 6.6077 (consistent across all)
# MLflow run_id: [unique-run-id]
```

### Option 3: Advanced Usage
```bash
# Use make-like approach for dependency management
uv run python make_pipeline.py deploy --year 2021 --month 1

# Force rebuild everything
uv run python make_pipeline.py deploy --force

# List available tasks
uv run python make_pipeline.py --list

# Test individual components
./run_examples.sh  # Choose option 9 for component testing
```

##  📊 Feature Comparison

| Feature | Simple | Airflow | Prefect | Make-like | Interactive |
|---------|--------|---------|---------|-----------|-------------|
| **Complexity** | Low | High | Medium | Low | Very Low |
| **Production Ready** | Limited | Yes | Yes | Limited | No |
| **External Dependencies** | None | Airflow | Prefect | None | None |
| **UI/Monitoring** | No | Yes | Yes | CLI only | Menu |
| **Scalability** | Limited | High | High | Medium | Limited |
| **Learning Curve** | Easy | Steep | Medium | Easy | Easiest |
| **Retry Logic** | Basic | Advanced | Advanced | None | Basic |
| **Task Dependencies** | Linear | DAG | Flow | DAG | Sequential |
| **Parallel Execution** | No | Yes | Yes | No | No |
| **Error Recovery** | Basic | Advanced | Advanced | None | Basic |
| **Reference Alignment** | ✅ **FULL** | ✅ **FULL** | ✅ **FULL** | ✅ **FULL** | ✅ **Enhanced** |
| **MLflow Integration** | ✅ **EC2 Server** | ✅ **EC2 Server** | ✅ **EC2 Server** | ✅ **EC2 Server** | Basic |
| **Feature Engineering** | ✅ **PU_DO Only** | ✅ **PU_DO Only** | ✅ **PU_DO Only** | ✅ **PU_DO Only** | N/A |
| **XGBoost Implementation** | ✅ **Native API** | ✅ **Native API** | ✅ **Native API** | ✅ **Native API** | N/A |
| **Hyperparameters** | ✅ **Optimized** | ✅ **Optimized** | ✅ **Optimized** | ✅ **Optimized** | N/A |
| **Expected RMSE** | ✅ **~6.60** | ✅ **~6.60** | ✅ **~6.60** | ✅ **~6.60** | N/A |
| **Performance Consistency** | ✅ **Identical** | ✅ **Identical** | ✅ **Identical** | ✅ **Identical** | N/A |


## 🔧 Configuration

### 🎯 Centralized Configuration Management
All orchestration scripts use a **centralized configuration system** via `config.py` for consistent settings across all examples.

#### Configuration File: `orchestration_config.json`
```json
{
  "mlflow": {
    "tracking_server_host": "ec2-18-223-115-201.us-east-2.compute.amazonaws.com",
    "aws_profile": "mlops_zc", 
    "experiment_name": "nyc-taxi-experiment"
  },
  "data": {
    "year": 2021,
    "month": 1,
    "url_template": "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
  },
  "model": {
    "params": {
      "learning_rate": 0.09585355369315604,
      "max_depth": 30,
      "min_child_weight": 1.060597050922164,
      "objective": "reg:squarederror",
      "reg_alpha": 0.018060244040060163,
      "reg_lambda": 0.011658731377413597,
      "seed": 42
    },
    "num_boost_round": 30,
    "early_stopping_rounds": 50
  },
  "artifacts": {
    "models_dir": "/home/ubuntu/mlops-dlp/w3_Orchestration/mlflow/models",
    "data_dir": "/home/ubuntu/mlops-dlp/data"
  }
}
```

#### Using the Configuration Manager

**Display current configuration:**
```bash
python config.py --show
```

**Update MLflow settings:**
```bash
# Update tracking server
python -c "from config import get_config; c=get_config(); c.update_mlflow_settings('new-server.com'); c.save_config()"

# Update AWS profile
python -c "from config import get_config; c=get_config(); c.update_mlflow_settings(aws_profile='new-profile'); c.save_config()"
```

**Update data settings:**
```bash
python -c "from config import get_config; c=get_config(); c.update_data_settings(year=2022, month=3); c.save_config()"
```

#### Command Line Overrides

All scripts support command line overrides that temporarily override the configuration:

```bash
# Simple Pipeline
python simple_pipeline.py --year 2021 --month 2 --tracking-server-host custom-server.com --aws-profile custom-profile

# Airflow Pipeline  
python airflow_pipeline.py --year 2021 --month 2 --tracking-server-host custom-server.com --aws-profile custom-profile

# Prefect Pipeline
python prefect_pipeline.py --year 2021 --month 2 --tracking-server-host custom-server.com --aws-profile custom-profile

# Make Pipeline
python make_pipeline.py --year 2021 --month 2 --tracking-server-host custom-server.com --aws-profile custom-profile
```

## 🏗️ Architecture Patterns

### 1. Linear Pipeline (Simple)
```
[Data] → [Transform] → [Train] → [Deploy]
```
- **Best for**: Proof of concepts, simple workflows
- **Pros**: Easy to understand and debug
- **Cons**: No parallelism, limited error recovery

### 2. DAG-based Pipeline (Airflow/Make)
```
      ┌─[Transform]─┐
[Data]─┤             ├─[Train]─[Deploy]
      └─[Validate]──┘
```
- **Best for**: Complex workflows with dependencies
- **Pros**: Parallel execution, fine-grained control
- **Cons**: More complex setup and debugging

### 3. Flow-based Pipeline (Prefect)
```
@flow
def ml_pipeline():
    data = extract()
    validated = validate(data)
    features = transform(validated)
    model = train(features)
    deploy(model)
```
- **Best for**: Modern development workflows
- **Pros**: Python-native, great debugging, automatic retries
- **Cons**: Requires Prefect infrastructure

## 📚 Additional Resources

### Documentation
- [Apache Airflow Documentation](https://airflow.apache.org/docs/) - Comprehensive DAG development guide
- [Prefect Documentation](https://docs.prefect.io/) - Modern workflow orchestration
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) - ML lifecycle management
- [GNU Make Manual](https://www.gnu.org/software/make/manual/) - Classic build automation

### Tutorials and Guides
- [MLOps Principles](https://ml-ops.org/) - Industry best practices
- [Pipeline Design Patterns](https://martinfowler.com/articles/data-pipeline-patterns.html) - Architectural guidance
- [Production ML Systems](https://developers.google.com/machine-learning/guides/rules-of-ml) - Google's ML engineering guide

### Community Resources
- [r/MachineLearning](https://reddit.com/r/MachineLearning) - Community discussions
- [MLOps Community](https://mlops.community/) - Slack workspace and events
- [Made With ML](https://madewithml.com/) - Practical MLOps tutorials

---

## 💡 Tips

### For First-Time Users
1. **Start Simple**: Begin with `./run_examples.sh` for a guided experience
2. **Check Prerequisites**: Ensure all required packages are installed before starting
3. **Monitor Resources**: Keep an eye on memory and disk usage during execution
4. **Read the Logs**: Pay attention to the console output for understanding pipeline flow

### For Developers
1. **Compare Approaches**: Run multiple examples to understand trade-offs
2. **Experiment with Parameters**: Try different data periods and model settings
3. **Break Things Intentionally**: Test error handling by introducing failures
4. **Measure Performance**: Time different approaches for your use case

### For Production Use
1. **Choose Based on Scale**: Use Airflow/Prefect for complex, production workflows
2. **Implement Monitoring**: Add comprehensive logging and alerting
3. **Test Thoroughly**: Validate all edge cases and failure scenarios
4. **Document Dependencies**: Maintain clear dependency documentation

## 🔧 Common Commands Reference

```bash
# Quick test of all approaches
./run_examples.sh

# Run specific pipeline
python simple_pipeline.py  # Testing mode (aligned with reference script)
python airflow_pipeline.py  
python prefect_pipeline.py  # Testing mode (edit script to set testing=True)
python make_pipeline.py deploy

# Run Prefect with arguments (production mode)
python prefect_pipeline.py --year=2021 --month=1 --tracking-server-host=ec2-18-223-115-201.us-east-2.compute.amazonaws.com --aws-profile=mlops_zc

# Troubleshooting
python make_pipeline.py --clean  # Clean cache
python make_pipeline.py --list   # Show available tasks
python --version                 # Check Python version
pip list | grep -E "(mlflow|pandas|xgboost|scikit-learn|prefect)"  # Check packages

# Advanced usage
python make_pipeline.py deploy --year 2021 --month 3 --force
python make_pipeline.py validate  # Run only up to validation step

# MLflow server connectivity test
ping ec2-18-223-115-201.us-east-2.compute.amazonaws.com
aws configure list-profiles  # Check AWS profiles
```

---

## 📊 Expected Output

### Successful Pipeline Run
```
🎯 Target: deploy
📅 Data: 2023-01
🔄 Force rebuild: False
--------------------------------------------------

🚀 Running task: extract
✅ Task 'extract' completed in 15.23s

🚀 Running task: transform  
✅ Task 'transform' completed in 3.45s

🚀 Running task: features
✅ Task 'features' completed in 2.18s

🚀 Running task: train
✅ Task 'train' completed in 8.91s

🚀 Running task: validate
✅ Task 'validate' completed in 0.12s

🚀 Running task: deploy
✅ Task 'deploy' completed in 0.34s

==================================================
🎉 Pipeline completed successfully!
📊 Final model RMSE: 6.2847
🆔 Model run ID: a1b2c3d4e5f6g7h8i9j0
```

### Typical Performance Metrics
- **Total Runtime**: 30-60 seconds (depending on data size)
- **Data Size**: ~2M records (January 2021, aligned with reference)
- **Model RMSE**: 5-8 (typical range for trip duration prediction)
- **Memory Usage**: <2GB peak
- **Disk Usage**: ~500MB for artifacts
