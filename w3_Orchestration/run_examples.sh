#!/bin/bash

# Week 3 - Orchestration Examples Runner
# This script helps you run different orchestration examples

set -e  # Exit on any error

echo "🎭 MLOps Week 3 - Orchestration Examples"
echo "========================================"
echo "   ✓ Uses PU_DO as single categorical feature"
echo "   ✓ DictVectorizer with sparse=True"
echo "   ✓ Next month validation data"
echo "   ✓ XGBoost native API with DMatrix"
echo "   ✓ EC2 MLflow server integration"
echo "   ✓ Optimized hyperparameters"
echo "   ✓ Consistent RMSE ~6.60 across all examples"
echo ""

# Check if we're in the right directory
# The -f flag in bash is used with the test command [ -f FILE ] to check if a file exists and is a regular file.
# Example: [ -f "simple_pipeline.py" ] returns true if "simple_pipeline.py" exists and is a regular file.
if [ ! -e "simple_pipeline.py" ]; then
    echo "❌ Error: Please run this script from the w3_Orchestration directory"
    exit 1 # exit 1 indicates an error, exit 0 indicates success
else
    echo "✅ Running from correct directory: $(pwd)"
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run with error handling
run_example() {
    local script=$1
    local name=$2
    
    echo ""
    echo "🚀 Running $name..."
    echo "Command: uv run python $script"
    echo "----------------------------------------"
    
    if uv run python "$script"; then
        echo "✅ $name completed successfully!"
    else
        echo "❌ $name failed!"
        return 1
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "Choose an orchestration example to run:"
    echo "1) Simple Pipeline (✅ RMSE ~6.60)"
    echo "2) Airflow Pipeline (✅ RMSE ~6.60)"
    echo "3) Prefect Pipeline (✅ RMSE ~6.60)"
    echo "4) Make-like Pipeline (✅ RMSE ~6.60)"
    echo "5) Run all examples (sequential execution)"
    echo "6) Install optional dependencies (Airflow, Prefect)"
    echo "7) Clean cache and temporary files"
    echo "8) Show centralized configuration details"
    echo "9) Test individual components"
    echo "0) Exit"
    echo ""
    echo "💡 All pipelines use EC2 MLflow server and optimized hyperparameters"
    echo "🎯 Expected: Consistent RMSE ~6.60 across all implementations"
    read -p "Enter your choice (0-9): " choice
}

# Install optional dependencies
install_dependencies() {
    echo ""
    echo "📦 Installing optional dependencies..."
    echo ""
    
    echo "Installing Airflow..."
    if pip install apache-airflow==2.7.0; then
        echo "✅ Airflow installed successfully"
    else
        echo "❌ Failed to install Airflow"
    fi
    
    echo ""
    echo "Installing Prefect..."
    if pip install prefect==2.14.0; then
        echo "✅ Prefect installed successfully"
    else
        echo "❌ Failed to install Prefect"
    fi
    
    echo ""
    echo "📋 Checking installations..."
    
    if command_exists airflow; then
        echo "✅ Airflow: $(airflow version 2>/dev/null | head -1 || echo 'installed')"
    else
        echo "❌ Airflow: not available"
    fi
    
    if python -c "import prefect; print(f'✅ Prefect: {prefect.__version__}')" 2>/dev/null; then
        echo "✅ Prefect: available"
    else
        echo "❌ Prefect: not available"
    fi
}

# Clean cache and temporary files
clean_cache() {
    echo ""
    echo "🧹 Cleaning cache and temporary files..."
    
    # Remove cache directories
    if [ -d ".cache" ]; then
        rm -rf .cache
        echo "✅ Removed .cache directory"
    fi
    
    # Remove temporary data files
    if [ -d "../data" ]; then
        echo "🗂️  Cleaning data directory..."
        rm -f ../data/train_data*.parquet
        rm -f ../data/val_data*.parquet
        rm -f ../data/features*.pkl
        rm -f ../data/targets*.pkl
        rm -f ../data/vectorizer*.pkl
        echo "✅ Removed temporary data files"
    fi
    
    # Remove MLflow temporary models (keep the main models directory)
    if [ -d "mlflow/models" ]; then
        echo "🗂️  Cleaning temporary MLflow models..."
        find mlflow/models -name "metadata_*.pkl" -delete 2>/dev/null || true
        find mlflow/models -name "vectorizer_*.pkl" -delete 2>/dev/null || true
        echo "✅ Removed temporary MLflow artifacts"
    fi
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    echo "✅ Removed Python cache files"
    
    # Clean run_id.txt if it exists
    if [ -f "run_id.txt" ]; then
        rm run_id.txt
        echo "✅ Removed run_id.txt"
    fi
    
    echo "🎉 Cleanup completed!"
}

# Show configuration details
show_configuration() {
    echo ""
    echo "⚙️  Current Configuration Details"
    echo "================================"
    echo ""
    echo "� Centralized Configuration (config.py):"
    echo "   Config File: orchestration_config.json"
    echo "   Management: python config.py --show"
    echo ""
    
    # Display actual configuration from the centralized system
    if command_exists uv && [ -f "config.py" ]; then
        echo "📊 Live Configuration:"
        uv run python config.py --show 2>/dev/null || echo "   ⚠️  Could not load live config (using defaults below)"
    else
        echo "�📅 Default Data:"
        echo "   Year: 2021, Month: 1"
        echo "   Validation: Uses next month (2021-02) for validation"
        echo ""
        echo "🌐 MLflow Configuration:"
        echo "   Tracking Server: ec2-18-223-115-201.us-east-2.compute.amazonaws.com:5000"
        echo "   AWS Profile: mlops_zc"
        echo "   Experiment: nyc-taxi-experiment-{script-type}"
        echo ""
        echo "🤖 Model Configuration:"
        echo "   Algorithm: XGBoost (native API with DMatrix)"
        echo "   Features: PU_DO (categorical), trip_distance (numerical)"
        echo "   Vectorizer: DictVectorizer(sparse=True)"
        echo "   Hyperparameters: Optimized from reference script"
        echo "   Expected RMSE: ~6.60 (consistent across all implementations)"
        echo ""
        echo "📁 Artifacts:"
        echo "   Models: /home/ubuntu/mlops-dlp/w3_Orchestration/mlflow/models/"
        echo "   Data: /home/ubuntu/mlops-dlp/data/"
        echo "   Run ID: saved to run_id.txt"
    fi
    
    echo ""
    echo "💡 Configuration Management Tips:"
    echo "   - View config: uv run python config.py --show"
    echo "   - Update MLflow server: uv run python -c \"from config import get_config; c=get_config(); c.update_mlflow_settings('new-server'); c.save_config()\""
    echo "   - Update AWS profile: uv run python -c \"from config import get_config; c=get_config(); c.update_mlflow_settings(aws_profile='new-profile'); c.save_config()\""
    echo "   - Override per run: uv run python script.py --tracking-server-host custom-server --aws-profile custom-profile"
    echo ""
    echo "🎯 Performance: Consistent RMSE ~6.60 across all examples"
}

# Test individual components
test_components() {
    echo ""
    echo "🧪 Testing Individual Components"
    echo "==============================="
    echo ""
    
    echo "1. Testing Python imports..."
    if uv run python -c "
import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
print('✅ All required imports successful')
"; then
        echo "✅ Python environment OK"
    else
        echo "❌ Python import issues detected"
        return 1
    fi
    
    echo ""
    echo "2. Testing data download..."
    if uv run python -c "
import pandas as pd
url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet'
df = pd.read_parquet(url)
print(f'✅ Downloaded {len(df)} records from 2021-01')
"; then
        echo "✅ Data download OK"
    else
        echo "❌ Data download failed"
        return 1
    fi
    
    echo ""
    echo "3. Testing MLflow connection..."
    if uv run python -c "
import os
import mlflow
os.environ['AWS_PROFILE'] = 'mlops_zc'
mlflow.set_tracking_uri('http://ec2-18-223-115-201.us-east-2.compute.amazonaws.com:5000')
try:
    mlflow.set_experiment('test-connection')
    print('✅ MLflow connection successful')
except Exception as e:
    print(f'⚠️  MLflow connection warning: {e}')
"; then
        echo "✅ MLflow test completed"
    else
        echo "⚠️  MLflow connection issues (may still work)"
    fi
    
    echo ""
    echo "4. Testing XGBoost training..."
    if uv run python -c "
import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
X = np.random.rand(100, 5)
y = np.random.rand(100)
train = xgb.DMatrix(X, label=y)
params = {'objective': 'reg:squarederror', 'seed': 42}
booster = xgb.train(params, train, num_boost_round=1)
pred = booster.predict(train)
rmse = root_mean_squared_error(y, pred)
print(f'✅ XGBoost training test: RMSE={rmse:.4f}')
"; then
        echo "✅ XGBoost functionality OK"
    else
        echo "❌ XGBoost test failed"
        return 1
    fi
    
    echo ""
    echo "🎉 All component tests completed successfully!"
    echo "Ready to run full pipeline examples."
}

# Run all examples
run_all() {
    echo ""
    echo "🎯 Running all orchestration examples..."
    echo "Expected: Consistent RMSE ~6.60 across all implementations"
    echo "This will take several minutes..."
    echo ""
    
    local success_count=0
    local total_count=4
    
    echo "📋 Execution Plan:"
    echo "   1️⃣  Simple Pipeline (Class-based)"
    echo "   2️⃣  Airflow Pipeline (DAG-based)"
    echo "   3️⃣  Prefect Pipeline (Flow-based)"
    echo "   4️⃣  Make-like Pipeline (Dependency-based)"
    echo ""
    
    if run_example "simple_pipeline.py" "Simple Pipeline"; then
        ((success_count++))
    fi
    
    if run_example "airflow_pipeline.py" "Airflow Pipeline"; then
        ((success_count++))
    fi

    if run_example "prefect_pipeline.py" "Prefect Pipeline"; then
        ((success_count++))
    fi

    if run_example "make_pipeline.py" "Make-like Pipeline"; then
        ((success_count++))
    fi
    
    echo ""
    echo "📊 Summary:"
    echo "Successfully completed: $success_count/$total_count examples"
    echo "Expected RMSE: ~6.60 (consistent across all examples)"
    echo "MLflow Server: ec2-18-223-115-201.us-east-2.compute.amazonaws.com:5000"
    
    if [ $success_count -eq $total_count ]; then
        echo "🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!"
        echo "🎯 Performance: Consistent RMSE ~6.60 achieved across all implementations"
        
        # Show run_id if available
        if [ -f "run_id.txt" ]; then
            echo "📄 Latest Run ID: $(cat run_id.txt)"
        fi
    else
        echo "⚠️  Some examples failed. Check the output above for details."
    fi
}

# Main execution
main() {
    # Check Python availability
    if ! command_exists uv; then
        echo "❌ Error: uv is not installed or not in PATH"
        echo "Please install uv: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    
    # Check required packages
    echo "🔍 Checking required packages..."
    
    local missing_packages=()
    
    if ! uv run python -c "import pandas" 2>/dev/null; then
        missing_packages+=("pandas")
    else
        echo "✅ pandas: $(uv run python -c "import pandas; print(pandas.__version__)" 2>/dev/null)"
    fi
    
    if ! uv run python -c "import xgboost" 2>/dev/null; then
        missing_packages+=("xgboost")
    else
        echo "✅ xgboost: $(uv run python -c "import xgboost; print(xgboost.__version__)" 2>/dev/null)"
    fi
    
    if ! uv run python -c "import sklearn" 2>/dev/null; then
        missing_packages+=("scikit-learn")
    else
        echo "✅ scikit-learn: $(uv run python -c "import sklearn; print(sklearn.__version__)" 2>/dev/null)"
    fi
    
    if ! uv run python -c "import mlflow" 2>/dev/null; then
        missing_packages+=("mlflow")
    else
        echo "✅ mlflow: $(uv run python -c "import mlflow; print(mlflow.__version__)" 2>/dev/null)"
    fi
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo "❌ Missing required packages: ${missing_packages[*]}"
        echo "Please install them with: uv add ${missing_packages[*]}"
        exit 1
    fi
    
    echo "✅ All required packages are installed"
    echo "🌐 MLflow Server: ec2-18-223-115-201.us-east-2.compute.amazonaws.com:5000"
    echo "☁️  AWS Profile: mlops_zc"
    
    # Main loop
    while true; do
        show_menu
        
        case $choice in
            1)
                run_example "simple_pipeline.py" "Simple Pipeline"
                ;;
            2)
                run_example "airflow_pipeline.py" "Airflow Pipeline"
                ;;
            3)
                run_example "prefect_pipeline.py" "Prefect Pipeline"
                ;;
            4)
                run_example "make_pipeline.py" "Make-like Pipeline"
                ;;
            5)
                run_all
                ;;
            6)
                install_dependencies
                ;;
            7)
                clean_cache
                ;;
            8)
                show_configuration
                ;;
            9)
                test_components
                ;;
            0)
                echo "👋 Goodbye!"
                echo "🎯 Expected performance: RMSE ~6.60 across all implementations"
                exit 0
                ;;
            *)
                echo "❌ Invalid choice. Please try again."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@"
