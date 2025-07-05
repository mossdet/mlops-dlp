#!/bin/bash

# Week 3 - Orchestration Examples Runner
# This script helps you run different orchestration examples

set -e  # Exit on any error

echo "🎭 MLOps Week 3 - Orchestration Examples"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "simple_pipeline.py" ]; then
    echo "❌ Error: Please run this script from the week03 directory"
    exit 1
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
    echo "Command: python $script"
    echo "----------------------------------------"
    
    if python "$script"; then
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
    echo "1) Simple Pipeline (Class-based)"
    echo "2) Airflow Pipeline (Professional DAG)"
    echo "3) Prefect Pipeline (Modern workflow)"
    echo "4) Make-like Pipeline (Dependency-based)"
    echo "5) Run all examples"
    echo "6) Install optional dependencies"
    echo "7) Clean cache and temporary files"
    echo "0) Exit"
    echo ""
    read -p "Enter your choice (0-7): " choice
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
        rm -rf ../data
        echo "✅ Removed temporary data directory"
    fi
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    echo "✅ Removed Python cache files"
    
    echo "🎉 Cleanup completed!"
}

# Run all examples
run_all() {
    echo ""
    echo "🎯 Running all orchestration examples..."
    echo "This will take several minutes..."
    
    local success_count=0
    local total_count=4
    
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
    
    if [ $success_count -eq $total_count ]; then
        echo "🎉 All examples completed successfully!"
    else
        echo "⚠️  Some examples failed. Check the output above for details."
    fi
}

# Main execution
main() {
    # Check Python availability
    if ! command_exists python; then
        echo "❌ Error: Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check required packages
    echo "🔍 Checking required packages..."
    
    local missing_packages=()
    
    if ! python -c "import pandas" 2>/dev/null; then
        missing_packages+=("pandas")
    fi
    
    if ! python -c "import xgboost" 2>/dev/null; then
        missing_packages+=("xgboost")
    fi
    
    if ! python -c "import sklearn" 2>/dev/null; then
        missing_packages+=("scikit-learn")
    fi
    
    if ! python -c "import mlflow" 2>/dev/null; then
        missing_packages+=("mlflow")
    fi
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo "❌ Missing required packages: ${missing_packages[*]}"
        echo "Please install them with: pip install ${missing_packages[*]}"
        exit 1
    fi
    
    echo "✅ All required packages are installed"
    
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
            0)
                echo "👋 Goodbye!"
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
