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


# Main execution
main() {
    
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
