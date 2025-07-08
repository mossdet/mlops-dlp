AIRFLOW_VERSION=2.9.1  # or your desired version
PYTHON_VERSION=3.12    # match your current python version (e.g., 3.9, 3.10)
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

uv pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
