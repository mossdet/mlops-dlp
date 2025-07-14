# Example Dockerfile for a Flask application
FROM python:3.14.0b3-slim-bullseye

# install uv python
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app
# Copy the application code into the container (more specifically, into the /app directory)
# if you want to add/copy local content, it must be located at the same level in the directory tree as the Dockerfile
COPY ["pyproject.toml", "flask_example_01.py", "./"]

RUN uv_project_env_path=$(python -c "import sysconfig; print(sysconfig.get_config_var('prefix'))") \
    && UV_PROJECT_ENVIRONMENT=$uv_project_env_path
RUN uv sync
#RUN uv add psycopg2-binary
# RUN uv sync