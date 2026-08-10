FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (git is required for the agent to commit/push)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure the entrypoint is executable
RUN chmod +x main.py

# Run as a non-root user.
#
# This image exists to run model-authored code and a project's test suite, both
# of which are untrusted input executed inside the container. As root, anything
# escaping the Python process has root in the container, and a bind-mounted
# repository -- the normal way this is used -- is writable as root.
#
# Placed after pip install so dependencies still install system-wide, and after
# chmod so that still runs as root.
#
# The image is given ownership of /app so a run can write logs and backups. A
# mounted repository still belongs to the host user, so pass
# `--user $(id -u):$(id -g)` when your uid is not 1000 -- that also stops the
# agent leaving root-owned files behind on your machine.
RUN useradd --create-home --uid 1000 agent \
    && chown -R agent:agent /app
USER agent

# Entry command
ENTRYPOINT ["python", "main.py"]
