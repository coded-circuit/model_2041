# Use the official NVIDIA CUDA image as base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set the working directory to /app
WORKDIR /app

# Copy the entire project to the container
COPY . /app

# Install system dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs\
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Set environment variables for pip
ENV PIP_DEFAULT_TIMEOUT=3600 
ENV PIP_NO_CACHE_DIR=1

# Install Python dependencies from requirements_api.txt (located in /testing)
RUN pip3 install -r /app/testing/requirements_api.txt

# Expose the Flask port
EXPOSE 7860

# Set the PYTHONPATH to the app directory
ENV PYTHONPATH="/app"

# Make sure api_server.py is executable
RUN chmod +x /app/testing/api_server.py

# Start the Flask API server using the correct path for api_server.py
CMD ["python3", "/app/testing/api_server.py"]
