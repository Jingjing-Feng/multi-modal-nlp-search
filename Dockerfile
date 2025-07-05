# Use Python 3.12 slim as base
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    ffmpeg \
    # Additional dependencies that might be needed for some Python packages
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # Add torch explicitly as it's a dependency
    torch

COPY . .

# Make sure Python can find our modules
ENV PYTHONPATH=/app

# Make sure the app uses environment variables for ES connection
ENV ELASTICSEARCH_HOST=elasticsearch
ENV ELASTICSEARCH_PORT=9200

# Update the CMD to bind to all interfaces
# so that it can be accessed from outside the container
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"] 