# Use the official Python 3.9 image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container at /app
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && \
    pip install --no-cache-dir -r requirements.txt
