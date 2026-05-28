# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for the app
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest and install Python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source
COPY . /app

# Expose port for Cloud Run
EXPOSE 8080

# Use the PORT environment variable Cloud Run provides
ENV PORT 8080

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
