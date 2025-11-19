# Multi-stage build for production

# Stage 1: Base
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY app/ ./app/
COPY configs/ ./configs/

# Create directories
RUN mkdir -p logs data models

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run uvicorn server
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: GPU support (optional)
FROM application as gpu

# Install GPU dependencies
RUN pip install --no-cache-dir \
    torch==2.1.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html
