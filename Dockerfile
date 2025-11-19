# Multi-stage build for production

# Stage 1: Base
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies (lightweight by default)
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with better error handling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY app/ ./app/
COPY configs/ ./configs/
COPY main.py .
COPY examples.py .

# Create directories
RUN mkdir -p logs data models

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run uvicorn server
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Full features (includes heavy ML packages)
FROM base as full-dependencies

# Install build tools for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-full.txt .
COPY requirements.txt .

# Install with longer timeout for large packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 -r requirements-full.txt

# Stage 5: Full application
FROM full-dependencies as full-application

# Copy application code
COPY app/ ./app/
COPY configs/ ./configs/
COPY main.py .
COPY examples.py .

# Create directories
RUN mkdir -p logs data models

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run uvicorn server
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
