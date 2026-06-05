# SPADE Forensics API - Docker Image
# Multi-stage build for optimized image size

# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package with API dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[api]"

# Runtime stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd -m -u 1000 spade && \
    mkdir -p /data /app && \
    chown -R spade:spade /data /app

# Set working directory
WORKDIR /app
USER spade

# Copy application code
COPY --chown=spade:spade src/ ./src/

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SPADE_MAX_TARGETS=1000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"

# Expose API port
EXPOSE 8000

# Volume for persistent data
VOLUME ["/data"]

# Run API server
CMD ["python", "-m", "spade", "serve", "--host", "0.0.0.0", "--port", "8000"]
