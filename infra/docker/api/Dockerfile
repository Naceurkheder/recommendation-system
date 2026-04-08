FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for C compilation and Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libomp-dev \
    libpq-dev \
    uuid-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build C recommendation engine (if source available)
RUN if [ -d "/app/../../../src/host-cuda/openmp/src" ]; then \
    cd /app/../../../src/host-cuda/openmp/src && \
    make openmp && \
    mkdir -p /app/lib && \
    cp bin/similarity_openmp /app/lib/ 2>/dev/null || true; \
    fi || echo "C code not available, skipping build"

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
