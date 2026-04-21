FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libomp-dev \
    libpq-dev \
    uuid-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY src/vm-services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Compile shared C library from MPI sources (matrix.c present there; OpenMP parallel)
# Excludes main.c and mpi_similarity.c — only sequential/OpenMP functions needed
COPY src/host-cuda/mpi/src/src/ /tmp/c_src/
COPY src/host-cuda/mpi/src/include/ /tmp/c_include/
RUN gcc -shared -fPIC -fopenmp -O3 -std=c11 \
        -I/tmp/c_include \
        /tmp/c_src/data.c \
        /tmp/c_src/file_reader.c \
        /tmp/c_src/parser.c \
        /tmp/c_src/matrix.c \
        /tmp/c_src/recommendations.c \
        -o /usr/local/lib/librec_engine.so \
        -luuid -lm -fopenmp && \
    ldconfig && \
    rm -rf /tmp/c_src /tmp/c_include

# Copy application code (volume-mounted in dev; this is for prod image)
COPY src/vm-services/api/ .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
