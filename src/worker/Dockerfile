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

COPY src/worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Compile shared C library from MPI sources (same as API image)
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

COPY src/worker/ .

CMD ["python", "worker.py"]
