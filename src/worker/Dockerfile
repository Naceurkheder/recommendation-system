FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    libomp-dev \
    uuid-dev \
    libuuid1 \
    libpq-dev \
    curl \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY src/host-cuda/openmp/src/ /build/openmp/
COPY src/host-cuda/cuda/src/   /build/cuda/

RUN gcc -shared -fPIC -O3 -std=c11 -fopenmp \
    -I/build/openmp/include \
    /build/openmp/src/data.c \
    /build/openmp/src/file_reader.c \
    /build/openmp/src/parser.c \
    /build/openmp/src/matrix.c \
    /build/openmp/src/recommendations.c \
    -luuid -lm \
    -o /usr/local/lib/librec_engine_openmp.so \
 && gcc -shared -fPIC -O3 -std=c11 -fopenmp \
    -I/build/cuda/include \
    /build/cuda/src/data.c \
    /build/cuda/src/file_reader.c \
    /build/cuda/src/parser.c \
    /build/cuda/src/matrix.c \
    /build/cuda/src/cuda_compat.c \
    /build/cuda/src/recommendations.c \
    -luuid -lm \
    -o /usr/local/lib/librec_engine_cuda.so \
 && cp /usr/local/lib/librec_engine_openmp.so /usr/local/lib/librec_engine.so \
 && ldconfig \
 && rm -rf /build

COPY src/worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/worker/ .

CMD ["python", "worker.py"]
