#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAGRANT_DIR="$PROJECT_ROOT/infra/vagrant"
DIST_DIR="$PROJECT_ROOT/dist"

OMP_SRC="$PROJECT_ROOT/src/host-cuda/openmp/src/src"
OMP_INC="$PROJECT_ROOT/src/host-cuda/openmp/src/include"
MPI_SRC="$PROJECT_ROOT/src/host-cuda/mpi/src/src"
MPI_INC="$PROJECT_ROOT/src/host-cuda/mpi/src/include"
CUDA_SRC="$PROJECT_ROOT/src/host-cuda/cuda/src/src"
CUDA_INC="$PROJECT_ROOT/src/host-cuda/cuda/src/include"

log()  { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }
die()  { printf "[%s] ERROR: %s\n" "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

log "Checking prerequisites"
command -v vagrant >/dev/null 2>&1 || die "vagrant not found"
command -v gcc     >/dev/null 2>&1 || die "gcc not found"

log "Compiling C engines on host"
mkdir -p "$DIST_DIR"
GCC_FLAGS="-O3 -std=c11 -fopenmp -luuid -lm"

gcc -shared -fPIC $GCC_FLAGS \
    -I"$OMP_INC" \
    "$OMP_SRC/data.c" "$OMP_SRC/file_reader.c" "$OMP_SRC/parser.c" \
    "$OMP_SRC/matrix.c" "$OMP_SRC/recommendations.c" \
    -o "$DIST_DIR/librec_engine_openmp.so" \
  || die "openmp build failed"

gcc -shared -fPIC $GCC_FLAGS \
    -I"$CUDA_INC" \
    "$CUDA_SRC/data.c" "$CUDA_SRC/file_reader.c" "$CUDA_SRC/parser.c" \
    "$CUDA_SRC/matrix.c" "$CUDA_SRC/cuda_compat.c" "$CUDA_SRC/recommendations.c" \
    -o "$DIST_DIR/librec_engine_cuda.so" \
  || die "cuda fallback build failed"

if command -v mpicc >/dev/null 2>&1; then
    mpicc -O3 -fopenmp -std=c11 \
        -I"$MPI_INC" \
        "$MPI_SRC/data.c" "$MPI_SRC/file_reader.c" "$MPI_SRC/parser.c" \
        "$MPI_SRC/matrix.c" "$MPI_SRC/recommendations.c" "$MPI_SRC/mpi_bench_main.c" \
        -o "$DIST_DIR/similarity_mpi_bench" \
        -luuid -lm \
      || die "mpi bench build failed"
fi

cp "$DIST_DIR/librec_engine_openmp.so" "$DIST_DIR/librec_engine.so"

[ -f "$PROJECT_ROOT/.env" ] || cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"

cd "$VAGRANT_DIR"
if ! vagrant status 2>/dev/null | grep -q "running"; then
    vagrant up
fi

log "Bringing up stack inside VM"
vagrant ssh -- bash /vagrant/scripts/run_all.sh
STACK_EXIT=$?

VM_IP="192.168.56.10"
log "Stack ready"
echo "  Platform    http://$VM_IP:3000"
echo "  API         http://$VM_IP:8000"
echo "  LocalStack  http://$VM_IP:4566/_localstack/health"
echo "  Postgres    $VM_IP:5432"
echo "  Redis       $VM_IP:6379"

exit $STACK_EXIT
