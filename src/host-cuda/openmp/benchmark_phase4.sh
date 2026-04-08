#!/bin/bash

# Phase 4 Benchmarking Script
# Compares performance of OpenMP, MPI, and CUDA implementations

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Phase 4: Advanced Parallelism - Benchmarking & Comparison    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Results file
RESULTS_FILE="PHASE4_BENCHMARK_RESULTS.txt"
> "$RESULTS_FILE"

# Function to extract timing from output
extract_timing() {
    local output="$1"
    local label="$2"
    echo "$output" | grep "${label}:" | awk '{print $NF}' | head -1
}

# Function to run benchmark and capture results
run_benchmark() {
    local impl_name=$1
    local cmd=$2
    local output_var=$3
    
    echo -e "${BLUE}[${impl_name}]${NC} Running benchmark..."
    
    if eval "$cmd" > /tmp/bench_output.txt 2>&1; then
        local output=$(cat /tmp/bench_output.txt)
        eval "$output_var='$output'"
        echo -e "${GREEN}[${impl_name}]${NC} Benchmark completed"
        return 0
    else
        echo -e "${YELLOW}[${impl_name}]${NC} Benchmark failed or not available"
        return 1
    fi
}

# Check if MPI is available
check_mpi() {
    if command -v mpicc &> /dev/null; then
        echo -e "${GREEN}✓${NC} MPI (mpicc) is available"
        return 0
    else
        echo -e "${YELLOW}✗${NC} MPI (mpicc) not found. Install with: sudo apt-get install libopenmpi-dev"
        return 1
    fi
}

# Check if CUDA is available
check_cuda() {
    if command -v nvcc &> /dev/null; then
        echo -e "${GREEN}✓${NC} CUDA (nvcc) is available"
        nvcc --version
        return 0
    else
        echo -e "${YELLOW}✗${NC} CUDA (nvcc) not found. Check GPU setup"
        return 1
    fi
}

# Check if GPU is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓${NC} GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        return 0
    else
        echo -e "${YELLOW}✗${NC} No GPU detected (nvidia-smi not found)"
        return 1
    fi
}

echo "═══ Prerequisites Check ═══"
echo ""

echo "Checking for MPI..."
HAS_MPI=0
check_mpi && HAS_MPI=1
echo ""

echo "Checking for CUDA..."
HAS_CUDA=0
check_cuda && HAS_CUDA=1
echo ""

echo "Checking for GPU..."
HAS_GPU=0
check_gpu && HAS_GPU=1
echo ""

echo "═══ Build Phase ═══"
echo ""

# Build OpenMP
echo -e "${BLUE}Building OpenMP version...${NC}"
if make -C "$PROJECT_DIR/src" openmp -j4; then
    echo -e "${GREEN}✓ OpenMP build successful${NC}"
    echo ""
else
    echo -e "${YELLOW}✗ OpenMP build failed${NC}"
    exit 1
fi

# Build MPI if available
MPI_BINARY=""
if [ $HAS_MPI -eq 1 ]; then
    echo -e "${BLUE}Building MPI version...${NC}"
    if make -C "$PROJECT_DIR/src" mpi -j4; then
        echo -e "${GREEN}✓ MPI build successful${NC}"
        MPI_BINARY="$PROJECT_DIR/src/bin/similarity_mpi"
        echo ""
    else
        echo -e "${YELLOW}✗ MPI build failed${NC}"
        HAS_MPI=0
        echo ""
    fi
fi

# Build CUDA if available
CUDA_BINARY=""
if [ $HAS_CUDA -eq 1 ] && [ $HAS_GPU -eq 1 ]; then
    echo -e "${BLUE}Building CUDA version...${NC}"
    if make -C "$PROJECT_DIR/src" cuda -j4; then
        echo -e "${GREEN}✓ CUDA build successful${NC}"
        CUDA_BINARY="$PROJECT_DIR/src/bin/similarity_cuda"
        echo ""
    else
        echo -e "${YELLOW}✗ CUDA build failed${NC}"
        HAS_CUDA=0
        echo ""
    fi
else
    if [ $HAS_CUDA -eq 0 ]; then
        echo -e "${YELLOW}⊘ Skipping CUDA build (CUDA not available)${NC}"
    elif [ $HAS_GPU -eq 0 ]; then
        echo -e "${YELLOW}⊘ Skipping CUDA build (No GPU detected)${NC}"
    fi
    echo ""
fi

echo "═══ Benchmark Phase ═══"
echo ""

# Initialize results
declare -A timing_results
declare -A speedup_results

# Run OpenMP benchmark
echo -e "${BLUE}[OpenMP]${NC} Running benchmark (3x)..."
openmp_times=()
for i in {1..3}; do
    echo "  Run $i/3..."
    if timeout 300 "$PROJECT_DIR/src/bin/similarity_openmp" > /tmp/omp_output.txt 2>&1; then
        timing=$(grep "TOTAL:" /tmp/omp_output.txt | awk '{print $(NF-1)}')
        openmp_times+=($timing)
        echo "    Time: ${timing}s"
    fi
done

if [ ${#openmp_times[@]} -gt 0 ]; then
    openmp_avg=$(printf "%s\n" "${openmp_times[@]}" | awk '{sum+=$1} END {print sum/NR}')
    timing_results["OpenMP"]=$openmp_avg
    echo -e "${GREEN}✓ OpenMP Average: ${openmp_avg}s${NC}"
    cat /tmp/omp_output.txt >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
fi
echo ""

# Run MPI benchmark if available
if [ $HAS_MPI -eq 1 ] && [ ! -z "$MPI_BINARY" ] && [ -x "$MPI_BINARY" ]; then
    echo -e "${BLUE}[MPI]${NC} Running benchmark (3x with 4 processes)..."
    mpi_times=()
    for i in {1..3}; do
        echo "  Run $i/3..."
        if timeout 300 mpirun -np 4 "$MPI_BINARY" > /tmp/mpi_output.txt 2>&1; then
            timing=$(grep "TOTAL:" /tmp/mpi_output.txt | awk '{print $(NF-1)}')
            mpi_times+=($timing)
            echo "    Time: ${timing}s"
        fi
    done
    
    if [ ${#mpi_times[@]} -gt 0 ]; then
        mpi_avg=$(printf "%s\n" "${mpi_times[@]}" | awk '{sum+=$1} END {print sum/NR}')
        timing_results["MPI"]=$mpi_avg
        speedup=$(echo "scale=2; ${timing_results[OpenMP]} / $mpi_avg" | bc)
        speedup_results["MPI"]=$speedup
        echo -e "${GREEN}✓ MPI Average: ${mpi_avg}s (Speedup: ${speedup}x)${NC}"
        cat /tmp/mpi_output.txt >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
    fi
else
    echo -e "${YELLOW}⊘ MPI benchmark skipped (not available or not built)${NC}"
fi
echo ""

# Run CUDA benchmark if available
if [ $HAS_CUDA -eq 1 ] && [ ! -z "$CUDA_BINARY" ] && [ -x "$CUDA_BINARY" ]; then
    echo -e "${BLUE}[CUDA]${NC} Running benchmark (3x)..."
    cuda_times=()
    for i in {1..3}; do
        echo "  Run $i/3..."
        if timeout 300 "$CUDA_BINARY" > /tmp/cuda_output.txt 2>&1; then
            timing=$(grep "TOTAL:" /tmp/cuda_output.txt | awk '{print $(NF-1)}')
            cuda_times+=($timing)
            echo "    Time: ${timing}s"
        fi
    done
    
    if [ ${#cuda_times[@]} -gt 0 ]; then
        cuda_avg=$(printf "%s\n" "${cuda_times[@]}" | awk '{sum+=$1} END {print sum/NR}')
        timing_results["CUDA"]=$cuda_avg
        speedup=$(echo "scale=2; ${timing_results[OpenMP]} / $cuda_avg" | bc)
        speedup_results["CUDA"]=$speedup
        echo -e "${GREEN}✓ CUDA Average: ${cuda_avg}s (Speedup: ${speedup}x)${NC}"
        cat /tmp/cuda_output.txt >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
    fi
else
    echo -e "${YELLOW}⊘ CUDA benchmark skipped (not available or no GPU)${NC}"
fi
echo ""

echo "═══ Results Summary ═══"
echo ""

# Create results table
echo "| Implementation | Avg Time (s) | Speedup vs OpenMP |" | tee -a "$RESULTS_FILE"
echo "|---|---|---|" | tee -a "$RESULTS_FILE"

for impl in "OpenMP" "MPI" "CUDA"; do
    if [ ! -z "${timing_results[$impl]}" ]; then
        if [ "$impl" = "OpenMP" ]; then
            speedup="1.00x (baseline)"
        else
            speedup="${speedup_results[$impl]}x"
        fi
        printf "| %-14s | %-12.6f | %-17s |\n" "$impl" "${timing_results[$impl]}" "$speedup" | tee -a "$RESULTS_FILE"
    fi
done

echo ""
echo "═══ Performance Analysis ═══"
echo ""

# Find best implementation
best_impl=""
best_time=999999
for impl in "${!timing_results[@]}"; do
    if (( $(echo "${timing_results[$impl]} < $best_time" | bc -l) )); then
        best_time="${timing_results[$impl]}"
        best_impl="$impl"
    fi
done

if [ ! -z "$best_impl" ]; then
    echo -e "${GREEN}✓ Best Performance: ${best_impl}${NC} (${best_time}s)"
    echo "Best Performance: $best_impl (${best_time}s)" >> "$RESULTS_FILE"
fi

echo ""
echo "═══ Details ═══"
echo ""
echo "• Full results saved to: $RESULTS_FILE"
echo "• OpenMP: Sequential reference implementation using shared-memory parallelism"
echo "• MPI: Distributed CPU parallelism across multiple processes"
echo "• CUDA: GPU-accelerated implementation"
echo ""
echo -e "${GREEN}✓ Benchmarking complete!${NC}"
echo ""
