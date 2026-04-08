#!/bin/bash
# Benchmarking Script for Task 3
# Measures execution time with 1, 2, 4, 8 threads
# Usage: ./benchmark.sh

PROJECT_DIR="/home/na/Desktop/Projects/rec-engine/src/host-cuda/openmp/src"
BINARY="$PROJECT_DIR/bin/similarity"
RESULTS_FILE="$PROJECT_DIR/../BENCHMARK_RESULTS.txt"

cd "$PROJECT_DIR"

echo "=== Recommendation Engine OpenMP Benchmarking ==="
echo "Binary: $BINARY"
echo "Starting benchmarks..."
echo ""

> "$RESULTS_FILE"

for THREADS in 1 2 4 8; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running with OMP_NUM_THREADS=$THREADS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    {
        echo ""
        echo "=== Benchmark Run: $THREADS Threads ==="
        echo "Timestamp: $(date)"
        echo ""
        OMP_NUM_THREADS=$THREADS time -v "$BINARY" 2>&1
        echo ""
    } >> "$RESULTS_FILE"
    
    echo ""
done

echo ""
echo "✅ Benchmarking complete!"
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "To view results:"
echo "  cat $RESULTS_FILE"
echo ""
echo "To calculate speedup:"
echo "  grep 'Total computation time' $RESULTS_FILE"
