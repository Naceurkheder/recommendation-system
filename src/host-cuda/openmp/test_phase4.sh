#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/src" && pwd)"
cd "$PROJECT_DIR"

echo "Phase 4 Integration Test"
echo "========================================"
echo ""

TEST_PASSED=0
TEST_FAILED=0

test_executable() {
    local name=$1
    local binary=$2
    local cmd=$3
    
    echo -n "Testing $name... "
    if [ ! -f "$binary" ]; then
        echo "SKIP (binary not found)"
        return 0
    fi
    
    if timeout 60 bash -c "$cmd" > /tmp/test_output.txt 2>&1; then
        if grep -q "TOTAL:" /tmp/test_output.txt; then
            echo "PASS"
            ((TEST_PASSED++))
            return 0
        else
            echo "FAIL (no timing output)"
            ((TEST_FAILED++))
            return 1
        fi
    else
        echo "FAIL (execution error or timeout)"
        ((TEST_FAILED++))
        return 1
    fi
}

echo "Building all targets..."
make clean > /dev/null 2>&1
make openmp mpi > /dev/null 2>&1
echo "Build complete"
echo ""

test_executable "OpenMP" "bin/similarity_openmp" "./bin/similarity_openmp"
test_executable "MPI (4 procs)" "bin/similarity_mpi" "mpirun -np 4 ./bin/similarity_mpi"

echo ""
echo "========================================"
echo "Tests Passed: $TEST_PASSED"
echo "Tests Failed: $TEST_FAILED"

if [ $TEST_FAILED -eq 0 ]; then
    echo "✓ All tests passed"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi
