/*
 * Performance Tuning Configuration
 * Advanced optimization parameters for recommendation engine
 */

#ifndef PERF_CONFIG_H
#define PERF_CONFIG_H

// ============================================================================
// OpenMP Configuration
// ============================================================================

// Number of threads (0 = auto-detect)
#define OPENMP_NUM_THREADS 0

// Scheduling strategy:
// - "static": Best for homogeneous work distribution
// - "dynamic": Better for load imbalance
// - "guided": Balance between static and dynamic
#define OPENMP_SCHEDULE "guided"

// Chunk size for scheduling (0 = auto)
#define OPENMP_CHUNK_SIZE 0

// ============================================================================
// CUDA Configuration
// ============================================================================

// Block size for similarity kernel
// 32x32 = 1024 threads/block (optimal for most architectures)
#define CUDA_BLOCK_SIZE 32

// Tile size for shared memory
#define CUDA_TILE_SIZE 32

// Maximum threads per block
#define CUDA_MAX_THREADS_PER_BLOCK 1024

// Enable CUDA graphs for kernel launching (CUDA 11+)
#define CUDA_USE_GRAPHS 0

// ============================================================================
// Memory Optimization
// ============================================================================

// Use pinned (page-locked) memory for CPU-GPU transfers (faster but uses system
// RAM)
#define USE_PINNED_MEMORY 1

// Prefetch data to GPU before computation
#define CUDA_PREFETCH_DATA 1

// ============================================================================
// Algorithm Tuning
// ============================================================================

// Similarity matrix symmetry optimization (saves 50% computation)
#define COMPUTE_SYMMETRIC_ONLY 1

// SIMD unroll factor for similarity computation
// Increase for better vectorization, but may hurt cache
#define SIMD_UNROLL_FACTOR 4

// Top-K quickselect threshold (use full sort if k > threshold * n)
#define TOPK_QUICKSELECT_THRESHOLD 0.5

// ============================================================================
// Parallelization Strategies
// ============================================================================

// MPI buffer size for collective operations
#define MPI_BUFFER_SIZE (64 * 1024 * 1024)  // 64 MB

// Hybrid MPI+OpenMP threads per MPI process
#define MPI_THREADS_PER_RANK 4

// Use MPI non-blocking operations for overlapping compute/communication
#define MPI_USE_IRECV 1

// ============================================================================
// Numerical Parameters
// ============================================================================

// Similarity threshold (values below this are zeroed)
#define SIMILARITY_THRESHOLD 0.0f

// Norm threshold (vectors with norm below this are ignored)
#define NORM_THRESHOLD 1e-10f

// ============================================================================
// Debugging & Profiling
// ============================================================================

// Verbosity level: 0=silent, 1=error, 2=warning, 3=info, 4=debug
#define VERBOSITY_LEVEL 2

// Enable performance counters
#define ENABLE_PERF_COUNTERS 1

// Profile memory allocation/deallocation
#define PROFILE_MEMORY 0

// ============================================================================
// Validation
// ============================================================================

// Validate results (slower but catches bugs)
#define VALIDATE_RESULTS 1

// Check for NaN/Inf values
#define CHECK_NUMERICAL_STABILITY 1

#endif
