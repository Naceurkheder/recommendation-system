#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_similarity.h"

#define CUDA_CHECK(call)                                                     \
  {                                                                          \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA Error: %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                      \
      return NULL;                                                           \
    }                                                                        \
  }

#define BLOCK_SIZE 32
#define TILE_SIZE 32

__global__ void compute_similarity_kernel_v2(float *matrix_data, float *norms,
                                             float *similarity, int num_users,
                                             int num_items) {
  __shared__ float shared_matrix[TILE_SIZE][TILE_SIZE];

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < num_users && j < num_users) {
    if (i == j) {
      similarity[i * num_users + j] = 1.0f;
      return;
    }

    if (norms[i] > 1e-6f && norms[j] > 1e-6f) {
      float dot_product = 0.0f;

      for (int k = 0; k < num_items; k += TILE_SIZE) {
        int tile_end = min(TILE_SIZE, num_items - k);

        if (threadIdx.x < tile_end) {
          shared_matrix[threadIdx.y][threadIdx.x] =
              matrix_data[i * num_items + k + threadIdx.x];
        }
        __syncthreads();

        for (int t = 0; t < tile_end; t++) {
          dot_product += shared_matrix[threadIdx.y][t] *
                         matrix_data[j * num_items + k + t];
        }
        __syncthreads();
      }

      float cosine_sim = dot_product / (norms[i] * norms[j]);
      cosine_sim = fminf(fmaxf(cosine_sim, -1.0f), 1.0f);
      similarity[i * num_users + j] = cosine_sim;
    } else {
      similarity[i * num_users + j] = 0.0f;
    }
  }
}

CUDAContext *cuda_init(const Matrix *matrix, const float *norms) {
  if (!matrix || !norms) {
    fprintf(stderr, "Error: Invalid input to cuda_init\n");
    return NULL;
  }

  CUDAContext *ctx = (CUDAContext *)malloc(sizeof(CUDAContext));
  if (!ctx) {
    fprintf(stderr, "Error: Memory allocation failed for CUDA context\n");
    return NULL;
  }

  ctx->num_users = matrix->rows;
  ctx->num_items = matrix->cols;

  size_t matrix_size = matrix->rows * matrix->cols * sizeof(float);
  size_t norms_size = matrix->rows * sizeof(float);
  size_t similarity_size = matrix->rows * matrix->rows * sizeof(float);

  printf("[CUDA] Initializing GPU memory:\n");
  printf("[CUDA]   Matrix: %.2f MB\n", matrix_size / (1024.0f * 1024.0f));
  printf("[CUDA]   Norms: %.2f MB\n", norms_size / (1024.0f * 1024.0f));
  printf("[CUDA]   Similarity: %.2f MB\n",
         similarity_size / (1024.0f * 1024.0f));

  CUDA_CHECK(cudaMalloc((void **)&ctx->d_matrix_data, matrix_size));
  CUDA_CHECK(cudaMalloc((void **)&ctx->d_norms, norms_size));
  CUDA_CHECK(cudaMalloc((void **)&ctx->d_similarity, similarity_size));

  CUDA_CHECK(cudaMemcpy(ctx->d_matrix_data, matrix->data, matrix_size,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(ctx->d_norms, norms, norms_size, cudaMemcpyHostToDevice));

  printf("[CUDA] GPU memory initialized successfully\n");

  return ctx;
}

void cuda_free(CUDAContext *ctx) {
  if (ctx) {
    if (ctx->d_matrix_data) {
      cudaFree(ctx->d_matrix_data);
    }
    if (ctx->d_norms) {
      cudaFree(ctx->d_norms);
    }
    if (ctx->d_similarity) {
      cudaFree(ctx->d_similarity);
    }
    free(ctx);
  }
}

float *cuda_compute_similarity(CUDAContext *ctx) {
  if (!ctx) {
    fprintf(stderr, "Error: Invalid CUDA context\n");
    return NULL;
  }

  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks((ctx->num_users + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (ctx->num_users + BLOCK_SIZE - 1) / BLOCK_SIZE);

  printf("[CUDA] Launching optimized kernel:\n");
  printf("[CUDA]   Grid: (%d, %d) blocks\n", blocks.x, blocks.y);
  printf("[CUDA]   Block: (%d, %d) threads\n", threads.x, threads.y);
  printf("[CUDA]   Total threads: %d\n",
         blocks.x * blocks.y * threads.x * threads.y);

  compute_similarity_kernel_v2<<<blocks, threads>>>(
      ctx->d_matrix_data, ctx->d_norms, ctx->d_similarity, ctx->num_users,
      ctx->num_items);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("[CUDA] Kernel execution completed\n");

  float *similarity =
      (float *)malloc(ctx->num_users * ctx->num_users * sizeof(float));
  if (!similarity) {
    fprintf(stderr, "Error: Host memory allocation failed\n");
    return NULL;
  }

  CUDA_CHECK(cudaMemcpy(similarity, ctx->d_similarity,
                        ctx->num_users * ctx->num_users * sizeof(float),
                        cudaMemcpyDeviceToHost));

  printf("[CUDA] Results copied back to host\n");

  return similarity;
}
