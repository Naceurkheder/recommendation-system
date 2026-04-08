#define _POSIX_C_SOURCE 200809L
#include "mpi_similarity.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

MPIContext *mpi_init_context() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPIContext *ctx = (MPIContext *)malloc(sizeof(MPIContext));
  if (!ctx) {
    fprintf(stderr, "Error: Memory allocation failed for MPI context\n");
    return NULL;
  }

  ctx->rank = rank;
  ctx->size = size;

  if (rank == 0) {
    printf("[MPI] Initialized with %d processes\n", size);
  }

  return ctx;
}

void mpi_finalize_context(MPIContext *ctx) {
  if (ctx) {
    free(ctx);
  }
}

float *mpi_compute_similarity(const Matrix *matrix, const float *norms,
                              MPIContext *ctx) {
  if (!matrix || !norms || !ctx) {
    return NULL;
  }

  int num_users = matrix->rows;
  int num_items = matrix->cols;
  int rank = ctx->rank;
  int size = ctx->size;

  // Load-balanced row distribution
  int rows_per_process = num_users / size;
  int extra_rows = num_users % size;

  int start_row =
      rank * rows_per_process + (rank < extra_rows ? rank : extra_rows);
  int local_rows = rows_per_process + (rank < extra_rows ? 1 : 0);
  int end_row = start_row + local_rows;

  if (rank == 0) {
    printf("[MPI] Computing similarity with %d users across %d processes\n",
           num_users, size);
    printf("[MPI] Each process: rows=%d items=%d\n", local_rows, num_items);
  }

  // Allocate local similarity matrix
  float *local_similarity =
      (float *)calloc(local_rows * num_users, sizeof(float));
  if (!local_similarity) {
    fprintf(stderr, "[MPI Rank %d] Error: Memory allocation failed\n", rank);
    return NULL;
  }

// Hybrid MPI+OpenMP: compute rows in parallel on each process
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < local_rows; i++) {
    int global_i = start_row + i;

    // Diagonal: self-similarity = 1.0
    local_similarity[i * num_users + global_i] = 1.0f;

    // Compute similarities with loop unrolling for SIMD
    const float *row_i = &matrix->data[global_i * num_items];

    for (int j = 0; j < num_users; j++) {
      if (global_i != j && norms[global_i] > 1e-10f && norms[j] > 1e-10f) {
        float dot_product = 0.0f;
        const float *row_j = &matrix->data[j * num_items];

        // Vectorization-friendly inner loop
        int k = 0;
        for (; k <= num_items - 4; k += 4) {
          dot_product += row_i[k] * row_j[k];
          dot_product += row_i[k + 1] * row_j[k + 1];
          dot_product += row_i[k + 2] * row_j[k + 2];
          dot_product += row_i[k + 3] * row_j[k + 3];
        }

        for (; k < num_items; k++) {
          dot_product += row_i[k] * row_j[k];
        }

        float cosine_sim = dot_product / (norms[global_i] * norms[j]);
        // Numerical stability
        cosine_sim = (cosine_sim > 1.0f)
                         ? 1.0f
                         : (cosine_sim < -1.0f ? -1.0f : cosine_sim);
        local_similarity[i * num_users + j] = cosine_sim;
      } else {
        local_similarity[i * num_users + j] = 0.0f;
      }
    }
  }

  if (rank == 0) {
    printf("[MPI Rank 0] Rank %d computed %d rows (rows %d-%d)\n", rank,
           local_rows, start_row, end_row - 1);
  }

  // Gather all results to rank 0
  int *recv_counts = NULL;
  int *displs = NULL;

  if (rank == 0) {
    recv_counts = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));

    int displacement = 0;
    for (int p = 0; p < size; p++) {
      int p_rows = num_users / size;
      if (p < (num_users % size)) {
        p_rows++;
      }
      recv_counts[p] = p_rows * num_users;
      displs[p] = displacement;
      displacement += p_rows * num_users;
    }
  }

  float *global_similarity = NULL;
  if (rank == 0) {
    global_similarity = (float *)calloc(num_users * num_users, sizeof(float));
    if (!global_similarity) {
      fprintf(stderr, "[MPI Rank 0] Error: Memory allocation failed\n");
      free(recv_counts);
      free(displs);
      free(local_similarity);
      return NULL;
    }
  }

  // Use MPI_Gatherv to collect results
  int local_size = local_rows * num_users;
  MPI_Gatherv(local_similarity, local_size, MPI_FLOAT, global_similarity,
              rank == 0 ? recv_counts : NULL, rank == 0 ? displs : NULL,
              MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("[MPI Rank 0] Successfully gathered similarity matrix\n");
    free(recv_counts);
    free(displs);
  }

  free(local_similarity);

  return rank == 0 ? global_similarity : NULL;
}
