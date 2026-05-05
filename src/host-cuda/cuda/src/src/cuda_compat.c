#include "matrix.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32

float *compute_similarity_cuda(const Matrix *matrix, const float *norms) {
  if (!matrix || !norms) return NULL;

  const int N = matrix->rows;
  const int M = matrix->cols;

  float *similarity = (float *)calloc(N * N, sizeof(float));
  if (!similarity) {
    fprintf(stderr, "[CUDA-CPU] Error: cannot alloc %d×%d similarity\n", N, N);
    return NULL;
  }

#pragma omp parallel for collapse(2) schedule(dynamic, 1)
  for (int bi = 0; bi < N; bi += TILE_SIZE) {
    for (int bj = 0; bj < N; bj += TILE_SIZE) {
      int i_end = (bi + TILE_SIZE < N) ? bi + TILE_SIZE : N;
      int j_end = (bj + TILE_SIZE < N) ? bj + TILE_SIZE : N;

      for (int i = bi; i < i_end; i++) {
        for (int j = bj; j < j_end; j++) {
          if (i == j) {
            similarity[i * N + j] = 1.0f;
            continue;
          }
          if (norms[i] <= 1e-6f || norms[j] <= 1e-6f) continue;

          const float *ri = &matrix->data[i * M];
          const float *rj = &matrix->data[j * M];
          float dot = 0.0f;
          int k = 0;

          for (; k <= M - 4; k += 4) {
            dot += ri[k] * rj[k];
            dot += ri[k + 1] * rj[k + 1];
            dot += ri[k + 2] * rj[k + 2];
            dot += ri[k + 3] * rj[k + 3];
          }
          for (; k < M; k++)
            dot += ri[k] * rj[k];

          float sim = dot / (norms[i] * norms[j]);
          similarity[i * N + j] = fminf(fmaxf(sim, -1.0f), 1.0f);
        }
      }
    }
  }
  return similarity;
}
