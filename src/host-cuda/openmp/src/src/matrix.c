#define _POSIX_C_SOURCE 200809L
#include "matrix.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <uuid/uuid.h>

#include "data.h"
#include "file_reader.h"
#include "parser.h"

typedef struct {
  uuid_t uuid;
  int index;
} UUIDMap;

Matrix *load_matrix(const char *filename) {
  FILE *file = open_file(filename);
  if (!file) {
    fprintf(stderr, "Error: Cannot open file %s\n", filename);
    return NULL;
  }

  int max_user_id = -1;
  UUIDMap *product_map = (UUIDMap *)malloc(10000 * sizeof(UUIDMap));
  int num_products = 0;

  char line[256];
  struct data entry;

  while (fgets(line, sizeof(line), file)) {
    if (parse_line(line, &entry)) {
      if (entry.user_id > max_user_id)
        max_user_id = entry.user_id;

      int found = 0;
      for (int i = 0; i < num_products; i++) {
        if (uuid_compare(product_map[i].uuid, entry.product_id) == 0) {
          found = 1;
          break;
        }
      }
      if (!found) {
        uuid_copy(product_map[num_products].uuid, entry.product_id);
        product_map[num_products].index = num_products;
        num_products++;
      }
    }
  }

  int num_users = max_user_id + 1;
  printf("[OpenMP] Loaded: %d users, %d products\n", num_users, num_products);

  Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
  matrix->rows = num_users;
  matrix->cols = num_products;
  matrix->data = (float *)calloc(num_users * num_products, sizeof(float));

  if (!matrix->data) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    free(matrix);
    free(product_map);
    fclose(file);
    return NULL;
  }

  rewind(file);
  while (fgets(line, sizeof(line), file)) {
    if (parse_line(line, &entry)) {
      int prod_idx = -1;
      for (int i = 0; i < num_products; i++) {
        if (uuid_compare(product_map[i].uuid, entry.product_id) == 0) {
          prod_idx = i;
          break;
        }
      }
      if (prod_idx >= 0) {
        matrix->data[entry.user_id * num_products + prod_idx] = entry.rating;
      }
    }
  }

  fclose(file);
  free(product_map);
  return matrix;
}

void free_matrix(Matrix *matrix) {
  if (matrix) {
    free(matrix->data);
    free(matrix);
  }
}

void print_matrix(const Matrix *matrix) {
  if (!matrix) return;
  printf("Matrix (%d x %d):\n", matrix->rows, matrix->cols);
  int max_rows = (matrix->rows > 5) ? 5 : matrix->rows;
  int max_cols = (matrix->cols > 5) ? 5 : matrix->cols;
  for (int i = 0; i < max_rows; i++) {
    for (int j = 0; j < max_cols; j++)
      printf("%.2f ", matrix->data[i * matrix->cols + j]);
    printf("\n");
  }
}

float *compute_norms(const Matrix *matrix) {
  if (!matrix) return NULL;

  float *norms = (float *)malloc(matrix->rows * sizeof(float));
  if (!norms) return NULL;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < matrix->rows; i++) {
    float sq = 0.0f;
    for (int j = 0; j < matrix->cols; j++) {
      float v = matrix->data[i * matrix->cols + j];
      sq += v * v;
    }
    norms[i] = sqrtf(sq);
  }
  return norms;
}

float *compute_similarity_omp(const Matrix *matrix, const float *norms) {
  if (!matrix || !norms) return NULL;

  const int N = matrix->rows;
  const int M = matrix->cols;

  float *similarity = (float *)calloc(N * N, sizeof(float));
  if (!similarity) {
    fprintf(stderr, "[OpenMP] Error: cannot alloc %d×%d similarity matrix\n", N, N);
    return NULL;
  }

#pragma omp parallel for collapse(2) schedule(dynamic, 4)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i == j) {
        similarity[i * N + j] = 1.0f;
        continue;
      }
      if (norms[i] <= 1e-10f || norms[j] <= 1e-10f) continue;

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
  return similarity;
}

float *compute_similarity(const Matrix *matrix, const float *norms) {
  return compute_similarity_omp(matrix, norms);
}
