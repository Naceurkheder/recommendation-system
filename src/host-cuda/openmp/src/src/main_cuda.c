#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <uuid/uuid.h>

#include "cuda_similarity.h"
#include "data.h"
#include "file_reader.h"
#include "matrix.h"
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
  int num_interactions = 0;

  UUIDMap *product_map = (UUIDMap *)malloc(10000 * sizeof(UUIDMap));
  int num_products = 0;

  char line[256];
  struct data entry;

  while (fgets(line, sizeof(line), file)) {
    if (parse_line(line, &entry)) {
      if (entry.user_id > max_user_id) {
        max_user_id = entry.user_id;
      }

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

      num_interactions++;
    }
  }

  int num_users = max_user_id + 1;

  printf("Loaded: %d users, %d products, %d interactions\n", num_users,
         num_products, num_interactions);

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
        int index = entry.user_id * num_products + prod_idx;
        matrix->data[index] = entry.rating;
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

  if (matrix->rows > 100 || matrix->cols > 100) {
    printf("(Matrix too large to print - showing first 5x5)\n");
  }

  int max_rows = (matrix->rows > 5) ? 5 : matrix->rows;
  int max_cols = (matrix->cols > 5) ? 5 : matrix->cols;

  for (int i = 0; i < max_rows; i++) {
    for (int j = 0; j < max_cols; j++) {
      printf("%.2f ", matrix->data[i * matrix->cols + j]);
    }
    printf("\n");
  }
}

float *compute_norms(const Matrix *matrix) {
  if (!matrix) return NULL;

  float *norms = (float *)malloc(matrix->rows * sizeof(float));
  if (!norms) {
    fprintf(stderr, "Error: Memory allocation failed for norms\n");
    return NULL;
  }

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < matrix->rows; i++) {
    float norm_sq = 0.0f;

    for (int j = 0; j < matrix->cols; j++) {
      float val = matrix->data[i * matrix->cols + j];
      norm_sq += val * val;
    }

    norms[i] = sqrtf(norm_sq);
  }

  return norms;
}

int main() {
  const char *filename = "../data/matrix.csv";

  printf("=== User-Item Matrix Similarity Computation ===\n");
  printf("=== CUDA Version (GPU Accelerated) ===\n\n");

  printf("Step 1: Loading matrix from %s\n", filename);
  double start_time = omp_get_wtime();

  Matrix *matrix = load_matrix(filename);
  if (!matrix) {
    fprintf(stderr, "Error: Failed to load matrix\n");
    return 1;
  }

  double load_time = omp_get_wtime() - start_time;
  printf("Matrix loading time: %.6f seconds\n\n", load_time);

  printf("Step 2: Matrix preview:\n");
  print_matrix(matrix);
  printf("\n");

  printf("Step 3: Computing norms for %d users...\n", matrix->rows);
  start_time = omp_get_wtime();

  float *norms = compute_norms(matrix);
  if (!norms) {
    fprintf(stderr, "Error: Failed to compute norms\n");
    free_matrix(matrix);
    return 1;
  }

  double norm_time = omp_get_wtime() - start_time;
  printf("Norm computation time: %.6f seconds\n\n", norm_time);

  printf("Step 4: Initializing CUDA and computing similarity...\n");
  start_time = omp_get_wtime();

  CUDAContext *cuda_ctx = cuda_init(matrix, norms);
  if (!cuda_ctx) {
    fprintf(stderr, "Error: Failed to initialize CUDA\n");
    free(norms);
    free_matrix(matrix);
    return 1;
  }

  double cuda_init_time = omp_get_wtime() - start_time;
  printf("CUDA initialization time: %.6f seconds\n", cuda_init_time);

  start_time = omp_get_wtime();
  float *similarity = cuda_compute_similarity(cuda_ctx);
  if (!similarity) {
    fprintf(stderr, "Error: Failed to compute similarity on GPU\n");
    cuda_free(cuda_ctx);
    free(norms);
    free_matrix(matrix);
    return 1;
  }
  double compute_time = omp_get_wtime() - start_time;
  printf("GPU similarity computation time: %.6f seconds\n\n", compute_time);

  double similarity_time = cuda_init_time + compute_time;

  printf("Step 5: Similarity Matrix Preview (first 5x5):\n");
  if (matrix->rows > 5 || matrix->rows > 5) {
    printf("(First 5 rows x 5 cols)\n");
  }
  int max_rows = (matrix->rows > 5) ? 5 : matrix->rows;
  int max_cols = (matrix->rows > 5) ? 5 : matrix->rows;
  for (int i = 0; i < max_rows; i++) {
    for (int j = 0; j < max_cols; j++) {
      printf("%.2f ", similarity[i * matrix->rows + j]);
    }
    printf("\n");
  }
  printf("\n");

  // Summary
  double total_time = load_time + norm_time + similarity_time;
  printf("=== TIMING SUMMARY ===\n");
  printf("Loading:    %.6f seconds (%.2f%%)\n", load_time,
         100.0 * load_time / total_time);
  printf("Norms:      %.6f seconds (%.2f%%)\n", norm_time,
         100.0 * norm_time / total_time);
  printf("Similarity: %.6f seconds (%.2f%%)\n", similarity_time,
         100.0 * similarity_time / total_time);
  printf("TOTAL:      %.6f seconds\n\n", total_time);

  // Cleanup
  free(similarity);
  cuda_free(cuda_ctx);
  free(norms);
  free_matrix(matrix);

  return 0;
}
