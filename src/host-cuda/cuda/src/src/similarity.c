#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <uuid/uuid.h>

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

// Highly optimized cosine similarity computation with cache-friendly access
float *compute_similarity(const Matrix *matrix, const float *norms) {
  if (!matrix || !norms) return NULL;

  int num_users = matrix->rows;
  int num_items = matrix->cols;

  float *similarity = (float *)calloc(num_users * num_users, sizeof(float));
  if (!similarity) {
    fprintf(stderr, "Error: Memory allocation failed for similarity matrix\n");
    return NULL;
  }

// Diagonal: self-similarity = 1.0
#pragma omp parallel for schedule(static)
  for (int i = 0; i < num_users; i++) {
    similarity[i * num_users + i] = 1.0f;
  }

// Compute upper triangle only, then mirror (symmetric matrix optimization)
// Use guided schedule for load balancing: outer loop has fewer iterations
#pragma omp parallel for schedule(guided) collapse(2)
  for (int i = 0; i < num_users; i++) {
    for (int j = i + 1; j < num_users; j++) {
      if (norms[i] > 1e-10f && norms[j] > 1e-10f) {
        // Compute dot product with SIMD optimization
        float dot_product = 0.0f;

        // Pointers for better cache locality
        const float *row_i = &matrix->data[i * num_items];
        const float *row_j = &matrix->data[j * num_items];

        // Unroll loop for better SIMD vectorization
        int k = 0;
        for (; k <= num_items - 4; k += 4) {
          dot_product += row_i[k] * row_j[k];
          dot_product += row_i[k + 1] * row_j[k + 1];
          dot_product += row_i[k + 2] * row_j[k + 2];
          dot_product += row_i[k + 3] * row_j[k + 3];
        }

        // Process remaining elements
        for (; k < num_items; k++) {
          dot_product += row_i[k] * row_j[k];
        }

        // Compute cosine similarity with numerical stability
        float cosine_sim = dot_product / (norms[i] * norms[j]);

        // Clamp to [-1, 1] range for numerical stability
        cosine_sim = (cosine_sim > 1.0f)
                         ? 1.0f
                         : (cosine_sim < -1.0f ? -1.0f : cosine_sim);

        // Store symmetric values
        similarity[i * num_users + j] = cosine_sim;
        similarity[j * num_users + i] = cosine_sim;
      } else {
        similarity[i * num_users + j] = 0.0f;
        similarity[j * num_users + i] = 0.0f;
      }
    }
  }

  return similarity;
}
