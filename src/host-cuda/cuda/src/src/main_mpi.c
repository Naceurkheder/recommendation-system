#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <uuid/uuid.h>

#include "data.h"
#include "file_reader.h"
#include "matrix.h"
#include "mpi_similarity.h"
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

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const char *filename = "../data/matrix.csv";

  if (rank == 0) {
    printf("=== User-Item Matrix Similarity Computation ===\n");
    printf("=== MPI Version (Distributed CPU) ===\n\n");
  }

  // Only rank 0 loads the matrix
  Matrix *matrix = NULL;
  double load_time = 0.0;

  if (rank == 0) {
    printf("Step 1: Loading matrix from %s\n", filename);
    double start = MPI_Wtime();

    matrix = load_matrix(filename);
    if (!matrix) {
      fprintf(stderr, "Error: Failed to load matrix\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    load_time = MPI_Wtime() - start;
    printf("Matrix loading time: %.6f seconds\n\n", load_time);
  }

  // Broadcast matrix dimensions
  int num_users = 0, num_items = 0;
  if (rank == 0) {
    num_users = matrix->rows;
    num_items = matrix->cols;
  }
  MPI_Bcast(&num_users, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&num_items, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Allocate matrix on all processes
  if (rank != 0) {
    matrix = (Matrix *)malloc(sizeof(Matrix));
    matrix->rows = num_users;
    matrix->cols = num_items;
    matrix->data = (float *)malloc(num_users * num_items * sizeof(float));
  }

  // Broadcast matrix data
  MPI_Bcast(matrix->data, num_users * num_items, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Compute norms
  if (rank == 0) {
    printf("Step 2: Computing norms for %d users...\n", matrix->rows);
  }
  double norm_start = MPI_Wtime();

  float *norms = compute_norms(matrix);
  if (!norms) {
    fprintf(stderr, "[Rank %d] Error: Failed to compute norms\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  double norm_time = MPI_Wtime() - norm_start;

  if (rank == 0) {
    printf("Norm computation time: %.6f seconds\n\n", norm_time);
  }

  // Broadcast norms
  MPI_Bcast(norms, num_users, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Initialize MPI context
  MPIContext *ctx = mpi_init_context();

  if (rank == 0) {
    printf("Step 3: Computing cosine similarity matrix (%d x %d) with MPI...\n",
           matrix->rows, matrix->rows);
  }

  double sim_start = MPI_Wtime();
  float *similarity = mpi_compute_similarity(matrix, norms, ctx);
  double sim_time = MPI_Wtime() - sim_start;

  if (rank == 0) {
    printf("Similarity computation time: %.6f seconds\n\n", sim_time);

    printf("Step 4: Similarity Matrix Preview (first 5x5):\n");
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
    double total_time = load_time + norm_time + sim_time;
    printf("=== TIMING SUMMARY ===\n");
    printf("Loading:    %.6f seconds (%.2f%%)\n", load_time,
           100.0 * load_time / total_time);
    printf("Norms:      %.6f seconds (%.2f%%)\n", norm_time,
           100.0 * norm_time / total_time);
    printf("Similarity: %.6f seconds (%.2f%%)\n", sim_time,
           100.0 * sim_time / total_time);
    printf("TOTAL:      %.6f seconds\n\n", total_time);

    free(similarity);
  }

  // Cleanup
  free(norms);
  free_matrix(matrix);
  mpi_finalize_context(ctx);

  MPI_Finalize();

  return 0;
}
