#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "data.h"
#include "file_reader.h"
#include "matrix.h"
#include "parser.h"

int main() {
  const char *filename = "../data/matrix.csv";

  printf("=== User-Item Matrix Similarity Computation ===\n\n");

  printf("Step 1: Loading matrix from %s\n", filename);
  clock_t start_time = clock();

  Matrix *matrix = load_matrix(filename);
  if (!matrix) {
    fprintf(stderr, "Error: Failed to load matrix\n");
    return 1;
  }

  clock_t load_time = clock();
  printf("Matrix loading time: %.4f seconds\n\n",
         (double)(load_time - start_time) / CLOCKS_PER_SEC);

  printf("Step 2: Matrix preview:\n");
  print_matrix(matrix);
  printf("\n");

  printf("Step 3: Computing norms for %d users...\n", matrix->rows);
  start_time = clock();

  float *norms = compute_norms(matrix);
  if (!norms) {
    fprintf(stderr, "Error: Failed to compute norms\n");
    free_matrix(matrix);
    return 1;
  }

  clock_t norm_time = clock();
  printf("Norm computation time: %.4f seconds\n\n",
         (double)(norm_time - start_time) / CLOCKS_PER_SEC);

  printf("Step 4: Computing cosine similarity matrix (%d x %d)...\n",
         matrix->rows, matrix->rows);
  start_time = clock();

  float *similarity = compute_similarity(matrix, norms);
  if (!similarity) {
    fprintf(stderr, "Error: Failed to compute similarity\n");
    free(norms);
    free_matrix(matrix);
    return 1;
  }

  clock_t similarity_time = clock();
  printf("Similarity computation time: %.4f seconds\n\n",
         (double)(similarity_time - start_time) / CLOCKS_PER_SEC);

  printf("Step 5: Similarity matrix preview (first 5x5):\n");
  int preview_size = (matrix->rows > 5) ? 5 : matrix->rows;

  for (int i = 0; i < preview_size; i++) {
    for (int j = 0; j < preview_size; j++) {
      printf("%.4f ", similarity[i * matrix->rows + j]);
    }
    printf("\n");
  }
  printf("\n");

  printf("=== Summary ===\n");
  printf("Users: %d\n", matrix->rows);
  printf("Items: %d\n", matrix->cols);
  printf("Total computation time: %.4f seconds\n",
         (double)(similarity_time - start_time) / CLOCKS_PER_SEC);

  free(similarity);
  free(norms);
  free_matrix(matrix);

  return 0;
}
