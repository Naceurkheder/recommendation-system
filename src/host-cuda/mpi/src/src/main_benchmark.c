#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "data.h"
#include "file_reader.h"
#include "matrix.h"
#include "parser.h"
#include "recommendations.h"

// Forward declarations for different implementations
float *compute_similarity(const Matrix *matrix, const float *norms);
float *compute_norms(const Matrix *matrix);

int main() {
  const char *filename = "../data/matrix.csv";

  printf("╔════════════════════════════════════════════════════════════╗\n");
  printf("║  Recommendation Engine - OpenMP Optimized Version         ║\n");
  printf("║  High-Performance Collaborative Filtering System          ║\n");
  printf("╚════════════════════════════════════════════════════════════╝\n\n");

  printf("Step 1: Loading matrix from %s\n", filename);
  double start_time = omp_get_wtime();

  Matrix *matrix = load_matrix(filename);
  if (!matrix) {
    fprintf(stderr, "Error: Failed to load matrix\n");
    return 1;
  }

  double load_time = omp_get_wtime() - start_time;
  printf("✓ Matrix loading time: %.6f seconds\n\n", load_time);

  printf("Step 2: Matrix preview (first 5x5):\n");
  print_matrix(matrix);
  printf("\n");

  printf("Step 3: Computing L2 norms for %d users...\n", matrix->rows);
  start_time = omp_get_wtime();

  float *norms = compute_norms(matrix);
  if (!norms) {
    fprintf(stderr, "Error: Failed to compute norms\n");
    free_matrix(matrix);
    return 1;
  }

  double norm_time = omp_get_wtime() - start_time;
  printf("✓ Norm computation time: %.6f seconds\n\n", norm_time);

  printf("Step 4: Computing cosine similarity matrix (%d x %d)...\n",
         matrix->rows, matrix->rows);
  start_time = omp_get_wtime();

  float *similarity = compute_similarity(matrix, norms);
  if (!similarity) {
    fprintf(stderr, "Error: Failed to compute similarity\n");
    free(norms);
    free_matrix(matrix);
    return 1;
  }

  double similarity_time = omp_get_wtime() - start_time;
  printf("✓ Similarity computation time: %.6f seconds\n\n", similarity_time);

  printf("Step 5: Similarity Matrix Preview (first 5x5):\n");
  if (matrix->rows > 5 || matrix->rows > 5) {
    printf("(First 5 rows x 5 cols)\n");
  }
  int max_rows = (matrix->rows > 5) ? 5 : matrix->rows;
  int max_cols = (matrix->rows > 5) ? 5 : matrix->rows;
  for (int i = 0; i < max_rows; i++) {
    for (int j = 0; j < max_cols; j++) {
      printf("%.4f ", similarity[i * matrix->rows + j]);
    }
    printf("\n");
  }
  printf("\n");

  // Generate recommendations for sample users
  printf("Step 6: Generating recommendations for sample users...\n");
  int num_test_users = (matrix->rows > 5) ? 5 : matrix->rows;
  int k = 5;               // Top-5 recommendations
  int num_neighbors = 10;  // Use top-10 similar users

  for (int user = 0; user < num_test_users; user++) {
    printf("\n👤 Recommendations for User %d:\n", user);

    // Get similar users
    UserRec *similar_users =
        get_similar_users(similarity, user, k, matrix->rows);
    if (similar_users) {
      printf("   Similar Users:\n");
      for (int i = 0; i < k; i++) {
        printf("     • User %d (similarity: %.4f)\n", similar_users[i].user_id,
               similar_users[i].similarity_score);
      }
      free(similar_users);
    }

    // Get item recommendations
    ItemRec *item_recs =
        get_item_recommendations(similarity, matrix->data, user, k,
                                 matrix->rows, matrix->cols, num_neighbors);
    if (item_recs) {
      printf("   Item Recommendations (Top-%d):\n", k);
      for (int i = 0; i < k; i++) {
        printf("     • Item %d (predicted rating: %.4f)\n",
               item_recs[i].item_id, item_recs[i].predicted_rating);
      }
      free(item_recs);
    }
  }

  // Performance summary
  printf("\n╔════════════════════════════════════════════════════════════╗\n");
  printf("║                    TIMING SUMMARY                         ║\n");
  printf("╚════════════════════════════════════════════════════════════╝\n");

  double total_time = load_time + norm_time + similarity_time;
  printf("Loading:    %.6f seconds (%.2f%%)\n", load_time,
         100.0 * load_time / total_time);
  printf("Norms:      %.6f seconds (%.2f%%)\n", norm_time,
         100.0 * norm_time / total_time);
  printf("Similarity: %.6f seconds (%.2f%%)\n", similarity_time,
         100.0 * similarity_time / total_time);
  printf("─────────────────────────────────────────────────────────\n");
  printf("TOTAL:      %.6f seconds\n\n", total_time);

  // Performance metrics
  int num_users = matrix->rows;
  int num_items = matrix->cols;
  long long total_ops = (long long)num_users * num_users * num_items;
  double gflops = (total_ops / 1e9) / similarity_time;

  printf("Matrix dimensions: %d users × %d items\n", num_users, num_items);
  printf("Total FLOPs: %.2e\n", (double)total_ops);
  printf("Performance: %.2f GFLOPS\n", gflops);

  // Cleanup
  free(similarity);
  free(norms);
  free_matrix(matrix);

  printf("\n✓ Computation complete!\n");
  return 0;
}
