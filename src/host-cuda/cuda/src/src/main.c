#include <stdio.h>
#include <stdlib.h>

#include "cuda_similarity.h"
#include "matrix.h"
#include "recommendations.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <data.csv>\n", argv[0]);
    return 1;
  }

  Matrix *matrix = load_matrix(argv[1]);
  if (!matrix) {
    fprintf(stderr, "Failed to load matrix\n");
    return 1;
  }

  printf("Matrix loaded: %d users, %d items\n", matrix->rows, matrix->cols);
  print_matrix(matrix);

  float *norms = compute_norms(matrix);
  if (!norms) {
    fprintf(stderr, "Failed to compute norms\n");
    free_matrix(matrix);
    return 1;
  }

  printf("Initializing GPU...\n");
  CUDAContext *ctx = cuda_init(matrix, norms);
  if (!ctx) {
    fprintf(stderr, "Failed to initialize CUDA context\n");
    free(norms);
    free_matrix(matrix);
    return 1;
  }

  printf("Computing similarity matrix on GPU...\n");
  float *similarity = cuda_compute_similarity(ctx);
  if (!similarity) {
    fprintf(stderr, "Failed to compute similarity\n");
    cuda_free(ctx);
    free(norms);
    free_matrix(matrix);
    return 1;
  }

  printf("Getting recommendations for user 0...\n");
  UserRec *similar = get_similar_users(similarity, 0, 5, matrix->rows);
  if (similar) {
    printf("Similar users:\n");
    for (int i = 0; i < 5; i++) {
      printf("  User %d: %.4f\n", similar[i].user_id,
             similar[i].similarity_score);
    }
    free(similar);
  }

  ItemRec *items = get_item_recommendations(similarity, matrix->data, 0, 5,
                                            matrix->rows, matrix->cols, 5);
  if (items) {
    printf("Recommended items:\n");
    for (int i = 0; i < 5; i++) {
      printf("  Item %d: %.4f\n", items[i].item_id, items[i].predicted_rating);
    }
    free(items);
  }

  free(similarity);
  cuda_free(ctx);
  free(norms);
  free_matrix(matrix);

  return 0;
}
