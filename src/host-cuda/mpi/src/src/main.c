#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "mpi_similarity.h"
#include "recommendations.h"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0 && argc < 2) {
    fprintf(stderr, "Usage: %s <data.csv>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  Matrix *matrix = NULL;
  float *norms = NULL;

  if (rank == 0) {
    matrix = load_matrix(argv[1]);
    if (!matrix) {
      fprintf(stderr, "Failed to load matrix\n");
      MPI_Finalize();
      return 1;
    }

    printf("Matrix loaded: %d users, %d items\n", matrix->rows, matrix->cols);
    print_matrix(matrix);

    norms = compute_norms(matrix);
    if (!norms) {
      fprintf(stderr, "Failed to compute norms\n");
      free_matrix(matrix);
      MPI_Finalize();
      return 1;
    }
  }

  MPIContext *ctx = mpi_init_context();
  if (!ctx) {
    fprintf(stderr, "[Rank %d] Failed to initialize MPI context\n", rank);
    MPI_Finalize();
    return 1;
  }

  printf("Computing similarity matrix...\n");
  float *similarity = mpi_compute_similarity(matrix, norms, ctx);

  if (rank == 0) {
    if (similarity) {
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
          printf("  Item %d: %.4f\n", items[i].item_id,
                 items[i].predicted_rating);
        }
        free(items);
      }

      free(similarity);
    }

    free(norms);
    free_matrix(matrix);
  }

  mpi_finalize_context(ctx);

  MPI_Finalize();

  return 0;
}
