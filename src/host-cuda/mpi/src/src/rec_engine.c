/**
 * Recommendation Engine C Extension
 * Provides CTYpes-compatible interface for Python integration
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"
#include "recommendations.h"

// Global state for the recommendation engine
typedef struct {
  Matrix *matrix;
  float *norms;
  float *similarity;
  int initialized;
} RecEngineState;

static RecEngineState engine = {0};

/**
 * Initialize the recommendation engine
 * Called once to load data and precompute similarities
 */
int rec_engine_init(const char *csv_filename) {
  if (engine.initialized) {
    fprintf(stderr, "Warning: Engine already initialized\n");
    return 1;
  }

  // Load matrix from CSV
  engine.matrix = load_matrix(csv_filename);
  if (!engine.matrix) {
    fprintf(stderr, "Error: Failed to load matrix from %s\n", csv_filename);
    return 0;
  }

  printf("[RecEngine] Loaded matrix: %d users, %d items\n", engine.matrix->rows,
         engine.matrix->cols);

  // Compute norms
  engine.norms = compute_norms(engine.matrix);
  if (!engine.norms) {
    fprintf(stderr, "Error: Failed to compute norms\n");
    free_matrix(engine.matrix);
    engine.matrix = NULL;
    return 0;
  }

  printf("[RecEngine] Computed norms for %d users\n", engine.matrix->rows);

  // Compute similarity matrix (precomputation)
  engine.similarity = compute_similarity(engine.matrix, engine.norms);
  if (!engine.similarity) {
    fprintf(stderr, "Error: Failed to compute similarity matrix\n");
    free(engine.norms);
    engine.norms = NULL;
    free_matrix(engine.matrix);
    engine.matrix = NULL;
    return 0;
  }

  printf("[RecEngine] Computed similarity matrix (%d x %d)\n",
         engine.matrix->rows, engine.matrix->rows);

  engine.initialized = 1;
  return 1;
}

/**
 * Get similar users for a given user ID
 * Returns array of user IDs sorted by similarity
 */
int *rec_engine_get_similar_users(int user_id, int k) {
  if (!engine.initialized || !engine.similarity) {
    fprintf(stderr, "Error: Engine not initialized\n");
    return NULL;
  }

  if (user_id < 0 || user_id >= engine.matrix->rows) {
    fprintf(stderr, "Error: Invalid user ID: %d\n", user_id);
    return NULL;
  }

  UserRec *results =
      get_similar_users(engine.similarity, user_id, k, engine.matrix->rows);
  if (!results) return NULL;

  // Convert to simple int array (user IDs only)
  int *user_ids = (int *)malloc((k + 1) * sizeof(int));
  if (!user_ids) {
    free(results);
    return NULL;
  }

  for (int i = 0; i < k; i++) {
    user_ids[i] = results[i].user_id;
  }
  user_ids[k] = -1;  // Sentinel value

  free(results);
  return user_ids;
}

/**
 * Get item recommendations for a user
 * Returns array of item IDs sorted by predicted rating
 */
int *rec_engine_get_item_recommendations(int user_id, int k,
                                         int num_neighbors) {
  if (!engine.initialized || !engine.similarity || !engine.matrix) {
    fprintf(stderr, "Error: Engine not initialized\n");
    return NULL;
  }

  if (user_id < 0 || user_id >= engine.matrix->rows) {
    fprintf(stderr, "Error: Invalid user ID: %d\n", user_id);
    return NULL;
  }

  ItemRec *results = get_item_recommendations(
      engine.similarity, engine.matrix->data, user_id, k, engine.matrix->rows,
      engine.matrix->cols, num_neighbors);

  if (!results) return NULL;

  // Convert to simple int array (item IDs only)
  int *item_ids = (int *)malloc((k + 1) * sizeof(int));
  if (!item_ids) {
    free(results);
    return NULL;
  }

  for (int i = 0; i < k; i++) {
    item_ids[i] = results[i].item_id;
  }
  item_ids[k] = -1;  // Sentinel value

  free(results);
  return item_ids;
}

/**
 * Get the similarity score between two users
 */
float rec_engine_get_similarity(int user_id_a, int user_id_b) {
  if (!engine.initialized || !engine.similarity) {
    return -1.0f;
  }

  if (user_id_a < 0 || user_id_a >= engine.matrix->rows || user_id_b < 0 ||
      user_id_b >= engine.matrix->rows) {
    return -1.0f;
  }

  return engine.similarity[user_id_a * engine.matrix->rows + user_id_b];
}

/**
 * Get number of users and items
 */
void rec_engine_get_dimensions(int *num_users, int *num_items) {
  if (engine.initialized && engine.matrix) {
    *num_users = engine.matrix->rows;
    *num_items = engine.matrix->cols;
  } else {
    *num_users = 0;
    *num_items = 0;
  }
}

/**
 * Free memory and cleanup
 */
void rec_engine_cleanup() {
  if (engine.similarity) {
    free(engine.similarity);
    engine.similarity = NULL;
  }
  if (engine.norms) {
    free(engine.norms);
    engine.norms = NULL;
  }
  if (engine.matrix) {
    free_matrix(engine.matrix);
    engine.matrix = NULL;
  }
  engine.initialized = 0;
}

/**
 * Free allocated arrays returned by recommendation functions
 */
void rec_engine_free_array(int *arr) {
  if (arr) {
    free(arr);
  }
}

/**
 * Print engine status
 */
void rec_engine_print_status() {
  printf("\n=== Recommendation Engine Status ===\n");
  if (engine.initialized) {
    printf("Status: INITIALIZED\n");
    printf("Users: %d\n", engine.matrix->rows);
    printf("Items: %d\n", engine.matrix->cols);
    printf("Similarity matrix: %.2f MB\n",
           (engine.matrix->rows * engine.matrix->rows * sizeof(float)) /
               (1024.0f * 1024.0f));
  } else {
    printf("Status: NOT INITIALIZED\n");
  }
  printf("=====================================\n\n");
}
