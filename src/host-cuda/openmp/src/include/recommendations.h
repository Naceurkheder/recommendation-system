#ifndef RECOMMENDATIONS_H
#define RECOMMENDATIONS_H

#include <stdint.h>

typedef struct {
  int user_id;
  float similarity_score;
} UserRec;

typedef struct {
  int item_id;
  float predicted_rating;
} ItemRec;

/**
 * Get top-k most similar users for a given user
 * @param similarity_matrix: Precomputed similarity matrix (num_users x
 * num_users)
 * @param user_id: User ID to get recommendations for
 * @param k: Number of recommendations
 * @param num_users: Total number of users
 * @return Array of k most similar users (must be freed)
 */
UserRec *get_similar_users(float *similarity_matrix, int user_id, int k,
                           int num_users);

/**
 * Get top-k item recommendations for a user based on similar users
 * @param similarity_matrix: Precomputed similarity matrix
 * @param rating_matrix: User-item rating matrix (num_users x num_items)
 * @param user_id: User ID
 * @param k: Number of recommendations
 * @param num_users: Total number of users
 * @param num_items: Total number of items
 * @param num_neighbors: Number of similar users to use for prediction
 * @return Array of k recommended items (must be freed)
 */
ItemRec *get_item_recommendations(float *similarity_matrix,
                                  float *rating_matrix, int user_id, int k,
                                  int num_users, int num_items,
                                  int num_neighbors);

/**
 * Efficient quickselect for finding top-k elements
 */
void quickselect_topk(float *arr, int n, int k, int *indices);

#endif
