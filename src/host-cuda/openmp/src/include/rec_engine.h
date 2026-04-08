#ifndef REC_ENGINE_H
#define REC_ENGINE_H

/**
 * Recommendation Engine C Interface
 *
 * This module provides a high-level API for building and using
 * a collaborative filtering recommendation engine. It handles:
 * - Loading user-item preference matrices from CSV
 * - Computing L2 norms for normalization
 * - Computing cosine similarity between users
 * - Generating top-k recommendations
 * - Integration with Python via ctypes
 */

/**
 * Initialize the recommendation engine
 * Must be called once before any recommendations
 *
 * @param csv_filename Path to CSV file with columns: user_id, product_id,
 * rating
 * @return 1 on success, 0 on failure
 */
int rec_engine_init(const char *csv_filename);

/**
 * Get similar users for a given user
 *
 * @param user_id Target user ID
 * @param k Number of similar users to return
 * @return Array of user IDs ending with -1 sentinel (caller must free)
 */
int *rec_engine_get_similar_users(int user_id, int k);

/**
 * Get item recommendations for a user
 * Based on collaborative filtering using similar users
 *
 * @param user_id Target user ID
 * @param k Number of recommendations
 * @param num_neighbors Number of similar users to use in prediction
 * @return Array of item IDs ending with -1 sentinel (caller must free)
 */
int *rec_engine_get_item_recommendations(int user_id, int k, int num_neighbors);

/**
 * Get similarity score between two users
 *
 * @param user_id_a First user
 * @param user_id_b Second user
 * @return Similarity score in [-1, 1] range (cosine similarity)
 */
float rec_engine_get_similarity(int user_id_a, int user_id_b);

/**
 * Get matrix dimensions
 *
 * @param num_users Output: number of users
 * @param num_items Output: number of items
 */
void rec_engine_get_dimensions(int *num_users, int *num_items);

/**
 * Free an array returned by recommendation functions
 */
void rec_engine_free_array(int *arr);

/**
 * Cleanup and free all engine resources
 */
void rec_engine_cleanup();

/**
 * Print engine status information
 */
void rec_engine_print_status();

#endif
