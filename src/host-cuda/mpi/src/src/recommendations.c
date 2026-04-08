#include "recommendations.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

static int partition(float *arr, int *indices, int low, int high) {
  float pivot = arr[indices[high]];
  int i = low - 1;

  for (int j = low; j < high; j++) {
    if (arr[indices[j]] > pivot) {
      i++;
      int tmp_idx = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp_idx;
    }
  }

  int tmp_idx = indices[i + 1];
  indices[i + 1] = indices[high];
  indices[high] = tmp_idx;

  return i + 1;
}

static void quickselect(float *arr, int *indices, int low, int high, int k) {
  if (low < high) {
    int pi = partition(arr, indices, low, high);

    if (pi < k) {
      quickselect(arr, indices, pi + 1, high, k);
    } else if (pi > k) {
      quickselect(arr, indices, low, pi - 1, k);
    }
  }
}

void quickselect_topk(float *arr, int n, int k, int *indices) {
  if (k > n) k = n;
  if (k <= 0) return;

  for (int i = 0; i < n; i++) {
    indices[i] = i;
  }

  quickselect(arr, indices, 0, n - 1, k - 1);

  for (int i = 0; i < k - 1; i++) {
    for (int j = i + 1; j < k; j++) {
      if (arr[indices[j]] > arr[indices[i]]) {
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
      }
    }
  }
}

UserRec *get_similar_users(float *similarity_matrix, int user_id, int k,
                           int num_users) {
  if (!similarity_matrix || user_id < 0 || user_id >= num_users || k <= 0) {
    return NULL;
  }

  if (k > num_users - 1) k = num_users - 1;

  UserRec *results = (UserRec *)malloc(k * sizeof(UserRec));
  if (!results) return NULL;

  float *sims = (float *)malloc(num_users * sizeof(float));
  if (!sims) {
    free(results);
    return NULL;
  }

  memcpy(sims, &similarity_matrix[user_id * num_users],
         num_users * sizeof(float));

  int *indices = (int *)malloc(num_users * sizeof(int));
  if (!indices) {
    free(sims);
    free(results);
    return NULL;
  }

  quickselect_topk(sims, num_users, k, indices);

  int result_idx = 0;
  for (int i = 0; i < k && result_idx < k; i++) {
    int idx = indices[i];
    if (idx != user_id) {
      results[result_idx].user_id = idx;
      results[result_idx].similarity_score = sims[idx];
      result_idx++;
    }
  }

  if (result_idx < k) {
    for (int i = k; i < num_users && result_idx < k; i++) {
      int idx = indices[i];
      if (idx != user_id) {
        results[result_idx].user_id = idx;
        results[result_idx].similarity_score = sims[idx];
        result_idx++;
      }
    }
  }

  free(sims);
  free(indices);

  return results;
}

ItemRec *get_item_recommendations(float *similarity_matrix,
                                  float *rating_matrix, int user_id, int k,
                                  int num_users, int num_items,
                                  int num_neighbors) {
  if (!similarity_matrix || !rating_matrix || user_id < 0 ||
      user_id >= num_users || k <= 0) {
    return NULL;
  }

  if (k > num_items) k = num_items;
  if (num_neighbors > num_users - 1) num_neighbors = num_users - 1;

  UserRec *similar_users =
      get_similar_users(similarity_matrix, user_id, num_neighbors, num_users);
  if (!similar_users) return NULL;

  float *predicted_ratings = (float *)calloc(num_items, sizeof(float));
  float *weight_sums = (float *)calloc(num_items, sizeof(float));
  if (!predicted_ratings || !weight_sums) {
    free(similar_users);
    free(predicted_ratings);
    free(weight_sums);
    return NULL;
  }

#pragma omp parallel for collapse(2)
  for (int n = 0; n < num_neighbors; n++) {
    for (int item = 0; item < num_items; item++) {
      int neighbor_id = similar_users[n].user_id;
      float similarity = similar_users[n].similarity_score;
      float rating = rating_matrix[neighbor_id * num_items + item];

      if (rating > 0.0f) {
#pragma omp atomic
        predicted_ratings[item] += similarity * rating;
#pragma omp atomic
        weight_sums[item] += similarity;
      }
    }
  }

#pragma omp parallel for schedule(static)
  for (int item = 0; item < num_items; item++) {
    if (weight_sums[item] > 0.0f) {
      predicted_ratings[item] /= weight_sums[item];
    }
  }

  ItemRec *results = (ItemRec *)malloc(k * sizeof(ItemRec));
  if (!results) {
    free(similar_users);
    free(predicted_ratings);
    free(weight_sums);
    return NULL;
  }

  int *indices = (int *)malloc(num_items * sizeof(int));
  if (!indices) {
    free(results);
    free(similar_users);
    free(predicted_ratings);
    free(weight_sums);
    return NULL;
  }

  quickselect_topk(predicted_ratings, num_items, k, indices);

  for (int i = 0; i < k; i++) {
    int idx = indices[i];
    results[i].item_id = idx;
    results[i].predicted_rating = predicted_ratings[idx];
  }

  free(indices);
  free(similar_users);
  free(predicted_ratings);
  free(weight_sums);

  return results;
}
