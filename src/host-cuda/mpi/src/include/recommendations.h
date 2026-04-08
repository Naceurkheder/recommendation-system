#ifndef RECOMMENDATIONS_H
#define RECOMMENDATIONS_H

typedef struct {
  int user_id;
  float similarity_score;
} UserRec;

typedef struct {
  int item_id;
  float predicted_rating;
} ItemRec;

UserRec *get_similar_users(float *similarity_matrix, int user_id, int k,
                           int num_users);

ItemRec *get_item_recommendations(float *similarity_matrix,
                                  float *rating_matrix, int user_id, int k,
                                  int num_users, int num_items,
                                  int num_neighbors);

void quickselect_topk(float *arr, int n, int k, int *indices);

#endif
