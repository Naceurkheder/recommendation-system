#ifndef CUDA_SIMILARITY_H
#define CUDA_SIMILARITY_H

typedef struct {
  float *d_matrix_data;
  float *d_norms;
  float *d_similarity;
  int num_users;
  int num_items;
} CUDAContext;

typedef struct {
  float *data;
  int rows;
  int cols;
} Matrix;

CUDAContext *cuda_init(const Matrix *matrix, const float *norms);
void cuda_free(CUDAContext *ctx);
float *cuda_compute_similarity(CUDAContext *ctx);

#endif
