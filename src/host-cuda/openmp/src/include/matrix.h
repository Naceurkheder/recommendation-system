#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>

#include "data.h"

typedef struct {
  float *data;
  int rows;
  int cols;
} Matrix;

Matrix *load_matrix(const char *filename);

void free_matrix(Matrix *matrix);

void print_matrix(const Matrix *matrix);

float *compute_norms(const Matrix *matrix);

float *compute_similarity(const Matrix *matrix, const float *norms);

float *compute_similarity_omp(const Matrix *matrix, const float *norms);

#endif
