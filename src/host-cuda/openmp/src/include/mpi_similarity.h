#ifndef MPI_SIMILARITY_H
#define MPI_SIMILARITY_H

#include <mpi.h>

#include "matrix.h"

typedef struct {
  int rank;
  int size;
  int rows_per_process;
  int start_row;
  int end_row;
} MPIContext;

MPIContext *mpi_init_context();

void mpi_finalize_context(MPIContext *ctx);

float *mpi_compute_similarity(const Matrix *matrix, const float *norms,
                              MPIContext *ctx);

#endif
