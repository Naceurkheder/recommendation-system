#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"
#include "recommendations.h"

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc < 3) {
    if (rank == 0)
      fprintf(stderr, "Usage: mpirun -np N %s <input.csv> <output.bin>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  double t0 = now_ms();

  int num_users = 0, num_items = 0;
  float *matrix_data = NULL;
  float *norms = NULL;

  if (rank == 0) {
    Matrix *m = load_matrix(argv[1]);
    if (!m) {
      fprintf(stderr, "[MPI Rank 0] Failed to load matrix from %s\n", argv[1]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    num_users = m->rows;
    num_items = m->cols;
    matrix_data = m->data;
    free(m);

    norms = (float *)malloc(num_users * sizeof(float));
    for (int i = 0; i < num_users; i++) {
      float sq = 0.0f;
      for (int j = 0; j < num_items; j++) {
        float v = matrix_data[i * num_items + j];
        sq += v * v;
      }
      norms[i] = sqrtf(sq);
    }
  }

  double t_load = now_ms();

  MPI_Bcast(&num_users, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&num_items, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    matrix_data = (float *)malloc((size_t)num_users * num_items * sizeof(float));
    norms       = (float *)malloc(num_users * sizeof(float));
    if (!matrix_data || !norms) {
      fprintf(stderr, "[MPI Rank %d] malloc failed\n", rank);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  MPI_Bcast(matrix_data, num_users * num_items, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(norms,       num_users,             MPI_FLOAT, 0, MPI_COMM_WORLD);

  double t_bcast = now_ms();

  int extra     = num_users % size;
  int start_row = rank * (num_users / size) + (rank < extra ? rank : extra);
  int local_rows = (num_users / size) + (rank < extra ? 1 : 0);

  float *local_sim = (float *)calloc((size_t)local_rows * num_users, sizeof(float));
  if (!local_sim) {
    fprintf(stderr, "[MPI Rank %d] calloc for local_sim failed\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

#pragma omp parallel for schedule(guided)
  for (int li = 0; li < local_rows; li++) {
    int gi = start_row + li;
    local_sim[li * num_users + gi] = 1.0f;

    if (norms[gi] <= 1e-10f) continue;

    const float *row_i = &matrix_data[gi * num_items];

    for (int j = 0; j < num_users; j++) {
      if (j == gi || norms[j] <= 1e-10f) continue;

      const float *row_j = &matrix_data[j * num_items];
      float dot = 0.0f;
      int k = 0;
      for (; k <= num_items - 4; k += 4) {
        dot += row_i[k] * row_j[k];
        dot += row_i[k + 1] * row_j[k + 1];
        dot += row_i[k + 2] * row_j[k + 2];
        dot += row_i[k + 3] * row_j[k + 3];
      }
      for (; k < num_items; k++)
        dot += row_i[k] * row_j[k];

      float sim = dot / (norms[gi] * norms[j]);
      local_sim[li * num_users + j] = fminf(fmaxf(sim, -1.0f), 1.0f);
    }
  }

  int *recv_counts = NULL;
  int *displs      = NULL;
  float *global_sim = NULL;

  if (rank == 0) {
    recv_counts = (int *)malloc(size * sizeof(int));
    displs      = (int *)malloc(size * sizeof(int));
    global_sim  = (float *)malloc((size_t)num_users * num_users * sizeof(float));
    if (!global_sim) {
      fprintf(stderr, "[MPI Rank 0] malloc for global_sim failed\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int disp = 0;
    for (int p = 0; p < size; p++) {
      int p_rows = num_users / size + (p < extra ? 1 : 0);
      recv_counts[p] = p_rows * num_users;
      displs[p]      = disp;
      disp          += p_rows * num_users;
    }
  }

  MPI_Gatherv(local_sim, local_rows * num_users, MPI_FLOAT,
              global_sim, recv_counts, displs, MPI_FLOAT,
              0, MPI_COMM_WORLD);

  double t_sim = now_ms();

  if (rank == 0) {
    FILE *fout = fopen(argv[2], "wb");
    if (!fout) {
      fprintf(stderr, "[MPI Rank 0] Cannot write output to %s\n", argv[2]);
    } else {
      fwrite(&num_users, sizeof(int), 1, fout);
      fwrite(&num_items, sizeof(int), 1, fout);
      fwrite(global_sim, sizeof(float), (size_t)num_users * num_users, fout);
      fclose(fout);
    }

    double t_end = now_ms();

    printf("{\"num_users\":%d,\"num_items\":%d,\"nprocs\":%d,"
           "\"load_ms\":%.1f,\"bcast_ms\":%.1f,"
           "\"similarity_ms\":%.1f,\"total_ms\":%.1f}\n",
           num_users, num_items, size,
           t_load  - t0,
           t_bcast - t_load,
           t_sim   - t_bcast,
           t_end   - t0);
    fflush(stdout);

    free(global_sim);
    free(recv_counts);
    free(displs);
  }

  free(local_sim);
  free(norms);
  free(matrix_data);

  MPI_Finalize();
  return 0;
}
