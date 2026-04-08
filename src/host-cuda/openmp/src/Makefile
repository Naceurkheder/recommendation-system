CC = gcc
MPICC = mpicc
NVCC = nvcc
HAVE_NVCC = $(shell which nvcc > /dev/null 2>&1 && echo 1 || echo 0)

CFLAGS = -Wall -Wextra -std=c11 -Iinclude -fopenmp -O3 -march=native -ffast-math
LDFLAGS = -luuid -lm -fopenmp

MPI_CFLAGS = -Wall -Wextra -std=c11 -Iinclude -fopenmp -O3 -march=native -ffast-math
MPI_LDFLAGS = -luuid -lm -fopenmp

CUDA_CFLAGS = -I include -O3 --use_fast_math
CUDA_LDFLAGS = -lcuda -lcudart -lm -luuid

SRC_DIR = src
BIN_DIR = bin

SRCS_OPENMP = $(SRC_DIR)/data.c $(SRC_DIR)/file_reader.c $(SRC_DIR)/parser.c \
              $(SRC_DIR)/similarity.c $(SRC_DIR)/recommendations.c $(SRC_DIR)/main_benchmark.c
OBJS_OPENMP = $(SRCS_OPENMP:.c=.o)
TARGET_OPENMP = $(BIN_DIR)/similarity_openmp

SRCS_MPI = $(SRC_DIR)/data.c $(SRC_DIR)/file_reader.c $(SRC_DIR)/parser.c \
           $(SRC_DIR)/mpi_similarity.c $(SRC_DIR)/recommendations.c $(SRC_DIR)/main_mpi.c
OBJS_MPI = $(SRCS_MPI:.c=.o)
TARGET_MPI = $(BIN_DIR)/similarity_mpi

SRCS_CUDA_C = $(SRC_DIR)/data.c $(SRC_DIR)/file_reader.c $(SRC_DIR)/parser.c \
              $(SRC_DIR)/recommendations.c $(SRC_DIR)/main_cuda.c
OBJS_CUDA_C = $(SRCS_CUDA_C:.c=.o)
SRCS_CUDA_CU = $(SRC_DIR)/cuda_similarity.cu
OBJS_CUDA_CU = $(SRCS_CUDA_CU:.cu=.o)
OBJS_CUDA = $(OBJS_CUDA_C) $(OBJS_CUDA_CU)
TARGET_CUDA = $(BIN_DIR)/similarity_cuda

.PHONY: all openmp mpi cuda clean run run_openmp run_mpi run_cuda

all: openmp mpi $(if $(filter 1,$(HAVE_NVCC)),cuda,)

openmp: $(TARGET_OPENMP)

mpi: $(TARGET_MPI)

cuda: $(if $(filter 1,$(HAVE_NVCC)),$(TARGET_CUDA),)
	$(if $(filter 0,$(HAVE_NVCC)),@echo "nvcc not found - skipping CUDA build",)

# OpenMP target
$(TARGET_OPENMP): $(OBJS_OPENMP)
	@mkdir -p $(BIN_DIR)
	$(CC) $(OBJS_OPENMP) -o $(TARGET_OPENMP) $(LDFLAGS)
	@echo "✓ Built OpenMP version: $(TARGET_OPENMP)"

# MPI target
$(TARGET_MPI): $(OBJS_MPI)
	@mkdir -p $(BIN_DIR)
	$(MPICC) $(OBJS_MPI) -o $(TARGET_MPI) $(MPI_LDFLAGS)
	@echo "✓ Built MPI version: $(TARGET_MPI)"

# CUDA target
$(TARGET_CUDA): $(OBJS_CUDA)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(OBJS_CUDA) -o $(TARGET_CUDA) $(CUDA_LDFLAGS)
	@echo "✓ Built CUDA version: $(TARGET_CUDA)"

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/mpi_similarity.o: $(SRC_DIR)/mpi_similarity.c
	$(MPICC) $(MPI_CFLAGS) -c $< -o $@

$(SRC_DIR)/main_mpi.o: $(SRC_DIR)/main_mpi.c
	$(MPICC) $(MPI_CFLAGS) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CUDA_CFLAGS) -c $< -o $@

clean:
	@rm -f $(SRC_DIR)/*.o $(TARGET_OPENMP) $(TARGET_MPI) $(TARGET_CUDA)
	@echo "✓ Cleaned all builds"

run: run_openmp

run_openmp: openmp
	@echo "\n=== Running OpenMP Version ==="
	./$(TARGET_OPENMP)

run_mpi: mpi
	@echo "\n=== Running MPI Version ==="
	mpirun -np 4 ./$(TARGET_MPI)

run_cuda: cuda
	@echo "\n=== Running CUDA Version ==="
	./$(TARGET_CUDA)

run_all: openmp mpi cuda
	@echo "\n=== Running OpenMP Version ==="
	./$(TARGET_OPENMP)
	@echo "\n=== Running MPI Version ==="
	mpirun -np 4 ./$(TARGET_MPI)
	@echo "\n=== Running CUDA Version ==="
	./$(TARGET_CUDA)
