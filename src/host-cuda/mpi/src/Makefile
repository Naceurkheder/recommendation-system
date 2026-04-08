MPICC = mpicc

MPI_CFLAGS = -Wall -Wextra -std=c11 -Iinclude -fopenmp -O3 -march=native -ffast-math
MPI_LDFLAGS = -luuid -lm -fopenmp

SRC_DIR = src
BIN_DIR = bin

SRCS_MPI = $(SRC_DIR)/data.c $(SRC_DIR)/file_reader.c $(SRC_DIR)/parser.c \
           $(SRC_DIR)/mpi_similarity.c $(SRC_DIR)/recommendations.c $(SRC_DIR)/main.c
OBJS_MPI = $(SRCS_MPI:.c=.o)
TARGET_MPI = $(BIN_DIR)/similarity_mpi

.PHONY: all mpi clean run

all: mpi

mpi: $(TARGET_MPI)

$(TARGET_MPI): $(OBJS_MPI)
	@mkdir -p $(BIN_DIR)
	$(MPICC) $(OBJS_MPI) -o $(TARGET_MPI) $(MPI_LDFLAGS)
	@echo "✓ Built MPI version: $(TARGET_MPI)"

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(MPICC) $(MPI_CFLAGS) -c $< -o $@

clean:
	@rm -f $(SRC_DIR)/*.o $(TARGET_MPI)
	@echo "✓ Cleaned MPI build"

run: mpi
	@echo "\n=== Running MPI Version ==="
	mpirun -np 4 ./$(TARGET_MPI)
