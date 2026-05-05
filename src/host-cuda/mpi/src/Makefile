MPICC = mpicc
CC    = gcc

MPI_CFLAGS = -Wall -Wextra -std=c11 -Iinclude -fopenmp -O3 -march=native -ffast-math
MPI_LDFLAGS = -luuid -lm -fopenmp

SRC_DIR = src
BIN_DIR = bin

SRCS_MPI = $(SRC_DIR)/data.c $(SRC_DIR)/file_reader.c $(SRC_DIR)/parser.c \
           $(SRC_DIR)/mpi_similarity.c $(SRC_DIR)/recommendations.c $(SRC_DIR)/main.c
OBJS_MPI = $(SRCS_MPI:.c=.o)
TARGET_MPI = $(BIN_DIR)/similarity_mpi

LIB_SRCS = $(SRC_DIR)/data.c $(SRC_DIR)/file_reader.c $(SRC_DIR)/parser.c \
           $(SRC_DIR)/matrix.c $(SRC_DIR)/recommendations.c
LIB_TARGET = librec_engine.so

BENCH_SRCS = $(SRC_DIR)/data.c $(SRC_DIR)/file_reader.c $(SRC_DIR)/parser.c \
             $(SRC_DIR)/matrix.c $(SRC_DIR)/recommendations.c \
             $(SRC_DIR)/mpi_bench_main.c
TARGET_BENCH = $(BIN_DIR)/similarity_mpi_bench

.PHONY: all mpi bench librec_engine.so clean run

all: mpi bench

mpi: $(TARGET_MPI)

$(TARGET_MPI): $(OBJS_MPI)
	@mkdir -p $(BIN_DIR)
	$(MPICC) $(OBJS_MPI) -o $(TARGET_MPI) $(MPI_LDFLAGS)

bench: $(TARGET_BENCH)

$(TARGET_BENCH): $(BENCH_SRCS)
	@mkdir -p $(BIN_DIR)
	$(MPICC) $(MPI_CFLAGS) $(BENCH_SRCS) -o $(TARGET_BENCH) $(MPI_LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(MPICC) $(MPI_CFLAGS) -c $< -o $@

librec_engine.so: $(LIB_SRCS)
	$(CC) -shared -fPIC -fopenmp -O3 -std=c11 \
	    -Iinclude \
	    $(LIB_SRCS) \
	    -o $(LIB_TARGET) \
	    -luuid -lm -fopenmp

clean:
	@rm -f $(SRC_DIR)/*.o $(TARGET_MPI) $(TARGET_BENCH) $(LIB_TARGET)

run: mpi
	mpirun -np 4 ./$(TARGET_MPI)
