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

# Sources compiled into the shared library (excludes MPI-specific and main)
LIB_SRCS = $(SRC_DIR)/data.c $(SRC_DIR)/file_reader.c $(SRC_DIR)/parser.c \
           $(SRC_DIR)/matrix.c $(SRC_DIR)/recommendations.c
LIB_TARGET = librec_engine.so

.PHONY: all mpi librec_engine.so clean run

all: mpi

mpi: $(TARGET_MPI)

$(TARGET_MPI): $(OBJS_MPI)
	@mkdir -p $(BIN_DIR)
	$(MPICC) $(OBJS_MPI) -o $(TARGET_MPI) $(MPI_LDFLAGS)
	@echo "✓ Built MPI version: $(TARGET_MPI)"

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(MPICC) $(MPI_CFLAGS) -c $< -o $@

# Build the shared library used by the Python ctypes wrappers.
# Output: src/host-cuda/mpi/src/librec_engine.so
# Install: sudo cp librec_engine.so /usr/local/lib && sudo ldconfig
librec_engine.so: $(LIB_SRCS)
	$(CC) -shared -fPIC -fopenmp -O3 -std=c11 \
	    -Iinclude \
	    $(LIB_SRCS) \
	    -o $(LIB_TARGET) \
	    -luuid -lm -fopenmp
	@echo "✓ Built shared library: $(LIB_TARGET)"
	@echo "  Install: sudo cp $(LIB_TARGET) /usr/local/lib && sudo ldconfig"

clean:
	@rm -f $(SRC_DIR)/*.o $(TARGET_MPI) $(LIB_TARGET)
	@echo "✓ Cleaned MPI build"

run: mpi
	@echo "\n=== Running MPI Version ==="
	mpirun -np 4 ./$(TARGET_MPI)
