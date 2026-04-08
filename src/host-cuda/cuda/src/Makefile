CC = gcc
NVCC = nvcc
HAVE_NVCC = $(shell which nvcc > /dev/null 2>&1 && echo 1 || echo 0)

CC_CFLAGS = -Wall -Wextra -std=c11 -Iinclude -O3 -march=native -ffast-math
CC_LDFLAGS = -luuid -lm

CUDA_CFLAGS = -Iinclude -O3 --use_fast_math
CUDA_LDFLAGS = -lcuda -lcudart -lm -luuid

SRC_DIR = src
BIN_DIR = bin

SRCS_CUDA_C = $(SRC_DIR)/data.c $(SRC_DIR)/file_reader.c $(SRC_DIR)/parser.c \
              $(SRC_DIR)/recommendations.c $(SRC_DIR)/main.c
OBJS_CUDA_C = $(SRCS_CUDA_C:.c=.o)
SRCS_CUDA_CU = $(SRC_DIR)/cuda_similarity.cu
OBJS_CUDA_CU = $(SRCS_CUDA_CU:.cu=.o)
OBJS_CUDA = $(OBJS_CUDA_C) $(OBJS_CUDA_CU)
TARGET_CUDA = $(BIN_DIR)/similarity_cuda

.PHONY: all cuda clean run

all: cuda

cuda: $(if $(filter 1,$(HAVE_NVCC)),$(TARGET_CUDA),)
	$(if $(filter 0,$(HAVE_NVCC)),@echo "nvcc not found - skipping CUDA build",)

$(TARGET_CUDA): $(OBJS_CUDA)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(OBJS_CUDA) -o $(TARGET_CUDA) $(CUDA_LDFLAGS)
	@echo "✓ Built CUDA version: $(TARGET_CUDA)"

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CC_CFLAGS) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CUDA_CFLAGS) -c $< -o $@

clean:
	@rm -f $(SRC_DIR)/*.o $(TARGET_CUDA)
	@echo "✓ Cleaned CUDA build"

run: cuda
	@echo "\n=== Running CUDA Version ==="
	./$(TARGET_CUDA)
