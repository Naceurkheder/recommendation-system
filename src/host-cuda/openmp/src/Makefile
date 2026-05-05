CC = gcc

CFLAGS = -Wall -Wextra -std=c11 -Iinclude -fopenmp -O3 -march=native -ffast-math
LDFLAGS = -luuid -lm -fopenmp

SRC_DIR = src
BIN_DIR = bin

SRCS = $(SRC_DIR)/data.c $(SRC_DIR)/file_reader.c $(SRC_DIR)/parser.c \
       $(SRC_DIR)/matrix.c $(SRC_DIR)/recommendations.c $(SRC_DIR)/main.c
OBJS = $(SRCS:.c=.o)
TARGET = $(BIN_DIR)/similarity_openmp

LIB_SRCS = $(SRC_DIR)/data.c $(SRC_DIR)/file_reader.c $(SRC_DIR)/parser.c \
            $(SRC_DIR)/matrix.c $(SRC_DIR)/recommendations.c
LIB_TARGET = librec_engine_openmp.so

.PHONY: all librec_engine_openmp.so clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

librec_engine_openmp.so: $(LIB_SRCS)
	$(CC) -shared -fPIC -fopenmp -O3 -std=c11 \
	    -Iinclude \
	    $(LIB_SRCS) \
	    -o $(LIB_TARGET) \
	    -luuid -lm -fopenmp

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@rm -f $(SRC_DIR)/*.o $(TARGET) $(LIB_TARGET)

run: $(TARGET)
	./$(TARGET)
