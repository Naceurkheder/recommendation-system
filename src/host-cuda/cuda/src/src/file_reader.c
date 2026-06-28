#include "file_reader.h"

#include <stdio.h>
#include <stdlib.h>

FILE *open_file(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    perror("Unable to open file");
    exit(EXIT_FAILURE);
  }
  return file;
}
