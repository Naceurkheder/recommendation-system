#include "file_reader.h"

#include <stdio.h>

FILE *open_file(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    perror("Unable to open file");
    return NULL;
  }
  return file;
}
