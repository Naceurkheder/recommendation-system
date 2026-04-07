#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <string.h>
#include <uuid/uuid.h>

#include "data.h"
#include "file_reader.h"
#include "parser.h"

int parse_line(char *line, struct data *entry) {
  char *saveptr;

  char *token = strtok_r(line, ",", &saveptr);
  if (!token) return 0;

  entry->user_id = atoi(token);

  token = strtok_r(NULL, ",", &saveptr);
  if (!token) return 0;

  if (uuid_parse(token, entry->product_id) != 0) return 0;

  token = strtok_r(NULL, ",", &saveptr);
  if (!token) return 0;

  entry->rating = atof(token);
  return 1;
}
