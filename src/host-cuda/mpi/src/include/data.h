#ifndef DATA_H
#define DATA_H

#include <uuid/uuid.h>

struct data {
  int user_id;
  uuid_t product_id;
  float rating;
};

void print_data(const struct data *entry);

#endif
