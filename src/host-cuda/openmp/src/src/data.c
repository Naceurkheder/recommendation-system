#include <stdio.h>

#include "data.h"

void print_data(const struct data *entry) {
  printf("User ID: %d, Product ID: ", entry->user_id);
  for (int i = 0; i < 16; i++) {
    printf("%02x", entry->product_id[i]);
  }
  printf(", Rating: %.2f\n", entry->rating);
}
