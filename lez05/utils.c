#include "utils.h"
#include <stdio.h>
#include <time.h>

double get_cur_time() {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void print_mat(const double *mat, const unsigned int rows,
               const unsigned int cols, const unsigned int ld) {
  unsigned int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      printf("%.1f \t", mat[i * ld + j]);
    }
    printf("\n");
  }
}

void init_mat(double *mat, const unsigned int rows, const unsigned int cols,
              const unsigned int ld) {
  unsigned int i, j;
  double x = 0.0;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      mat[i * ld + j] = ++x;
    }
  }
}
