
#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

double get_cur_time();

void print_mat(const double *mat, const unsigned int rows,
               const unsigned int cols, const unsigned int ld);

void init_mat(double *mat, const unsigned int rows, const unsigned int cols,
              const unsigned int ld);

#ifdef __cplusplus
}
#endif

#endif /* UTILS_H */