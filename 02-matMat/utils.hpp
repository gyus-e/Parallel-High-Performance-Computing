
#ifndef UTILS_HPP
#define UTILS_HPP

double get_cur_time();

void print_mat(const double *mat, const unsigned int rows,
               const unsigned int cols, const unsigned int ld);

void init_mat(double *mat, const unsigned int rows, const unsigned int cols,
              const unsigned int ld);

#endif /* UTILS_H */