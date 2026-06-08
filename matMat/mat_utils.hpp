#include <cstddef>

#ifndef MAT_UTILS_HPP
#define MAT_UTILS_HPP

void print_mat(const double *mat, const size_t rows, const size_t cols, const size_t ld);
               
void init_mat(double *mat, const size_t rows, const size_t cols, const size_t ld);

void matmatikj(const double *A, const double *B, double *C, 
               const size_t N, const size_t K, const size_t M, 
               const size_t ldA, const size_t ldB, const size_t ldC);

void matmatblock(const double *A, const double *B, double *C, 
                 const size_t N, const size_t K, const size_t M, 
                 const size_t ldA, const size_t ldB, const size_t ldC, 
                 const size_t dbA, const size_t dbB, const size_t dbC);
                 
#endif /* MAT_UTILS_HPP */
