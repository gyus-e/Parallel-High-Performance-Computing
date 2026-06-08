#include <cstddef>

#ifndef MAT_KERNELS_HPP
#define MAT_KERNELS_HPP

using matmatkernel_t = void (*)(
    const double *__restrict__ A, 
    const double *__restrict__ B,
    double *__restrict__ C, 
    const size_t N, const size_t K, const size_t M,
    const size_t ldA, const size_t ldB, const size_t ldC);

__global__ void hadamard(const double *__restrict__ A,
                         const double *__restrict__ B, 
                         double *__restrict__ C,
                         const size_t N, const size_t K, const size_t M,
                         const size_t ldA, const size_t ldB, const size_t ldC);

__global__ void matmat_coalesced(const double *__restrict__ A,
                                 const double *__restrict__ B,
                                 double *__restrict__ C, 
                                 const size_t N, const size_t K, const size_t M,
                                 const size_t ldA, const size_t ldB, const size_t ldC);

__global__ void matmat_uncoalesced(const double *__restrict__ A,
                                   const double *__restrict__ B,
                                   double *__restrict__ C, 
                                   const size_t N, const size_t K, const size_t M,
                                   const size_t ldA, const size_t ldB, const size_t ldC);

__global__ void matmat_tiled(const double *__restrict__ A,
                             const double *__restrict__ B,
                             double *__restrict__ C, 
                             const size_t N, const size_t K, const size_t M, 
                             const size_t ldA, const size_t ldB, const size_t ldC);

#endif /* MAT_KERNELS_HPP */
