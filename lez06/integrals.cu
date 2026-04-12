#include "integrals.hpp"
#include "trapezoid.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>

/**
Trapezoidal rule: the integral of f(x) from a to b is approximated by:
(h/2) * [f(x_0) + 2 * sum[i=1 to n-1](f(x_i)) + f(x_n)]
where:
h = (b - a) / n
x_i = a + i * h
*/

double integral_cpu(const double a, const double b, const double h,
                    const unsigned long n, const unsigned int nt, double *time) {
  double res = f(a) + f(b);
  double sum = 0;

  omp_set_num_threads(nt);

  double start = omp_get_wtime() * 1000;
  
  trap_cpu(a, n, h, sum);
  
  double end = omp_get_wtime() * 1000;
  if (time != nullptr) {
    *time = end - start;
  }

  res += 2 * sum;
  res *= h * 0.5;
  return res;
}

double integral_gpu_naive(const double a, const double b, const double h,
                    const unsigned long n, const unsigned int blockSize,
                    const unsigned int gridSize, double *time) {
  double res = f(a) + f(b);
  double *sum;
  cudaMallocManaged(&sum, sizeof(double));
  *sum = 0;

  dim3 dimBlock(blockSize);
  dim3 dimGrid(gridSize);

  double start = omp_get_wtime() * 1000;

  trap_gpu_naive<<<dimGrid, dimBlock>>>(a, n, h, sum);
  cudaDeviceSynchronize();

  double end = omp_get_wtime() * 1000;
  if (time != nullptr) {
    *time = end - start;
  }

  res += 2 * (*sum);
  res *= h * 0.5;
  cudaFree(sum);
  return res;
}

double integral_gpu_shared_mem_tree_sum(const double a, const double b, const double h,
                    const unsigned long n, const unsigned int blockSize,
                    const unsigned int gridSize, double *time) {
  double res = f(a) + f(b);
  double *sum;
  cudaMallocManaged(&sum, sizeof(double));
  *sum = 0;

  dim3 dimBlock(blockSize);
  dim3 dimGrid(gridSize);

  double start = omp_get_wtime() * 1000;

  trap_gpu_shared_mem_tree_sum<<<dimGrid, dimBlock>>>(a, n, h, sum);
  cudaDeviceSynchronize();

  double end = omp_get_wtime() * 1000;
  if (time != nullptr) {
    *time = end - start;
  }

  res += 2 * (*sum);
  res *= h * 0.5;
  cudaFree(sum);
  return res;
}

double integral_gpu_warp_shuffle_tree_sum(const double a, const double b, const double h,
                    const unsigned long n, const unsigned int blockSize,
                    const unsigned int gridSize, double *time) {
  double res = f(a) + f(b);
  double *sum;
  cudaMallocManaged(&sum, sizeof(double));

  dim3 dimBlock(blockSize);
  dim3 dimGrid(gridSize);

  double start = omp_get_wtime() * 1000;

  trap_gpu_warp_shuffle_tree_sum<<<dimGrid, dimBlock>>>(a, n, h, sum);
  cudaDeviceSynchronize();

  double end = omp_get_wtime() * 1000;  
  if (time != nullptr) {
    *time = end - start;
  }

  res += 2 * (*sum);
  res *= h * 0.5;
  cudaFree(sum);
  return res;                    
}
