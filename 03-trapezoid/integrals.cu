#include "integrals.hpp"
#include "trapezoid.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>

/*
Note for shared memory:
We should get the best performance if sdata is declared in the kernel as
  extern __shared__ double sdata[]
We want the shared memory size to be equal to the block size
  shMem = blockSize * sizeof(double)
And the kernel must be called with 3 arguments in the triple angle brackets
  <<<Nblocks, blockSize, shMem>>>
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



double integral_gpu_naive(const double a, const double b, const double h, const unsigned long n, 
                          const unsigned int blockSize, const unsigned int gridSize, double *time) {
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



double integral_gpu_shared_mem_tree_sum(const double a, const double b, const double h, const unsigned long n, 
                                        const unsigned int blockSize, const unsigned int gridSize, double *time) {
  double res = f(a) + f(b);
  double *sum;
  cudaMallocManaged(&sum, sizeof(double));
  *sum = 0;

  const unsigned int sharedMemSize = blockSize * sizeof(double);
  dim3 dimBlock(blockSize);
  dim3 dimGrid(gridSize);

  double start = omp_get_wtime() * 1000;

  trap_gpu_shared_mem_tree_sum<<<dimGrid, dimBlock, sharedMemSize>>>(a, n, h, sharedMemSize, sum);
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



double integral_gpu_warp_shuffle_tree_sum(const double a, const double b, const double h, const unsigned long n, 
                                          const unsigned int blockSize, const unsigned int gridSize, double *time) {
  double res = f(a) + f(b);
  double *sum;
  cudaMallocManaged(&sum, sizeof(double));
  *sum = 0;

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



double integral_gpu_shared_mem_dissemination_sum(const double a, const double b, const double h, const unsigned long n, 
                                                 const unsigned int blockSize, const unsigned int gridSize, double *time) {
  double res = f(a) + f(b);
  double *sum;
  cudaMallocManaged(&sum, sizeof(double));
  *sum = 0;
  
  const unsigned int warpsPerBlock = (blockSize + WARP_SIZE - 1) / WARP_SIZE;
  const unsigned int sharedMemSize = MAX_BLKSZ * sizeof(double);
  dim3 dimBlock(blockSize);
  dim3 dimGrid(gridSize);

  double start = omp_get_wtime() * 1000;

  trap_gpu_shared_mem_dissemination_sum<<<dimGrid, dimBlock, sharedMemSize>>>(a, n, h, sharedMemSize, sum);
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



double integral_gpu_warp_shuffle_dissemination_sum(const double a, const double b, const double h, const unsigned long n, 
                                                   const unsigned int blockSize, const unsigned int gridSize, double *time) {
  double res = f(a) + f(b);
  double *sum;
  cudaMallocManaged(&sum, sizeof(double));
  *sum = 0;

  dim3 dimBlock(blockSize);
  dim3 dimGrid(gridSize);

  double start = omp_get_wtime() * 1000;

  trap_gpu_warp_shuffle_dissemination_sum<<<dimGrid, dimBlock>>>(a, n, h, sum);
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
