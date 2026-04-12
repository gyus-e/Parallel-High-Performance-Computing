#include "trapezoid.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>

/**
GPU shared memory, tree-structured sum: 
pair up the threads so that half of the “active” threads add their partial sum to their partner’s partial sum. 
*/
__device__ void shared_mem_tree_sum(double *sdata, const unsigned int sdata_len) {
  const unsigned int tid = threadIdx.x;
  for (unsigned int stride = (blockDim.x >> 1); stride > 0; stride >>= 1) {
    if (tid < stride && tid + stride < sdata_len) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
}

/**
GPU warp shuffle, tree-structured sum: 
Warp shuffle instructions allow threads within a warp to read variables stored in another thread’s register in the warp.
This allows us to compute the global sum in registers, which are faster than shared memory.
(Only available in devices with compute capability >= 3.0)
*/
__device__ double warp_shuffle_tree_sum(double val) {
  for (unsigned int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  return val;
}


void trap_cpu(const double a, const unsigned long n, const double h,
              double &res) {
  double sum = res;
  #pragma omp parallel for reduction(+:sum)
  for (int i = 1; i < n; i++) {
    double x_i = a + i * h;
    sum += f(x_i);
  }
  res = sum;
}

__global__ void trap_gpu_naive(const double a, const unsigned long n,
                               const double h, double *res) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > 0 && i < n) {
    double x_i = a + i * h;
    atomicAdd(res, f(x_i));
  }
}

__global__ void trap_gpu_shared_mem_tree_sum(const double a, const unsigned long n,
                            const double h, double *res) {
  __shared__ double sdata[SHARED_MEM_SIZE];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid <= SHARED_MEM_SIZE && i < n && i > 0) {
    double x_i = a + i * h;
    sdata[tid] = f(x_i);
  }
  __syncthreads();

  shared_mem_tree_sum(sdata, SHARED_MEM_SIZE);
  if (tid == 0) {
    atomicAdd(res, sdata[0]);
  }
}

__global__ void trap_gpu_warp_shuffle_tree_sum(const double a, const unsigned long n,
                            const double h, double *res) {
  const unsigned int tid = threadIdx.x;
  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int lane = tid % warpSize;

  double val = 0;
  if (i > 0 && i < n) {
    double x_i = a + i * h;
    val = f(x_i);
  }
  __syncwarp(FULL_MASK);

  double sum = warp_shuffle_tree_sum(val);
  if (lane == 0) {
    atomicAdd(res, sum);
  }
}