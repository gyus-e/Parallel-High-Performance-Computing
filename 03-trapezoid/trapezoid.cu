#include "trapezoid.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>

/**
---------------------------------------------------------------------------------
TREE-STRUCTURED SUM
---------------------------------------------------------------------------------
Pair up the threads so that half of the “active” threads add their partial sum to their partner’s partial sum.
Note: __syncthreads() only synchronizes threads within the same block.
Note: sdata should be blockSize elements long
Note: the number of threads in the block should be a power of 2
*/
__device__ double shared_mem_tree_sum(double sdata[], const unsigned int sdataLen, const unsigned int block_tid, const unsigned int blockSize) {
  for (unsigned int stride = (blockSize >> 1); stride > 0; stride >>= 1) {
    if (block_tid < stride && block_tid + stride < sdataLen) {
      sdata[block_tid] += sdata[block_tid + stride];
    }
    __syncthreads();
  }
  return sdata[0];
}



/**
Warp shuffle instructions allow threads within a warp to read variables stored in another thread’s register in the warp. 
This allows us to compute the global sum in registers, which are faster than shared memory. 
Threads in a warp execute in lockstep, so synchronization is not needed.
(Only available in devices with compute capability >= 3.0)
Note: this only works for threads in the same warp
*/
__device__ double warp_shuffle_tree_sum(double val) {
  for (unsigned int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  return val;
}



/**
---------------------------------------------------------------------------------
DISSEMINATION SUM
---------------------------------------------------------------------------------
Within a warp, every thread reads a value from another thread in each step. 
At the end, all the threads in the warp have the same partial sum.
On each step, shift 'val' down by 'offset' lanes and add it to the local 'val’.
Make sure 0 <= source <= warpSize.
Note: this only works for threads in the same warp
*/
__device__ double shared_mem_dissemination_sum(double sdata[], const unsigned int sdataLen, const unsigned int lane, const unsigned int blockSize) {
  for (unsigned int offset = (blockSize >> 1); offset > 0; offset >>= 1) {
    const unsigned int source = (lane + offset) % blockSize;
    if (source < sdataLen && lane < sdataLen) { 
      sdata[lane] += sdata[source];
    }
    __syncwarp(FULL_MASK);
  }
  return sdata[lane];
}



/**
Warps can use warp shuffles to compute the warp sum.
To get the result in all the threads in the warp, 
we can use an alternative warp shuffle function: 
__shfl_xor_sync
Note: this only works for threads in the same warp
*/
__device__ double warp_shuffle_dissemination_sum(double val) {  
  for (unsigned int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(FULL_MASK, val, offset);
  }
  return val;
}



/**
---------------------------------------------------------------------------------
TRAPEZOIDAL RULE KERNELS
---------------------------------------------------------------------------------
The integral of f(x) from a to b is approximated by:
(h/2) * [f(x_0) + 2 * sum[i=1 to n-1](f(x_i)) + f(x_n)]
where:
h = (b - a) / n
x_i = a + i * h
*/
void trap_cpu(const double a, 
              const unsigned long n, 
              const double h,
              double &res) {
  double sum = res;
  #pragma omp parallel for reduction(+:sum)
  for (int i = 1; i < n; i++) {
    double x_i = a + i * h;
    sum += f(x_i);
  }
  res = sum;
}



__global__ void trap_gpu_naive(const double a, 
                               const unsigned long n,
                               const double h, 
                               double *res) {
  const unsigned int glob_tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (glob_tid > 0 && glob_tid < n) {
    double x_i = a + glob_tid * h;
    atomicAdd(res, f(x_i));
  }
}



__global__ void trap_gpu_shared_mem_tree_sum(const double a,
                                             const unsigned long n,
                                             const double h, 
                                             const unsigned int sdataLen,
                                             double *res) {
  extern __shared__ double sdata[];
  const unsigned int glob_tid = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int block_tid = threadIdx.x;

  sdata[block_tid] = 0;
  if (glob_tid < n && glob_tid > 0 && block_tid < sdataLen) {
    double x_i = a + glob_tid * h;
    sdata[block_tid] = f(x_i);
  }
  __syncthreads();

  double sum = shared_mem_tree_sum(sdata, sdataLen, block_tid, blockDim.x);
  if (block_tid == 0) {
    atomicAdd(res, sum);
  }
}



__global__ void trap_gpu_warp_shuffle_tree_sum(const double a,
                                               const unsigned long n,
                                               const double h, 
                                               double *res) {
  const unsigned int glob_tid = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int block_tid = threadIdx.x;
  const unsigned int lane = block_tid % warpSize;

  double val = 0;
  if (glob_tid > 0 && glob_tid < n) {
    double x_i = a + glob_tid * h;
    val = f(x_i);
  }

  double sum = warp_shuffle_tree_sum(val);
  if (lane == 0) {
    atomicAdd(res, sum);
  }
}



__global__ void trap_gpu_shared_mem_dissemination_sum(const double a,
                                                      const unsigned long n,
                                                      const double h,
                                                      double *res) {
  __shared__ double thread_calcs[MAX_BLKSZ];
  __shared__ double warp_sum_arr[WARP_SIZE];
  const unsigned int glob_tid = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int block_tid = threadIdx.x;
  const unsigned int lane = block_tid % warpSize;
  const unsigned int warp = block_tid / warpSize;
  // const unsigned int warpsPerBlock = (blockDim.x / WARP_SIZE) + ((blockDim.x % WARP_SIZE) != 0);
  // Nota che MAX_BLKSZ = 1024 e WARP_SIZE = 32, di conseguenza warpsPerBlock <= WARP_SIZE
  // Ovviamente warp <= warpsPerBlock è sempre verificato, e quindi anche warp <= WARP_SIZE

  double *shared_vals = &thread_calcs[warp * warpSize];

  double val = 0.0;
  if (glob_tid > 0 && glob_tid < n) {
    double x_i = a + glob_tid * h;
    val = f(x_i);
  }
  shared_vals[lane] = val; // Equivalente a thread_calcs[block_tid] = val
  warp_sum_arr[lane] = 0.0;
  __syncthreads();

  // double sum = shared_mem_tree_sum(shared_vals, warpSize, lane, warpSize);
  double sum = shared_mem_dissemination_sum(shared_vals, warpSize, lane, warpSize);
  // double sum = warp_shuffle_tree_sum(val);
  // double sum = warp_shuffle_dissemination_sum(val);
  __syncthreads();

  if (lane == 0 && warp <= WARP_SIZE) { 
    warp_sum_arr[warp] = sum; 
  }
  __syncthreads();

  if (warp == 0) {
    // sum = shared_mem_tree_sum(warp_sum_arr, warpSize, lane, warpSize);
    sum = shared_mem_dissemination_sum(warp_sum_arr, warpSize, lane, warpSize);
    // sum = warp_shuffle_tree_sum(warp_sum_arr[lane]);
    // sum = warp_shuffle_dissemination_sum(warp_sum_arr[lane]);
    __syncthreads();

    if (lane == 0) {
      atomicAdd(res, sum);
    }
  }
}



__global__ void trap_gpu_warp_shuffle_dissemination_sum(const double a,
                                           const unsigned long n,
                                           const double h,
                                           double *res) {
  const unsigned int glob_tid = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int block_tid = threadIdx.x;
  const unsigned int lane = block_tid % warpSize;

  double val = 0;
  if (glob_tid > 0 && glob_tid < n) {
    double x_i = a + glob_tid * h;
    val = f(x_i);
  }

  double sum = warp_shuffle_dissemination_sum(val);
  if (lane == 0) {
    atomicAdd(res, sum);
  }
}
