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
*/
__device__ double shared_mem_tree_sum(double sdata[]) {
  const unsigned int tid = threadIdx.x;
  for (unsigned int stride = (blockDim.x >> 1); stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
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
*/
__device__ double warp_shuffle_tree_sum(double val) {
  for (unsigned int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  return val;
}

/**
---------------------------------------------------------------------------------
DISSEMINATION SUM (SINGLE WARP)
---------------------------------------------------------------------------------
Within a warp, every thread reads a value from another thread in each step. 
At the end, all the threads in the warp have the same partial sum.
On each step, shift 'val' down by 'offset' lanes and add it to the local 'val’.
Make sure 0 <= source <= warpSize.
*/
__device__ double shared_mem_dissemination_sum_single_warp(double sdata[]) {
  const unsigned int tid = threadIdx.x;
  const unsigned int lane = tid % warpSize;

  for (unsigned int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
    const unsigned int source = (lane + offset) % warpSize;
    sdata[lane] += sdata[source];
    __syncwarp(FULL_MASK);
  }
  return sdata[lane];
}

/**
Warps can use warp shuffles to compute the warp sum.
To get the result in all the threads in the warp, 
we can use an alternative warp shuffle function: 
__shfl_xor
*/
__device__ double warp_shuffle_dissemination_sum_single_warp(double val) {  
  for (unsigned int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(FULL_MASK, val, offset);
  }
  return val;
}

/**
---------------------------------------------------------------------------------
DISSEMINATION SUM (MULTIPLE WARPS)
---------------------------------------------------------------------------------
If we have several warps of threads synchronized in a block, we can use warp 0 to add the sum computed by each of the other warps.

Two threads are in the same warp if their ranks in the block have the same quotient when divided by warpSize
warp = tid/warpSize

warp 0 consists of all the threads with rank between 0 and warpSize-1

There can be a shared array to store up to 32 elements,
thread with lane 0 of each warp w will store its own sum in the w element of that array
*/
__device__ double warp_shuffle_dissemination_sum(double val) { 
  return warp_shuffle_dissemination_sum_single_warp(val);
}

/**
If we’re using shared memory instead of warp shuffles to
compute the warp sums, we’ll need enough shared memory
for each warp in a thread block.
Since shared variables are shared by all the threads in a
thread block, we need an array large enough to hold the
contributions of all of the threads to the sum.
We can declare an array with 1024 elements — the largest
possible block size — and partition it among the warps.
*/
__device__ double shared_mem_dissemination_sum(double sdata[]) { 
  return shared_mem_dissemination_sum_single_warp(sdata);
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

__global__ void trap_gpu_shared_mem_tree_sum(const double a,
                                             const unsigned long n,
                                             const double h, double *res) {
  extern __shared__ double sdata[];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  sdata[tid] = 0;
  if (i < n && i > 0) {
    double x_i = a + i * h;
    sdata[tid] = f(x_i);
  }
  __syncthreads();

  shared_mem_tree_sum(sdata);
  if (tid == 0) {
    atomicAdd(res, sdata[0]);
  }
}

__global__ void trap_gpu_warp_shuffle_tree_sum(const double a,
                                               const unsigned long n,
                                               const double h, double *res) {
  const unsigned int tid = threadIdx.x;
  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int lane = tid % warpSize;

  double val = 0;
  if (i > 0 && i < n) {
    double x_i = a + i * h;
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
  extern __shared__ double sdata[];
  const unsigned int tid = threadIdx.x;
  const unsigned int lane = tid % warpSize;
  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  sdata[tid] = 0;
  if (i < n && i > 0) {
    double x_i = a + i * h;
    sdata[tid] = f(x_i);
  }
  __syncthreads();

  shared_mem_dissemination_sum(sdata);
  if (lane == 0) {
    atomicAdd(res, sdata[lane]);
  }
}

__global__ void trap_gpu_warp_shuffle_dissemination_sum(const double a,
                                                        const unsigned long n,
                                                        const double h,
                                                        double *res) {
  const unsigned int tid = threadIdx.x;
  const unsigned int lane = tid % warpSize;
  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  double val = 0;
  if (i > 0 && i < n) {
    double x_i = a + i * h;
    val = f(x_i);
  }

  double sum = warp_shuffle_dissemination_sum(val);
  if (lane == 0) {
    atomicAdd(res, sum);
  }
}
