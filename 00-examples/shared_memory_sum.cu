#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
/**
Performs an intra-block sum of an array in shared memory
using a tree-based reduction (shared-memory version).
Assume sdata[tid] is already loaded with each thread’s value
*/
__device__ float shared_mem_sum(float *sdata) {
  int tid = threadIdx.x;
  for (int stride = (blockDim.x << 1); stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  return sdata[0];
}

__global__ void kernel(float *sum) {
  __shared__ float sdata[32];
  
  const unsigned int tid = threadIdx.x;
  sdata[tid] = tid;

  __syncthreads();
  *sum = shared_mem_sum(sdata);
}

int main(int argc, char *argv[]) {
  float *sum;
  cudaMallocManaged(&sum, sizeof(float));

  kernel<<<1, 32>>>(sum);
  cudaDeviceSynchronize();
  
  printf("Sum: %f\n", *sum);
  float correct = 32 * 31 / 2;
  if (fabs(*sum - correct) > 1e-5) {
    printf("Result is incorrect. Expected %f.\n", correct);
  }
  
  cudaFree(sum);
  return 0;
}