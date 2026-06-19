#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * Within a thread block, threads are organized into groups of 32 threads called warps. 
 * When threads are executed by a warp, they are assigned a warp lane. 
 * Warp lanes are numbered 0 to 31 and threads from a thread block are assigned to warps in a predictable fashion
 * All threads in the warp execute the same instruction simultaneously. 
 * If some threads within a warp follow a control flow branch in execution while others do not, 
 * the threads which do not follow the branch will be masked off while the threads which follow the branch are executed.
 * It follows that utilization of the GPU is maximized when threads within a warp follow the same control flow path.
 * One implication of warp execution is that thread blocks are best specified to have a total number of threads which is a multiple of 32.
*/
const int WARP_SIZE = 32;
/*
 While it is not necessary to consider warps when writing CUDA code, 
 understanding the warp execution model is helpful in understanding concepts 
 such as global memory coalescing and shared memory bank access patterns. 
 Some advanced programming techniques use specialization of warps within a thread block 
 to limit thread divergence and maximize utilization. 
 This and other optimizations make use of the knowledge that threads are grouped into warps when executing.
*/


/**
 * Functions which execute on the GPU which can be invoked from the host are called kernels. 
 * Kernels are written to be run by many parallel threads simultaneously. 
 * The code for a kernel is specified using the __global__ declaration specifier. 
 * This indicates to the compiler that this function will be compiled for the GPU in a way that allows it to be invoked from a kernel launch. 
 * A kernel launch is an operation which starts a kernel running, usually from the CPU. 
 * Kernels are functions with a void return type.
 */
__global__ void vecAdd(float *A, float *B, float *C);


int main() {
  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;
  int N, size;
  int i;

  N = 1024;
  size = N * sizeof(float);

  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  for (i = 0; i < N; i++) {
    h_A[i] = i + 1.0f;
    h_B[i] = i + 1.0f;
  }

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  /**
   * The number of threads that will execute the kernel in parallel is specified as part of the kernel launch. 
   * This is called the execution configuration.
   * Different invocations of the same kernel may use different execution configurations, 
   * such as a different number of threads or thread blocks.
   * Triple chevron ("<<<", ">>>") notation is a CUDA C++ Language Extension which is used to launch kernels. 
   * Execution configuration parameters are specified as a comma separated list inside the chevrons, 
   * similar to parameters to a function call. 
   * The first two parameters to the triple chevron notation are the grid dimensions and the thread block dimensions, respectively. 
   * When using 1-dimensional thread blocks or grids, integers can be used to specify dimensions. 
   * When using 2 or 3-dimensional grids or thread blocks, the CUDA type dim3 is used as the grid and thread block dimension parameters.
   * The code below launches a single thread block containing N threads.
   * On current GPUs, a thread block may contain up to 1024 threads
   * Each thread will execute the exact same kernel code. 
   * Each thread can use its index within the thread block and grid to change the data it operates on.
   */
  dim3 DimGrid(4, 1);
  dim3 DimBlock(256, 1);
  vecAdd<<<DimGrid, DimBlock>>>(d_A, d_B, d_C);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  for (i = 0; i < N; i++) {
    printf("%.2f ", h_C[i]);
  }
  printf("\n");

  cudaFree(d_C);
  cudaFree(d_B);
  cudaFree(d_A);

  free(h_C);
  free(h_B);
  free(h_A);
}


__global__ void vecAdd(float *A, float *B, float *C) {
  /**
  * threadIdx gives the index of a thread within its thread block. Each thread in a thread block will have a different index. 
  * blockDim gives the dimensions of the thread block, which was specified in the execution configuration of the kernel launch.
  * blockIdx gives the index of a thread block within the grid. Each thread block will have a different index.
  * gridDim gives the dimensions of the grid, which was specified in the execution configuration when the kernel was launched.
  * Each of these intrinsics is a 3-component vector with a .x, .y, and .z member. 
  * Dimensions not specified by a launch configuration will default to 1. 
  * threadIdx and blockIdx are zero indexed
  */

  // calculate which element this thread is responsible for computing
  int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

  // Perform computation
  C[workIndex] = A[workIndex] + B[workIndex];
}

/**
To schedule a thread block to an SM, the total number of registers needed for each thread 
multiplied by the number of threads in the thread block must be less than or equal to the available registers in the SM. 
If the number of registers required for a thread block exceeds the size of the register file, 
the kernel is not launchable and the number of threads in the thread block must be decreased to make the thread block launchable.
*/