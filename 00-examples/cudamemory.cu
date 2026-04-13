#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/**
GPUs and CPUs both have directly attached DRAM chips. 
In systems with more than one GPU, each GPU has its own memory. 

From the perspective of device code, the DRAM attached to the GPU is called global memory, because it is accessible to all SMs in the GPU. 
This terminology does not mean it is necessarily accessible everywhere within the system. 

The DRAM attached to the CPU(s) is called system memory or host memory.

Like CPUs, GPUs use virtual memory addressing. 
On all currently-supported systems, the CPU and GPU use a single unified virtual memory space. 
This means that the virtual memory address range for each GPU in the system is unique and distinct from the CPU and every other GPU in the system. 
For a given virtual memory address, it is possible to determine whether that address is in GPU memory or system memory and, 
on systems with multiple GPUs, which GPU memory contains that address.

There are CUDA APIs to allocate GPU memory, CPU memory, and to copy between allocations on the CPU and GPU, within a GPU, or between GPUs in multi-GPU systems.
The locality of data can be explicitly controlled when desired. 

When an application allocates memory explicitly on the GPU or CPU, that memory is only accessible to code running on that device. 
That is, CPU memory can only be accessed from CPU code, and GPU memory can only be accessed from kernels running on the GPU
*/
int main() {
  int N = 1024;
  size_t SIZE = N * sizeof(float);

  // Allocate host memory
  float *h_A = (float *)malloc(SIZE);
  float *h_B = (float *)malloc(SIZE);

  // Allocate device memory
  float *d_A;
  cudaMalloc((void **)&d_A, SIZE);

  // Initialize host array
  for (int i = 0; i < N; i++) {
    h_A[i] = i * 1.0f;
  }

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice);

  // Copy from device to host
  cudaMemcpy(h_B, d_A, SIZE, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    printf("%.2f ", h_B[i]);
  }
  printf("\n");

  // Free memory
  free(h_A);
  free(h_B);
  cudaFree(d_A);

  printf("Memory transfer completed successfully.\n");

  return 0;
}
/**
In addition to the global memory, each GPU has some on-chip memory.
Each SM has its own register file and shared memory. 
These memories are part of the SM and can be accessed extremely quickly from threads executing within the SM, 
but they are not accessible to threads running in other SMs.

The register file stores thread local variables which are usually allocated by the compiler. 

The shared memory is accessible by all threads within a thread block or cluster. 
Shared memory can be used for exchanging data between threads of a thread block or cluster.

In addition to programmable memories, GPUs have both L1 and L2 caches. 
Each SM has an L1 cache which is part of the unified data cache. 
A larger L2 cache is shared by all SMs within a GPU. 
Each SM also has a separate constant cache, which is used to cache values in global memory that have been declared to be constant.

A CUDA feature called unified memory allows applications to make memory allocations which can be accessed from CPU or GPU. 
The CUDA runtime or underlying hardware enables access or relocates the data to the correct place when needed. 
Even with unified memory, optimal performance is attained by keeping the migration of memory to a minimum and accessing data 
from the processor directly attached to the memory where it resides as much as possible.
*/