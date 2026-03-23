#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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