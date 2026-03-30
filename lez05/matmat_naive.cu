#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.hpp"

#define BLOCK_SIZE 16

__device__ double dotProduct(const double *A, const double *B, const unsigned int K,
                                      const unsigned int ldA, const unsigned int ldB,
                                      const unsigned int rowA, const unsigned int colB) {
  double sum = 0.0;
  unsigned int k;
  for (k = 0; k < K; k++) {
    sum += A[rowA * ldA + k] * B[k * ldB + colB];
  }
  return sum;
}

__global__ void matmat(const double *A, const double *B, double *C,
                             const unsigned int N, const unsigned int M,
                             const unsigned int K, const unsigned int ldA,
                             const unsigned int ldB, const unsigned int ldC) {
  /**
  We assume 2-dimensional blocks and grid.
  One thread computes one element of C.
  No shared memory is used, so all threads read from global memory.
  */
  const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < M) {
    C[row * ldC + col] = dotProduct(A, B, K, ldA, ldB, row, col);
  }
}

int main() {
  const unsigned int N = 8192;
  const unsigned int M = 8192;
  const unsigned int K = 8192;
  const unsigned int LD = 8192;

  double start, end;

  double *h_A, *d_A; // N x K
  double *h_B, *d_B; // K x M
  double *h_C, *d_C; // N x M

  /**
  We need as many threads as elements in C.
  Recommended minimum number of threads in a block is 32, the warp size.
  Maximum number of threads in a block is 1024 (so the dimensions must multiply to a value less or equal than 1024).

  For example, a very simple configuration could be:
  dim3 DimGrid(1024);
  dim3 DimBlock(32, 32);
  But having a 1-dimensional grid of 1024 blocks means that blockIdx.y will always be equal to 0,
  so the row index of C will be computed only with threadIdx.y, which can take values from 0 to 31.

  We need a 2-dimensional grid and a 2-dimensional block.
  In a block, we want a little more than the warp size to exploit temporal parallelism.
  So for a problem of size 1024 x 1024, the following works:
  dim3 DimGrid(64, 64);
  dim3 DimBlock(16, 16);
  
  Note that, if we fix DimBlock, then DimGrid must be computed as:
  DimGrid.x = (M + DimBlock.x - 1) / DimBlock.x
  DimGrid.y = (N + DimBlock.y - 1) / DimBlock.y
  */
  dim3 DimGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);


  h_A = (double *)malloc(N * LD * sizeof(double));
  h_B = (double *)malloc(K * LD * sizeof(double));
  h_C = (double *)calloc(N * LD, sizeof(double));
  cudaMalloc((void **)&d_A, N * LD * sizeof(double));
  cudaMalloc((void **)&d_B, K * LD * sizeof(double));
  cudaMalloc((void **)&d_C, N * LD * sizeof(double));

  init_mat(h_A, N, K, LD);
  init_mat(h_B, K, M, LD);

  cudaMemcpy(d_A, h_A, N * LD * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * LD * sizeof(double), cudaMemcpyHostToDevice);

  start = get_cur_time();
  matmat<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, N, M, K, LD, LD, LD);
  cudaMemcpy(h_C, d_C, N * LD * sizeof(double), cudaMemcpyDeviceToHost);
  end = get_cur_time();
  printf("Time: %f ms\n", end - start);

  print_mat(h_C, 5, 5, LD);

  cudaFree(d_C);
  cudaFree(d_B);
  cudaFree(d_A);
  free(h_C);
  free(h_B);
  free(h_A);

  return 0;
}