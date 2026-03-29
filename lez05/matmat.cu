#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

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
  Each block is responsible for computing one square sub-matrix Csub of C 
  by loading tiles of input matrices A and B from global memory to shared memory.
  Each thread within the block: 
  1) loads a single element from A and B into shared memory
  2) computes a single element of Csub using all the elements loaded by the threads in the block
  */

  const unsigned int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned int localRow = threadIdx.y;
  const unsigned int localCol = threadIdx.x;

  __shared__ double shared_A[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ double shared_B[BLOCK_SIZE * BLOCK_SIZE];

  double res = 0.0f;
  
  for (int i = 0; i < K; i += BLOCK_SIZE) {
    shared_A[localRow * BLOCK_SIZE + localCol] = A[globalRow * ldA + (i + localCol)];
    shared_B[localRow * BLOCK_SIZE + localCol] = B[(i + localRow) * ldB + globalCol];
    __syncthreads(); // Barrier to ensure all threads loaded their share of data into shared memory before proceeding

    for (int k = 0; k < BLOCK_SIZE; k++) {
      res += shared_A[localRow * BLOCK_SIZE + k] * shared_B[k * BLOCK_SIZE + localCol];
    }
    __syncthreads(); // Barrier to ensure all threads have completed their computation using the current tile before replacing it with the next tile
  }

  if (globalRow < N && globalCol < M) {
    C[globalRow * ldC + globalCol] = res;
  }
}

int main() {
  const unsigned int N = 1024;
  const unsigned int M = 1024;
  const unsigned int K = 1024;
  const unsigned int LD = 2048;

  double start, end;

  double *h_A, *d_A; // N x K
  double *h_B, *d_B; // K x M
  double *h_C, *d_C; // N x M

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