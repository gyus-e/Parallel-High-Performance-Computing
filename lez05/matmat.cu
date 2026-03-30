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
  Each block is responsible for computing one square sub-matrix of C 
  by loading tiles of input matrices A and B from global memory to shared memory.
  Each thread within the block: 
  1) loads a single element from A and B into shared memory
  2) partially computes a single element of C using the elements from the tiles of A and B loaded by the threads in the block
  After that, all the threads in the block move on to the next tile.
  */

  const unsigned int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned int localRow = threadIdx.y;
  const unsigned int localCol = threadIdx.x;

  __shared__ double shared_A[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ double shared_B[BLOCK_SIZE * BLOCK_SIZE];

  double res = 0.0f;
  
  // A and B are divided into tiles of size BLOCK_SIZE x BLOCK_SIZE. 
  // The loop iterates over the K dimension in steps of BLOCK_SIZE, effectively processing one tile of A and B at a time.
  for (int i = 0; i < K; i += BLOCK_SIZE) {
    // Each thread loads one element of the tile of A and B into shared memory.
    // The element is determined by the thread's local row and column indices within the block, and the current tile (i) in the K dimension.
    if (globalRow < N && (i + localCol) < K) {
      shared_A[localRow * BLOCK_SIZE + localCol] = A[globalRow * ldA + (i + localCol)];
    } else {
      shared_A[localRow * BLOCK_SIZE + localCol] = 0.0;
    }
    if ((i + localRow) < K && globalCol < M) {
      shared_B[localRow * BLOCK_SIZE + localCol] = B[(i + localRow) * ldB + globalCol];
    } else {
      shared_B[localRow * BLOCK_SIZE + localCol] = 0.0;
    }
    // Barrier to ensure all threads loaded their share of data into shared memory before proceeding
    __syncthreads(); 

    // The thread accumulates the dot product of the current tile of A and B into the result.
    res += dotProduct(shared_A, shared_B, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, localRow, localCol);
    // Barrier to ensure all threads have completed their computation using the current tile before replacing it with the next tile
    __syncthreads(); 
  }

  if (globalRow < N && globalCol < M) {
    C[globalRow * ldC + globalCol] = res;
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