#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__host__ void print_mat(const double *mat, const unsigned int rows,
                        const unsigned int cols, const unsigned int ld) {
  unsigned int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      printf("%.1f \t", mat[i * ld + j]);
    }
    printf("\n");
  }
}

__host__ __device__ void init_mat(double *mat, const unsigned int rows,
                                  const unsigned int cols,
                                  const unsigned int ld) {
  unsigned int i, j;
  double x = 0.0;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      mat[i * ld + j] = ++x;
    }
  }
}

__host__ __device__ double dotProduct(const double *A, const double *B, const unsigned int K,
                                      const unsigned int ldA, const unsigned int ldB,
                                      const unsigned int rowA, const unsigned int colB) {
  double sum = 0.0;
  unsigned int k;
  for (k = 0; k < K; k++) {
    sum += A[rowA * ldA + k] * B[k * ldB + colB];
  }
  return sum;
}

__global__ void matmat_naive(const double *A, const double *B, double *C,
                             const unsigned int N, const unsigned int M,
                             const unsigned int K, const unsigned int ldA,
                             const unsigned int ldB, const unsigned int ldC) {
  /**
  One thread computes one element of the output matrix C.
  No shared memory is used, so all threads read from global memory.
  */
  const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < M) {
    C[row * ldC + col] = dotProduct(A, B, K, ldA, ldB, row, col);
  }
}

void matmatikj(const double *A, const double *B, double *C, const unsigned int N,
              const unsigned int M, const unsigned int K,
              const unsigned int ldA, const unsigned int ldB,
              const unsigned int ldC) {
  unsigned int i, j, k;
  for (i = 0; i < N; i++) {
    for (k = 0; k < K; k++) {
      for (j = 0; j < M; j++) {
        C[i * ldC + j] += A[i * ldA + k] * B[k * ldB + j];
      }
    }
  }
}

int main() {
  const unsigned int N = 32;
  const unsigned int M = 32;
  const unsigned int K = 32;
  const unsigned int LD = 2048;

  double *h_A, *d_A; // N x K
  double *h_B, *d_B; // K x M
  double *h_C, *d_C; // N x M

  /**
  We need as many threads as elements in C.
  We also need a 2-dimensional block to compute the row and column indices of C.
  For example, we use a grid of 16 blocks, and each block has 8x8 threads. 
  8x8x16 = 1024 = 32x32, which is the size of C.
  Reminder that warp size is 32, so each block should at least have 32 threads to keep the GPU busy.
  */
  dim3 DimGrid(16);
  dim3 DimBlock(8, 8);

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

  matmat_naive<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, N, M, K, LD, LD, LD);

  cudaMemcpy(h_C, d_C, N * LD * sizeof(double), cudaMemcpyDeviceToHost);

  print_mat(h_C, 5, 5, LD);
  printf("Check that it is equal to the following:\n");

  free(h_C);
  h_C = (double *)calloc(N * LD, sizeof(double));
  matmatikj(h_A, h_B, h_C, N, M, K, LD, LD, LD);
  print_mat(h_C, 5, 5, LD);

  cudaFree(d_C);
  cudaFree(d_B);
  cudaFree(d_A);
  free(h_C);
  free(h_B);
  free(h_A);

  return 0;
}