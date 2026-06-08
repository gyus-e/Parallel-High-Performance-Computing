#include "mat_kernels.hpp"
#include <cuda_runtime.h>

#define TILE_SIZE 32

inline __device__ double dotProduct(const double *__restrict__ A,
                                    const double *__restrict__ B,
                                    const size_t K, 
                                    const size_t ldA, const size_t ldB, 
                                    const unsigned int rowA,
                                    const unsigned int colB) {
  double res = 0.0;
  unsigned int k;
  for (k = 0; k < K; k++) {
    res += A[rowA * ldA + k] * B[k * ldB + colB];
  }
  return res;
}

__global__ void hadamard(const double *__restrict__ A,
                         const double *__restrict__ B, 
                         double *__restrict__ C,
                         const size_t N, const size_t K, const size_t M,
                         const size_t ldA, const size_t ldB, const size_t ldC) {
  const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned int idxA = row * ldA + col;
  const unsigned int idxB = row * ldB + col;
  const unsigned int idxC = row * ldC + col;

  if (row < N && col < M) {
    C[idxC] = A[idxA] * B[idxB];
  }

  return;
}

__global__ void matmat_coalesced(const double *__restrict__ A,
                                 const double *__restrict__ B,
                                 double *__restrict__ C, 
                                 const size_t N, const size_t K, const size_t M,
                                 const size_t ldA, const size_t ldB, const size_t ldC) {
  const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < M) {
    C[row * ldC + col] = dotProduct(A, B, K, ldA, ldB, row, col);
  }
}

__global__ void matmat_uncoalesced(const double *__restrict__ A,
                                   const double *__restrict__ B,
                                   double *__restrict__ C, 
                                   const size_t N, const size_t K, const size_t M,
                                   const size_t ldA, const size_t ldB, const size_t ldC) {
  const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < N && col < M) {
    C[row * ldC + col] = dotProduct(A, B, K, ldA, ldB, row, col);
  }
}

__global__ void matmat_tiled(const double *__restrict__ A,
                             const double *__restrict__ B,
                             double *__restrict__ C, 
                             const size_t N, const size_t K, const size_t M, 
                             const size_t ldA, const size_t ldB, const size_t ldC) {
  const unsigned int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned int localRow = threadIdx.y;
  const unsigned int localCol = threadIdx.x;

  __shared__ double shared_A[TILE_SIZE * TILE_SIZE];
  __shared__ double shared_B[TILE_SIZE * TILE_SIZE];

  double res = 0.0f;

  for (int i = 0; i < K; i += TILE_SIZE) {
    if (globalRow < N && (i + localCol) < K) {
      shared_A[localRow * TILE_SIZE + localCol] =
          A[globalRow * ldA + (i + localCol)];
    } else {
      shared_A[localRow * TILE_SIZE + localCol] = 0.0;
    }
    if ((i + localRow) < K && globalCol < M) {
      shared_B[localRow * TILE_SIZE + localCol] =
          B[(i + localRow) * ldB + globalCol];
    } else {
      shared_B[localRow * TILE_SIZE + localCol] = 0.0;
    }
    __syncthreads();

    res += dotProduct(shared_A, shared_B, TILE_SIZE, TILE_SIZE, TILE_SIZE,
                      localRow, localCol);
    __syncthreads();
  }

  if (globalRow < N && globalCol < M) {
    C[globalRow * ldC + globalCol] = res;
  }
}
