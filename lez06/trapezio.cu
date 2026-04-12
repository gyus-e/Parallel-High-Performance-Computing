#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WARP_SIZE 32
#define SHARED_MEM_SIZE 256

inline __device__ __host__ double f(const double x) { return 2*sin(x) + 3*pow(cos(x), 2) + 5*pow(x, 3); }

/**
Trapezoidal rule: the integral of f(x) from a to b is approximated by:
(h/2) * [f(x_0) + 2 * sum[i=1 to n-1](f(x_i)) + f(x_n)]
where:
h = (b - a) / n
x_i = a + i * h
*/

/**
CPU version: 
each thread computes one addend of the sum (2*f(x_i)) and adds it to the result using atomicAdd.
*/
void trap_cpu(const double a, const unsigned long n, const double h,
              double &res) {
  double sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for (int i = 1; i < n; i++) {
    double x_i = a + i * h;
    sum += f(x_i);
  }
  res = sum;
}

/**
GPU naive version: 
same as CPU, but we assume that the number of threads is at least n. 
We also assume a one-dimensional grid and block configuration.
*/
__global__ void trap_gpu_naive(const double a, const unsigned long n,
                               const double h, double *res) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0 || i >= n) {
    return;
  }
  double x_i = a + i * h;
  atomicAdd(res, f(x_i));
}

/**
GPU shared memory, tree-structured sum: 
pair up the threads so that half of the “active” threads add their partial sum to their partner’s partial sum. 
*/
__device__ void shared_mem_sum(double *sdata, const unsigned int sdata_len) {
  const unsigned int tid = threadIdx.x;
  for (unsigned int stride = (blockDim.x >> 1); stride > 0; stride >>= 1) {
    if (tid < stride && tid + stride < sdata_len) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
}

__global__ void trap_gpu_shared_mem(const double a, const unsigned long n,
                            const double h, double *res) {
  __shared__ double sdata[SHARED_MEM_SIZE];

  const unsigned int tid = threadIdx.x;
  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid <= SHARED_MEM_SIZE && i < n && i > 0) {
    double x_i = a + i * h;
    sdata[tid] = f(x_i);
  }
  __syncthreads();

  shared_mem_sum(sdata, SHARED_MEM_SIZE);
  if (tid == 0) {
    atomicAdd(res, sdata[0]);
  }
}

/**
GPU shared memory, tree-structured sum: 
use warp shuffle instructions (available in devices with compute capability >= 3.0).
Warp shuffle instructions allow threads within a warp to read variables stored in another thread’s register in the warp.
This allows us to compute the global sum in registers, which are faster than shared memory.
*/
__device__ double warp_sum(double val) {
  const unsigned int mask = 0xFFFFFFFF; // all threads in the warp are active
  for (unsigned int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  return val;
}

__global__ void trap_gpu_warp_shuffle(const double a, const unsigned long n,
                            const double h, double *res) {
  const unsigned int tid = threadIdx.x;
  const unsigned int lane = tid % WARP_SIZE;
  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  double val = 0;
  if (i > 0 && i < n) {
    double x_i = a + i * h;
    val = f(x_i);
  }

  double sum = warp_sum(val);
  if (lane == 0) {
    atomicAdd(res, sum);
  }
}


double integral_cpu(const double a, const double b, const double h,
                    const unsigned long n, const unsigned int nt) {
  double res = f(a) + f(b);
  double sum = 0;

  omp_set_num_threads(nt);

  double start = omp_get_wtime() * 1000;

  trap_cpu(a, n, h, sum);

  double end = omp_get_wtime() * 1000;
  printf("[CPU %s] Time taken: %lf ms\n", (nt == 1 ? "sequential" : "multithread"), end - start);

  res += 2 * sum;
  res *= h * 0.5;
  return res;
}

double integral_gpu_naive(const double a, const double b, const double h,
                    const unsigned long n, const unsigned int blockSize,
                    const unsigned int gridSize) {
  double res = f(a) + f(b);
  double *sum;
  cudaMallocManaged(&sum, sizeof(double));
  *sum = 0;

  dim3 dimBlock(blockSize);
  dim3 dimGrid(gridSize);

  double start = omp_get_wtime() * 1000;

  trap_gpu_naive<<<dimGrid, dimBlock>>>(a, n, h, sum);
  cudaDeviceSynchronize();

  double end = omp_get_wtime() * 1000;
  printf("[GPU naive] Time taken: %lf ms\n", end - start);

  res += 2 * (*sum);
  res *= h * 0.5;
  cudaFree(sum);
  return res;
}

double integral_gpu_shared_mem(const double a, const double b, const double h,
                    const unsigned long n, const unsigned int blockSize,
                    const unsigned int gridSize) {
  double res = f(a) + f(b);
  double *sum;
  cudaMallocManaged(&sum, sizeof(double));
  *sum = 0;

  dim3 dimBlock(blockSize);
  dim3 dimGrid(gridSize);

  double start = omp_get_wtime() * 1000;

  trap_gpu_shared_mem<<<dimGrid, dimBlock>>>(a, n, h, sum);
  cudaDeviceSynchronize();

  double end = omp_get_wtime() * 1000;
  printf("[GPU shared_mem] Time taken: %lf ms\n", end - start);

  res += 2 * (*sum);
  res *= h * 0.5;
  cudaFree(sum);
  return res;
}

double integral_gpu_warp_shuffle(const double a, const double b, const double h,
                    const unsigned long n, const unsigned int blockSize,
                    const unsigned int gridSize) {
  double res = f(a) + f(b);
  double *sum;
  cudaMallocManaged(&sum, sizeof(double));

  dim3 dimBlock(blockSize);
  dim3 dimGrid(gridSize);

  double start = omp_get_wtime() * 1000;

  trap_gpu_warp_shuffle<<<dimGrid, dimBlock>>>(a, n, h, sum);
  cudaDeviceSynchronize();

  double end = omp_get_wtime() * 1000;
  printf("[GPU warp_shuffle] Time taken: %lf ms\n", end - start);

  res += 2 * (*sum);
  res *= h * 0.5;
  cudaFree(sum);
  return res;                    
}


int checkErr(const double a, const double b, const char a_name[], const char b_name[]) {
  if (fabs(a - b) > 1e-6) {
    printf("Error: %s = %lf; %s = %lf.\n", a_name, a, b_name, b);
    return 1;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  const unsigned long n = pow(2, 16);
  const unsigned int nt = omp_get_max_threads();
  const unsigned int blockSize = 2*WARP_SIZE;
  const unsigned int gridSize = (n + blockSize - 1) / blockSize;
  printf("Using %lu subdivisions for the integral approximation.\n", n);
  printf("Using %u threads for the CPU version.\n", nt);
  printf("Using %u threads per block and %u blocks for the GPU version.\n\n",
         blockSize, gridSize);

  const double a = 0.0;
  const double b = M_PI;
  const double h = (b - a) / (double)n;

  const double cpu_st_res = integral_cpu(a, b, h, n, 1);
  printf("[CPU sequential] Integral of f(x) from %lf to %lf = %lf\n\n", a, b, cpu_st_res);

  const double cpu_mt_res = integral_cpu(a, b, h, n, nt);
  printf("[CPU multithread] Integral of f(x) from %lf to %lf = %lf\n\n", a, b, cpu_mt_res);

  const double gpu_naive_res = integral_gpu_naive(a, b, h, n, blockSize, gridSize);
  printf("[GPU naive] Integral of f(x) from %lf to %lf = %lf\n\n", a, b, gpu_naive_res);

  const double gpu_shared_mem_res = integral_gpu_shared_mem(a, b, h, n, blockSize, gridSize);
  printf("[GPU shared_mem] Integral of f(x) from %lf to %lf = %lf\n\n", a, b, gpu_shared_mem_res);

  const double gpu_warp_shuffle_res = integral_gpu_warp_shuffle(a, b, h, n, blockSize, gridSize);
  printf("[GPU warp_shuffle] Integral of f(x) from %lf to %lf = %lf\n\n", a, b, gpu_warp_shuffle_res);

  return checkErr(cpu_st_res, cpu_st_res, "cpu_st_res", "cpu_mt_res") +
         checkErr(cpu_st_res, gpu_naive_res, "cpu_st_res", "gpu_naive_res") +
         checkErr(cpu_st_res, gpu_shared_mem_res, "cpu_st_res", "gpu_shared_mem_res") +
         checkErr(cpu_st_res, gpu_warp_shuffle_res, "cpu_st_res", "gpu_warp_shuffle_res");
}
