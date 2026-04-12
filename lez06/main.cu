#include "integrals.hpp"
#include <math.h>
#include <omp.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WARP_SIZE 32

int checkErr(const double a, const double b, const char a_name[], const char b_name[]) {
  if (fabs(a - b) > 1e-6) {
    printf("Error: %s = %lf; %s = %lf.\n", a_name, a, b_name, b);
    return 1;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  const unsigned long n = pow(2, 20);
  const unsigned int nt = omp_get_max_threads();

  unsigned int k = 1;
  if (argc > 1) {
    k = atoi(argv[1]);
    if (k <= 0) {
      printf("Invalid value for k: %s. Using default k=1.\n", argv[1]);
      k = 1;
    }
  }
  const unsigned int blockSize = k*WARP_SIZE;
  const unsigned int gridSize = (n + blockSize - 1) / blockSize;

  printf("Using %lu subdivisions for the integral approximation.\n", n);
  printf("Using %u threads for the CPU version.\n", nt);
  printf("Using %u threads per block and %u blocks for the GPU version.\n\n",
         blockSize, gridSize);

  const double a = 0.0;
  const double b = M_PI;
  const double h = (b - a) / (double)n;

  double cpu_st_time;
  const char* cpu_st_label = "CPU sequential";
  const double cpu_st_res = integral_cpu(a, b, h, n, 1, &cpu_st_time);
  printf("[%s] Integral of f(x) from %lf to %lf = %lf\n", cpu_st_label, a, b, cpu_st_res);
  printf("[%s] Time taken: %lf ms\n\n", cpu_st_label, cpu_st_time);

  double cpu_mt_time, cpu_mt_sp;
  const char* cpu_mt_label = "CPU multithread";
  const double cpu_mt_res = integral_cpu(a, b, h, n, nt, &cpu_mt_time);
  cpu_mt_sp = cpu_st_time / cpu_mt_time;
  printf("[%s] Integral of f(x) from %lf to %lf = %lf\n", cpu_mt_label, a, b, cpu_mt_res);
  printf("[%s] Time taken: %lf ms; Speedup: %lf\n\n", cpu_mt_label, cpu_mt_time, cpu_mt_sp);

  double gpu_naive_time, gpu_naive_sp;
  const char* gpu_naive_label = "GPU naive";
  const double gpu_naive_res = integral_gpu_naive(a, b, h, n, blockSize, gridSize, &gpu_naive_time);
  gpu_naive_sp = cpu_st_time / gpu_naive_time;
  printf("[%s] Integral of f(x) from %lf to %lf = %lf\n", gpu_naive_label, a, b, gpu_naive_res);
  printf("[%s] Time taken: %lf ms; Speedup: %lf\n\n", gpu_naive_label, gpu_naive_time, gpu_naive_sp);

  double gpu_shared_mem_time, gpu_shared_mem_sp;
  const char* gpu_shared_mem_label = "GPU shared memory tree-structured sum";
  const double gpu_shared_mem_res = integral_gpu_shared_mem_tree_sum(a, b, h, n, blockSize, gridSize, &gpu_shared_mem_time);
  gpu_shared_mem_sp = cpu_st_time / gpu_shared_mem_time;
  printf("[%s] Integral of f(x) from %lf to %lf = %lf\n", gpu_shared_mem_label, a, b, gpu_shared_mem_res);
  printf("[%s] Time taken: %lf ms; Speedup: %lf\n\n", gpu_shared_mem_label, gpu_shared_mem_time, gpu_shared_mem_sp);

  double gpu_warp_shuffle_time, gpu_warp_shuffle_sp;
  const char* gpu_warp_shuffle_label = "GPU warp shuffle tree-structured sum";
  const double gpu_warp_shuffle_res = integral_gpu_warp_shuffle_tree_sum(a, b, h, n, blockSize, gridSize, &gpu_warp_shuffle_time);
  gpu_warp_shuffle_sp = cpu_st_time / gpu_warp_shuffle_time;
  printf("[%s] Integral of f(x) from %lf to %lf = %lf\n", gpu_warp_shuffle_label, a, b, gpu_warp_shuffle_res);
  printf("[%s] Time taken: %lf ms; Speedup: %lf\n\n", gpu_warp_shuffle_label, gpu_warp_shuffle_time, gpu_warp_shuffle_sp);

  double gpu_shared_mem_dissemination_time, gpu_shared_mem_dissemination_sp;
  const char* gpu_shared_mem_dissemination_label = "GPU shared memory dissemination sum";
  const double gpu_shared_mem_dissemination_res = integral_gpu_shared_mem_dissemination_sum(a, b, h, n, blockSize, gridSize, &gpu_shared_mem_dissemination_time);
  gpu_shared_mem_dissemination_sp = cpu_st_time / gpu_shared_mem_dissemination_time;
  printf("[%s] Integral of f(x) from %lf to %lf = %lf\n", gpu_shared_mem_dissemination_label, a, b, gpu_shared_mem_dissemination_res);
  printf("[%s] Time taken: %lf ms; Speedup: %lf\n\n", gpu_shared_mem_dissemination_label, gpu_shared_mem_dissemination_time, gpu_shared_mem_dissemination_sp);

  double gpu_warp_shuffle_dissemination_time, gpu_warp_shuffle_dissemination_sp;
  const char* gpu_warp_shuffle_dissemination_label = "GPU warp shuffle dissemination sum";
  const double gpu_warp_shuffle_dissemination_res = integral_gpu_warp_shuffle_dissemination_sum(a, b, h, n, blockSize, gridSize, &gpu_warp_shuffle_dissemination_time);
  gpu_warp_shuffle_dissemination_sp = cpu_st_time / gpu_warp_shuffle_dissemination_time;
  printf("[%s] Integral of f(x) from %lf to %lf = %lf\n", gpu_warp_shuffle_dissemination_label, a, b, gpu_warp_shuffle_dissemination_res);
  printf("[%s] Time taken: %lf ms; Speedup: %lf\n\n", gpu_warp_shuffle_dissemination_label, gpu_warp_shuffle_dissemination_time, gpu_warp_shuffle_dissemination_sp);

  return checkErr(cpu_st_res, cpu_st_res, "cpu_st_res", "cpu_mt_res") +
         checkErr(cpu_st_res, gpu_naive_res, "cpu_st_res", "gpu_naive_res") +
         checkErr(cpu_st_res, gpu_shared_mem_res, "cpu_st_res", "gpu_shared_mem_res") +
         checkErr(cpu_st_res, gpu_warp_shuffle_res, "cpu_st_res", "gpu_warp_shuffle_res") +
         checkErr(cpu_st_res, gpu_shared_mem_dissemination_res, "cpu_st_res", "gpu_shared_mem_dissemination_sum_res") +
         checkErr(cpu_st_res, gpu_warp_shuffle_dissemination_res, "cpu_st_res", "gpu_warp_shuffle_dissemination_res");
}
