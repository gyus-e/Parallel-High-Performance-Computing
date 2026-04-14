#include "integrals.hpp"
#include "utils.hpp"
#include <math.h>
#include <omp.h>
#include <stdio.h>



int checkErr(const double a, const double b, const char a_name[], const char b_name[]) {
  if (fabs(a - b) > 1e-6) {
    printf("\"Error\": {\"%s\": %lf},\n", b_name, b);
    return 1;
  }
  return 0;
}

void printJsonResult(const char* label, const double res, const double time_ms, const double speedup) {
  printf("\"%s\": {\n\t\"res\": %lf,\n\t\"time (ms)\": %lf,\n\t\"speedup\": %lf\n}", label, res, time_ms, speedup);
}

int main(int argc, char *argv[]) {
  unsigned int blocksize_exp = 1;
  unsigned int num_div_exp = 20;
  if (argc > 1) {
    blocksize_exp = atoi(argv[1]);
  }
  if (argc > 2) {
    num_div_exp = atoi(argv[2]);
  }
  const unsigned long n = pow(2, num_div_exp);
  const unsigned int k = pow(2, blocksize_exp);
  const unsigned int nt = omp_get_max_threads();
  const unsigned int blockSize = (k * WARP_SIZE) < MAX_BLKSZ ? (k * WARP_SIZE) : MAX_BLKSZ;
  const unsigned int gridSize = (n + blockSize - 1) / blockSize;

  const double a = A;
  const double b = B;
  const double h = (b - a) / (double)n;

  printf("{\n");
  printf("\"f(x)\": \"x*exp(-x)*cos(2*x)\",\n");
  printf("\"a\": %lf,\n", A);
  printf("\"b\": %lf,\n", B);
  printf("\"n\": %lf,\n", n);
  printf("\"nt (cpu)\": %u,\n", nt);
  printf("\"blockSize\": %u,\n", blockSize);
  printf("\"gridSize\": %u,\n", gridSize);

  double cpu_st_time;
  const char* cpu_st_label = "CPU sequential";
  const double cpu_st_res = integral_cpu(a, b, h, n, 1, &cpu_st_time);
  printJsonResult(cpu_st_label, cpu_st_res, cpu_st_time, 1.0);
  printf(",\n");

  double cpu_mt_time, cpu_mt_sp;
  const char* cpu_mt_label = "CPU multithread";
  const double cpu_mt_res = integral_cpu(a, b, h, n, nt, &cpu_mt_time);
  cpu_mt_sp = cpu_st_time / cpu_mt_time;
  printJsonResult(cpu_mt_label, cpu_mt_res, cpu_mt_time, cpu_mt_sp);
  printf(",\n");

  double gpu_naive_time, gpu_naive_sp;
  const char* gpu_naive_label = "GPU naive";
  const double gpu_naive_res = integral_gpu_naive(a, b, h, n, blockSize, gridSize, &gpu_naive_time);
  gpu_naive_sp = cpu_st_time / gpu_naive_time;
  printJsonResult(gpu_naive_label, gpu_naive_res, gpu_naive_time, gpu_naive_sp);
  printf(",\n");

  double gpu_shared_mem_time, gpu_shared_mem_sp;
  const char* gpu_shared_mem_label = "GPU shared memory tree-structured sum";
  const double gpu_shared_mem_res = integral_gpu_shared_mem_tree_sum(a, b, h, n, blockSize, gridSize, &gpu_shared_mem_time);
  gpu_shared_mem_sp = cpu_st_time / gpu_shared_mem_time;
  printJsonResult(gpu_shared_mem_label, gpu_shared_mem_res, gpu_shared_mem_time, gpu_shared_mem_sp);
  printf(",\n");

  double gpu_warp_shuffle_time, gpu_warp_shuffle_sp;
  const char* gpu_warp_shuffle_label = "GPU warp shuffle tree-structured sum";
  const double gpu_warp_shuffle_res = integral_gpu_warp_shuffle_tree_sum(a, b, h, n, blockSize, gridSize, &gpu_warp_shuffle_time);
  gpu_warp_shuffle_sp = cpu_st_time / gpu_warp_shuffle_time;
  printJsonResult(gpu_warp_shuffle_label, gpu_warp_shuffle_res, gpu_warp_shuffle_time, gpu_warp_shuffle_sp);
  printf(",\n");

  double gpu_shared_mem_dissemination_time, gpu_shared_mem_dissemination_sp;
  const char* gpu_shared_mem_dissemination_label = "GPU shared memory dissemination sum";
  const double gpu_shared_mem_dissemination_res = integral_gpu_shared_mem_dissemination_sum(a, b, h, n, blockSize, gridSize, &gpu_shared_mem_dissemination_time);
  gpu_shared_mem_dissemination_sp = cpu_st_time / gpu_shared_mem_dissemination_time;
  printJsonResult(gpu_shared_mem_dissemination_label, gpu_shared_mem_dissemination_res, gpu_shared_mem_dissemination_time, gpu_shared_mem_dissemination_sp);
  printf(",\n");

  double gpu_warp_shuffle_dissemination_time, gpu_warp_shuffle_dissemination_sp;
  const char* gpu_warp_shuffle_dissemination_label = "GPU warp shuffle dissemination sum";
  const double gpu_warp_shuffle_dissemination_res = integral_gpu_warp_shuffle_dissemination_sum(a, b, h, n, blockSize, gridSize, &gpu_warp_shuffle_dissemination_time);
  gpu_warp_shuffle_dissemination_sp = cpu_st_time / gpu_warp_shuffle_dissemination_time;
  printJsonResult(gpu_warp_shuffle_dissemination_label, gpu_warp_shuffle_dissemination_res, gpu_warp_shuffle_dissemination_time, gpu_warp_shuffle_dissemination_sp);
  printf("}\n");

  return checkErr(cpu_st_res, cpu_st_res, "cpu_st_res", "cpu_mt_res") +
         checkErr(cpu_st_res, gpu_naive_res, "cpu_st_res", "gpu_naive_res") +
         checkErr(cpu_st_res, gpu_shared_mem_res, "cpu_st_res", "gpu_shared_mem_res") +
         checkErr(cpu_st_res, gpu_warp_shuffle_res, "cpu_st_res", "gpu_warp_shuffle_res") +
         checkErr(cpu_st_res, gpu_shared_mem_dissemination_res, "cpu_st_res", "gpu_shared_mem_dissemination_res") +
         checkErr(cpu_st_res, gpu_warp_shuffle_dissemination_res, "cpu_st_res", "gpu_dissemination_res");
}
