#ifndef TRAPEZOID_HPP
#define TRAPEZOID_HPP

void trap_cpu(const double a, const unsigned long n, const double h,
              double &res);
__global__ void trap_gpu_naive(const double a, const unsigned long n,
                const double h, double *res);
__global__ void trap_gpu_shared_mem_tree_sum(const double a, const unsigned long n,
                            const double h, double *res);
__global__ void trap_gpu_warp_shuffle_tree_sum(const double a, const unsigned long n,
                            const double h, double *res);

#endif // TRAPEZOID_HPP