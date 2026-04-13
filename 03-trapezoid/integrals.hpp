#ifndef INTEGRALS_HPP
#define INTEGRALS_HPP

double integral_cpu(const double a, const double b, const double h, const unsigned long n, 
                    const unsigned int nt, double *time);

double integral_gpu_naive(const double a, const double b, const double h, const unsigned long n, 
                          const unsigned int blockSize, const unsigned int gridSize, double *time);

double integral_gpu_shared_mem_tree_sum(const double a, const double b, const double h, const unsigned long n, 
                                        const unsigned int blockSize, const unsigned int gridSize, double *time);

double integral_gpu_warp_shuffle_tree_sum(const double a, const double b, const double h, const unsigned long n, 
                                          const unsigned int blockSize, const unsigned int gridSize, double *time);

double integral_gpu_shared_mem_dissemination_sum(const double a, const double b, const double h, const unsigned long n, 
                                                 const unsigned int blockSize, const unsigned int gridSize, double *time);

double integral_gpu_warp_shuffle_dissemination_sum(const double a, const double b, const double h, const unsigned long n, 
                                                   const unsigned int blockSize, const unsigned int gridSize, double *time);

#endif /* INTEGRALS_HPP */
