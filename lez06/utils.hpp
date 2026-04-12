#include <math.h>

#ifndef UTILS_HPP
#define UTILS_HPP

#define WARP_SIZE 32
#define MAX_BLKSZ 1024
#define FULL_MASK 0xffffffff

inline __device__ __host__ double f(const double x) { return 2*sin(x) + 3*pow(cos(x), 2) + 5*pow(x, 3); }

#endif /* UTILS_HPP */
