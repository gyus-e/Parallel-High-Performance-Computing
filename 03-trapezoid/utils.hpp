#include <math.h>

#ifndef UTILS_HPP
#define UTILS_HPP

#define WARP_SIZE 32
#define MAX_BLKSZ 1024
#define FULL_MASK 0xffffffff

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define A 0.0
#define B (2*M_PI)
#define F_CORRECT -0.12212260462
inline __device__ __host__ double f(const double x) { return x*exp(-x)*cos(2*x); }

#endif /* UTILS_HPP */
