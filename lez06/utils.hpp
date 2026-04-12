#ifndef UTILS_HPP
#define UTILS_HPP

#define SHARED_MEM_SIZE 256
#define FULL_MASK 0xffffffff

inline __device__ __host__ double f(const double x) { return 2*sin(x) + 3*pow(cos(x), 2) + 5*pow(x, 3); }

#endif /* UTILS_HPP */