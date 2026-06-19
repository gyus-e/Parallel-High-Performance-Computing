#include <cuda_runtime.h>
#include <stdio.h>

__managed__ int managed_var;

__global__ void kernel(int *malloc_managed_var) {
    printf("Start of kernel: malloc_managed_var = %d; managed_var = %d\n", *malloc_managed_var, managed_var);
    managed_var = 42;
    *malloc_managed_var = 24;
    printf("End of kernel: malloc_managed_var = %d; managed_var = %d\n", *malloc_managed_var, managed_var);
}

int main() {
    int *malloc_managed_var;
    cudaMallocManaged(&malloc_managed_var, sizeof(int));

    *malloc_managed_var = 5;
    managed_var = 3;
    printf("Before kernel: malloc_managed_var = %d; managed_var = %d\n", *malloc_managed_var, managed_var);

    kernel<<<1, 1>>>(malloc_managed_var);
    cudaDeviceSynchronize();

    printf("After kernel: malloc_managed_var = %d; managed_var = %d\n", *malloc_managed_var, managed_var);
    return 0;
}
