#include <stdio.h>
#include <stdlib.h>

__global__ void sommaarray(float*,float*,float*);

int main(){
    float *A, *B, *C, *d_A, *d_B, *d_C;
    int size, i, N;

    N=10;
    size=N*sizeof(float);
    
    A=(float*)malloc(size);
    B=(float*)malloc(size);
    C=(float*)malloc(size);
    
    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_C,size);

    for(i=0;i<N;i++){
        A[i]=i+1.0f;
        B[i]=i+1.0f;
    }

    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

    dim3 DimGrid(1,1);
    dim3 DimBlock(N,1,1);
    sommaarray<<<DimGrid,DimBlock>>>(d_A,d_B,d_C);

    cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);
    for(i=0;i<N;i++){
        printf("%f ",C[i]);
    }
    printf("\n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void sommaarray(float *A,float *B,float *C){
    int i=threadIdx.x;
    C[i]=A[i]+B[i];
}
