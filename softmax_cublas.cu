#include "softmax_cublas.cuh"
#include <stdio.h>
#include <cublas_v2.h>
// #include <thrust/device_vector.h>
// #include <thrust/device_ptr.h>

__global__ void exp_kernel(float *x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * n) {
        out[i] = exp(x[i]);

    }
}

__global__ void ones_kernel(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 1;
    }
}



__global__ void inverse_kernel(float *x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 1 / x[i];
    }
}

/*
Iteration 1:
a) exp kernel, b) Reduce kernel, c) Divide kernel
*/

__host__ void softmax_cublas(float *input, float *output, int N) {
    int threads_per_block = 1024;
    int num_blocks_exp = (N*N + threads_per_block - 1) / threads_per_block;
    int num_blocks_inverse = (N + threads_per_block - 1) / threads_per_block;
    float alpha = 1.0f, beta = 0.0f;

    

    // Allocate memory for rowsums
    float *rowsums, *ones;
    cudaMallocManaged(&rowsums, N * sizeof(float));
    cudaMallocManaged(&ones, N * sizeof(float));
    
    
    



    exp_kernel <<<num_blocks_exp, threads_per_block>>>(input, input, N);
    ones_kernel <<<num_blocks_inverse, threads_per_block>>>(ones, N);

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    
    cublasSgemv(handle, CUBLAS_OP_T, N, N, &alpha, input, N, ones, 1, &beta, rowsums, 1);
    

    inverse_kernel<<<num_blocks_inverse, threads_per_block>>>(rowsums, rowsums, N);

    cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, N, N, input, N, rowsums, 1, output, N);

    cublasDestroy(handle);
    
   
}