#include "self_attention_backward.cuh"
#include "cublas_v2.h"

__global__ void elementwise_product(const float *P, const float *dP, float* out, float *dS, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < N) {
        out[idx]= 1;
    }

    if (idx < N * N) {
        dS[idx] = P[idx] * dP[idx];    
    }
}

__global__ void subtraction(const float* P, const float* rowsums, float* dS, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N * N) { // dS initially has P .* dP
        dS[idx] = dS[idx] - P[idx] * rowsums[idx / N];
    }
}

__host__ void softmax_backward(const float *P, const float* dP, float* dS, int N) {
    int threads = 1024;
    int blocks = (N * N + threads - 1) / threads;
    float *rowsums, *ones;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaMallocManaged(&rowsums, N * sizeof(float));
    cudaMallocManaged(&ones, N * sizeof(float));

    elementwise_product<<<blocks, threads>>>(P, dP, ones, dS, N);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemv(handle, CUBLAS_OP_T, N, N, &alpha, dS, N, ones, 1, &beta, rowsums, 1);
    cublasDestroy(handle);

    subtraction<<<blocks, threads>>>(P, rowsums, dS, N);
    cudaDeviceSynchronize();

    cudaFree(rowsums);
    cudaFree(ones);
}