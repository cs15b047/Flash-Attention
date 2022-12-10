#include "self_attention_backward.cuh"
#include "cublas_v2.h"

__global__ void elementwise_product(const float *P_, const float *dP_, float* out, float *dS_, const int N) {
    int idx1 = blockIdx.x;
    
    const float* P = P_ + idx1 * N * N, *dP = dP_ + idx1 * N * N;
    float *dS = dS_ + idx1 * N * N;
    
    // only 1st block will populate ones vector
    if(idx1 == 0) {
        for(int idx2 = threadIdx.x; idx2 < N; idx2 += blockDim.x) {
            out[idx2] = 1;
        }
        __syncthreads(); // Required ??
    }

    for(int idx2 = threadIdx.x; idx2 < N * N; idx2 += blockDim.x) {
        dS[idx2] = P[idx2] * dP[idx2];
    }
    __syncthreads();
}

__global__ void subtraction(const float* P_, const float* rowsums_, float* dS_, int N) {
    int idx1 = blockIdx.x;
    const float *P = P_ + idx1 * N * N, *rowsums = rowsums_ + idx1 * N;
    float* dS = dS_ + idx1 * N * N;
    for(int idx2 = threadIdx.x; idx2 < N * N; idx2 += blockDim.x) {
        dS[idx2] = dS[idx2] - P[idx2] * rowsums[idx2 / N];
    }
    __syncthreads();
}

__global__ void rowwise_sum(const float* dS_, float* rowsums, int N) {
    int idx1 = blockIdx.x;
    const float* dS = dS_ + idx1 * N * N;
    float* rowsum = rowsums + idx1 * N;
    
    // Each thread computes sum of a row
    for (int idx2 = threadIdx.x; idx2 < N; idx2 += blockDim.x) {
        float sum = 0;
        for(int idx3 = 0; idx3 < N; idx3++) {
            sum += dS[idx2 * N + idx3];
        }
        rowsum[idx2] = sum;
    }
    __syncthreads();
}


__host__ void softmax_backward(const float *P, const float* dP, float* dS, int N, int batch_size, int num_heads) {
    int threads = 1024;
    int blocks = batch_size * num_heads;
    float *rowsums, *ones;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaMallocManaged(&rowsums, batch_size * num_heads * N * sizeof(float));
    cudaMallocManaged(&ones, N * sizeof(float));

    elementwise_product<<<blocks, threads>>>(P, dP, ones, dS, N);
    cudaDeviceSynchronize();

    rowwise_sum<<<blocks, threads>>>(dS, rowsums, N);
    cudaDeviceSynchronize();

    cublasHandle_t handle;
    cublasCreate(&handle);
    // cublasSgemvBatched(handle, 
    //     CUBLAS_OP_T, 
    //     N, N, 
    //     &alpha, 
    //     dS, N, 
    //     N*N, 
    //     ones, 1, 
    //     0, // stride for ones --> same data --> 0 stride ?
    //     &beta, 
    //     rowsums, 1,
    //     N, // stride for rowsums
    //     batch_size * num_heads);
    // cublasDestroy(handle);

    subtraction<<<blocks, threads>>>(P, (const float*)rowsums, dS, N);
    cudaDeviceSynchronize();

    cudaFree(rowsums);
    cudaFree(ones);
}