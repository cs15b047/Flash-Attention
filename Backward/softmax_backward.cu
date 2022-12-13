#include "self_attention_backward.cuh"
#include "cublas_v2.h"
#include <stdio.h>
#include <cassert>

__global__ void elementwise_product(const float *P_, const float *dP_, float *dS_, const int N) {
    int idx1 = blockIdx.x;
    
    const float* P = P_ + idx1 * N * N, *dP = dP_ + idx1 * N * N;
    float *dS = dS_ + idx1 * N * N;

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


__global__ void fused_softmax(const float *P_, const float* dP_, float* dS_, int N, int batch_size, int num_heads) {
    int idx1 = blockIdx.x;
    const float* P = P_ + idx1 * N * N, *dP = dP_ + idx1 * N * N;
    float* dS = dS_ + idx1 * N * N;

    extern __shared__ float shMem[];
    float* shP = shMem;
    float* shPdP = shP + N;
    float* temp_sums = shPdP + N;

    // Each thread computes sum of a row
    for(int i = 0; i < N; i++) {
        // load row to shMem
        shP[threadIdx.x] = P[i * N + threadIdx.x];
        shPdP[threadIdx.x] = shP[threadIdx.x] * dP[i * N + threadIdx.x];
        __syncthreads();
        temp_sums[threadIdx.x] = shPdP[threadIdx.x];
        __syncthreads();

        // reduce
        for(int s = blockDim.x / 2; s > 0; s >>= 1) {
            if(threadIdx.x < s) {
                temp_sums[threadIdx.x] += temp_sums[threadIdx.x + s];
            }
            __syncthreads();
        }
        float rowsum = temp_sums[0];

        // write to dS
        dS[i * N + threadIdx.x] = shPdP[threadIdx.x] - shP[threadIdx.x] * rowsum;
        __syncthreads();
    }
}

__device__ void warpReduce(volatile float* sdata) {
    sdata[threadIdx.x] += sdata[threadIdx.x + 32];
    sdata[threadIdx.x] += sdata[threadIdx.x + 16];
    sdata[threadIdx.x] += sdata[threadIdx.x + 8];
    sdata[threadIdx.x] += sdata[threadIdx.x + 4];
    sdata[threadIdx.x] += sdata[threadIdx.x + 2];
    sdata[threadIdx.x] += sdata[threadIdx.x + 1];
}

// Assume N = 1024 = blockDim.x

__host__ void softmax_backward2(const float *P, const float* dP, float* dS, float* rowsums, int N, int batch_size, int num_heads) {
    int threads = 1024;
    int blocks = batch_size * num_heads;
    int shared_memory_size = 6 * N * sizeof(float);

    fused_softmax<<<blocks, threads, shared_memory_size>>>(P, dP, dS, N, batch_size, num_heads);
}

__host__ void softmax_backward1(const float *P, const float* dP, float* dS, float* rowsums, int N, int batch_size, int num_heads) {
    int threads = 1024;
    int blocks = batch_size * num_heads;

    // dS = P .* dP
    elementwise_product<<<blocks, threads>>>(P, dP, dS, N);
    cudaDeviceSynchronize();

    // rowsums = rowwise_sum(dS)
    rowwise_sum<<<blocks, threads>>>(dS, rowsums, N);
    cudaDeviceSynchronize();

    // dS = dS - P .* rowsums
    subtraction<<<blocks, threads>>>(P, (const float*)rowsums, dS, N);
}

__host__ void softmax_backward(const float *P, const float* dP, float* dS, float* rowsums, int N, int batch_size, int num_heads) {
    assert(N == 1024);
    // softmax_backward1(P, dP, dS, rowsums, N, batch_size, num_heads);
    softmax_backward2(P, dP, dS, rowsums, N, batch_size, num_heads);
    cudaDeviceSynchronize();
}