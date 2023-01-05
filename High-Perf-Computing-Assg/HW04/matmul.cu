#include "matmul.cuh"

__global__ 
void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / n, col = idx % n;
    if(idx < n * n) {
        float sum = 0;
        for(int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[idx] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    unsigned int blocks = (n * n + threads_per_block - 1) / threads_per_block;
    matmul_kernel<<<blocks, threads_per_block>>>(A, B, C, n);
}