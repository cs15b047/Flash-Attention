#include "matmul.cuh"
#include <stdio.h>
#include <type_traits>

template <typename T>
__global__ void matmul_kernel3(const T *A, const T *B, T *C, int n, int block_dim) {
    int bi = blockIdx.y, bj = blockIdx.x, tx = threadIdx.y, ty = threadIdx.x;
    int cx = bi * block_dim + tx, cy = bj * block_dim + ty;
    int grid_dim = gridDim.x;

    extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
    T *shMem = reinterpret_cast<T *>(my_smem);
    T* sA = shMem;
    T* sB = shMem + block_dim * block_dim;

    T c = 0;
    for(int block_idx = 0; block_idx < grid_dim; block_idx++) {
        // load a tile of A and B into shared memory
        int ax = bi * block_dim + tx, ay = block_idx * block_dim + ty;
        int bx = block_idx * block_dim + tx, by = bj * block_dim + ty;

        int sh_mem_addr = tx * block_dim + ty;
        
        if(ax < n && ay < n) sA[sh_mem_addr] = A[ax * n + ay];
        else sA[sh_mem_addr] = 0;
        if(bx < n && by < n) sB[sh_mem_addr] = B[bx * n + by];
        else sB[sh_mem_addr] = 0;
        __syncthreads();

        
        for(int k = 0; k < block_dim; k++) {
            c += sA[tx * block_dim + k] * sB[k * block_dim + ty];
        }
        __syncthreads();
    }
    
    if(cx >= n || cy >= n) return;

    C[cx * n + cy] = c;
}

// basic matrix multiplication
__global__ void matmul_kernel1(const float *A, const float *B, float *C, int n) {
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

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){
    dim3 threads(block_dim, block_dim);
    dim3 blocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    unsigned int shared_mem_size = 2 * (block_dim * block_dim) * sizeof(int);
    matmul_kernel3<int><<<blocks, threads, shared_mem_size>>>(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
    dim3 threads(block_dim, block_dim);
    dim3 blocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    unsigned int shared_mem_size = 2 * (block_dim * block_dim) * sizeof(float);
    matmul_kernel3<float><<<blocks, threads, shared_mem_size>>>(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    dim3 threads(block_dim, block_dim);
    dim3 blocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    unsigned int shared_mem_size = 2 * (block_dim * block_dim) * sizeof(double);
    matmul_kernel3<double><<<blocks, threads, shared_mem_size>>>(A, B, C, n, block_dim);
}