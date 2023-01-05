#include "batch_matmul.cuh"
#include <stdio.h>
using namespace std;

__global__ void matmul_tiled_coalesced_one_to_many_kernel(const float* A, const float* B, float* C, int M, int N, int K, const int BLOCK_SIZE) {
    // Block index
    int bx = blockIdx.x; //the B (and C) matrix sub-block column index
    int by = blockIdx.y; //the A (and C) matrix sub-block row index
    int bz  = blockIdx.z; 
    
    // Thread index
    int tx= threadIdx.x; //the column index in the sub-block
    int ty = threadIdx.y; //the row index in the sub-block
    // Index of the first sub-matrix of A processed by the block
    int aBegin= K * BLOCK_SIZE * by;
    // Index of the last sub-matrix of A processed by the block
    int aEnd= aBegin+ K - 1;
    // Step size used to iterate through the sub-matrices of A
    int aStep= BLOCK_SIZE;
    // Index of the first sub-matrix of B processed by the block
    int bBegin= BLOCK_SIZE * bx;
    // Step size used to iterate through the sub-matrices of B
    int bStep= BLOCK_SIZE * N;
    // The element of the block sub-matrix that is computed
    // by the thread
    float Csub0 = 0, Csub1 = 0, Csub2 = 0, Csub3 = 0; 

    extern __shared__ float shMem[];
    float* As = shMem;
    float* Bs = shMem + BLOCK_SIZE * BLOCK_SIZE;

    // Loop over all the sub-matrices (tiles) of A and B required to
    // compute the block sub-matrix; moving in A left to right in
    // a row, and in B from top to bottom in a column
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load tiles from global memory into shared memory; each
        // thread loads one element of the two tiles from A & B
        // utilize z index to load 4 tiles
        
        As[ty * BLOCK_SIZE + tx] = A[bz*M*K + a + K* ty + tx];
        As[ty * BLOCK_SIZE + tx + BLOCK_SIZE/2] = A[bz*M*K + a + K* ty + tx + BLOCK_SIZE/2];
        As[(ty + BLOCK_SIZE/2) * BLOCK_SIZE + tx] = A[bz*M*K +a + K* (ty + BLOCK_SIZE/2) + tx];
        As[(ty + BLOCK_SIZE/2) * BLOCK_SIZE + (tx + BLOCK_SIZE/2)] = A[bz*M*K + a + K* (ty + BLOCK_SIZE/2) + tx + BLOCK_SIZE/2];

       

        
        Bs[ty * BLOCK_SIZE + tx] = B[bz*N*K + b + N* ty + tx];
        Bs[ty * BLOCK_SIZE + tx + BLOCK_SIZE/2] = B[bz*N*K + b + N* ty + tx + BLOCK_SIZE/2];
        Bs[(ty + BLOCK_SIZE/2) * BLOCK_SIZE + tx] = B[bz*N*K + b + N* (ty + BLOCK_SIZE/2) + tx];
        Bs[(ty + BLOCK_SIZE/2) * BLOCK_SIZE + (tx + BLOCK_SIZE/2)] = B[bz*N*K + b + N * (ty + BLOCK_SIZE/2) + tx + BLOCK_SIZE/2];
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        // Each thread in this block computes one element
        // of the block sub-matrix (tile). Thread with indexes
        // ty and txcomputes in this tile the entry [ty][tx].
        for (int k = 0; k < BLOCK_SIZE; ++k){
            Csub0 += As[ty * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + tx];
            Csub1 += As[ty * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + tx + BLOCK_SIZE/2];
            Csub2 += As[(ty + BLOCK_SIZE/2) * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + tx];
            Csub3 += As[(ty + BLOCK_SIZE/2) * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + tx + BLOCK_SIZE/2];
            
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write the block sub-matrix to global memory;
    // each thread writes one element
    int c = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[bz*N*M +c + N * ty + tx] = Csub0;
    C[bz*N*M +c + N * ty + tx + BLOCK_SIZE/2] = Csub1;
    C[bz*N*M +c + N * (ty + BLOCK_SIZE/2) + tx] = Csub2;
    C[bz*N*M +c + N * (ty + BLOCK_SIZE/2) + tx + BLOCK_SIZE/2] = Csub3;
}


__host__ void batch_matmul(const float *A, const float *B, float *C, int batchsize, int M, int N, int K) {

    dim3 grid_dim((N + (2*32 - 1)) / (2 * 32), (M + (2*32 - 1)) / (2 * 32),1);
    dim3 block_dim(32, 32);
    int shared_mem_size = 2 * 64 * 64 * sizeof(float);

    for(int i = 0; i < batchsize; i++) {
        matmul_tiled_coalesced_one_to_many_kernel<<<grid_dim, block_dim, shared_mem_size>>>(A + i * M * K, B + i * K * N, C + i * M * N, M, N, K, 64);
        cudaDeviceSynchronize();
    }
   
}


__host__ void batch_matmul_single_kernel(const float *A, const float *B, float *C, int batchsize, int M, int N, int K) {
    
    dim3 grid_dim((N + (2*32 - 1)) / (2 * 32), (M + (2*32 - 1)) / (2 * 32), batchsize);
    dim3 block_dim(32, 32);
    int shared_mem_size = 2 * 64 * 64 * sizeof(float);

    matmul_tiled_coalesced_one_to_many_kernel<<<grid_dim, block_dim, shared_mem_size>>>(A, B, C, M, N, K, 64);
    cudaDeviceSynchronize();   
}


