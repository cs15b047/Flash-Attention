#include "softmax_cublas.cuh"
#include <stdio.h>
#include <cublas_v2.h>
#include "softmax.cuh"
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


__device__  void warpReduce(volatile float* sdata, int tid) {
    if(blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
    if(blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
    if(blockDim.x >= 16) sdata[tid] += sdata[tid +  8];
    if(blockDim.x >=  8) sdata[tid] += sdata[tid +  4];
    if(blockDim.x >=  4) sdata[tid] += sdata[tid +  2];
    if(blockDim.x >=  2) sdata[tid] += sdata[tid +  1];

}



__global__
void fused_softmax_kernel(float *input, float *output, int N, int dim,int row_per_block) {

    extern __shared__ float shMem[];
    float* sdata = shMem;
    float* temp = shMem + dim * row_per_block;

    #pragma unroll
    for(int i=0;i<row_per_block;i++){
        sdata[threadIdx.x+i*dim] = 0;
    }

    // loading data to shared memory
    if(threadIdx.x < dim){
        #pragma unroll
        for(int i=0;i<row_per_block;i++){
            sdata[threadIdx.x+i*dim] = (exp(input[(row_per_block*blockIdx.x+i) * dim + threadIdx.x]));
            temp[threadIdx.x+i*dim] =   (sdata[threadIdx.x+i*dim]);
        }
    }
    
    __syncthreads();
    int tid = threadIdx.x;

    // do reduction in shared mem
    if (dim >= 512) { if (tid < 256) { 
        
        #pragma unroll
        for(int i=0;i<row_per_block;i++){
                sdata[tid + i*dim] += sdata[tid + i*dim + 256]; 
        }} __syncthreads(); }
    
    
    if (dim >= 256) { if (tid < 128) { 
        #pragma unroll
        for(int i=0;i<row_per_block;i++){
                sdata[tid + i*dim] += sdata[tid + i*dim + 128];  
        }} __syncthreads(); }


    if (dim >= 128) { if (tid <  64) { 
        #pragma unroll
        for(int i=0;i<row_per_block;i++){
                sdata[tid + i*dim] += sdata[tid + i*dim + 64];  
        } }__syncthreads(); }
    
 
    if (tid < 32)
        #pragma unroll
        for(int i=0;i<row_per_block;i++){
            warpReduce(sdata, tid+i*dim);
        }

    
    // write result for this block to global mem

    #pragma unroll
    for(int i=0;i<row_per_block;i++){
       output[(row_per_block*blockIdx.x+i) * dim + threadIdx.x] = temp[tid+i*dim]/sdata[i*dim];
    }
	
    

}



__host__ void fused_softmax(float *input, float *output, int N, int dim,int row_per_block) {
    int threads_per_block = dim;
    int num_blocks_reduce = N;

    fused_softmax_kernel<<<num_blocks_reduce/row_per_block, threads_per_block, row_per_block * 2 * threads_per_block * sizeof(float)>>>(input, output, N, dim,row_per_block);
    cudaDeviceSynchronize();
    
    
}
