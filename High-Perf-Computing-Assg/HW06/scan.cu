#include "scan.cuh"
#include <stdio.h>

// intra-block prefix sum
__global__ void hillis_steele(const float* input, float* output, int n) {
    extern __shared__ float shMem[];

    unsigned int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // load data into shared memory
    float* input_array = shMem;
    float* output_array = shMem + blockDim.x;

    // No work for out-of-bounds threads
    if(idx >= n) return;

    if(idx < n) input_array[tid] = input[idx];
    else input_array[tid] = 0;
    __syncthreads();

    // scan with doubling stride and swapping input/output
    for(int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = input_array[tid];
        if(tid >= stride) val += input_array[tid - stride];
        output_array[tid] = val;
        __syncthreads();
        
        // swap input/output
        float* tmp = input_array;
        input_array = output_array;
        output_array = tmp;
    }
    // after last iteration, output is in input_array
    output[idx] = input_array[tid];
}

// get prefix sum of block total sums into separate array
// Will only work for 1 block of threads
__global__ void interblock_prefix_sums(const float* input, float* output, int n, int blocks) {
    int tid = threadIdx.x;
    
    if(tid >= blocks) return;

    int last_idx = (tid + 1) * blockDim.x - 1; // last index in a block --> block prefix sum
    
    if(last_idx < n) output[tid] = input[last_idx];
    else output[tid] = input[n-1]; // different for the last block
}

__global__ void add_interblock_prefix_sums(float* input, float* inter_block_prefix_sum, int n, int blocks) {
    int tid = threadIdx.x;
    int blockid = blockIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if(blockid == 0) return;
    if(idx >= n) return;

    float value_for_block = inter_block_prefix_sum[blockid - 1];
    
    input[idx] += value_for_block;
}



__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block) {
    unsigned int blocks = (n + threads_per_block - 1) / threads_per_block;
    unsigned int shared_mem_size = 2 * threads_per_block * sizeof(float);
    float* inter_block_prefix_sum, *block_prefix_sum;
    cudaMallocManaged(&block_prefix_sum, blocks * sizeof(float));
    cudaMallocManaged(&inter_block_prefix_sum, blocks * sizeof(float));

    hillis_steele<<<blocks, threads_per_block, shared_mem_size>>>(input, output, n);
    cudaDeviceSynchronize();

    interblock_prefix_sums<<<1, threads_per_block>>>(output, block_prefix_sum, n, blocks);
    cudaDeviceSynchronize();

    hillis_steele<<<1, threads_per_block, shared_mem_size>>>(block_prefix_sum, inter_block_prefix_sum, blocks);
    cudaDeviceSynchronize();

    // add interblock prefix sum to each block
    add_interblock_prefix_sums<<<blocks, threads_per_block>>>(output, inter_block_prefix_sum, n, blocks);
    cudaDeviceSynchronize();
}