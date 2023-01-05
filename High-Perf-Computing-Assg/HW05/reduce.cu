#include "reduce.cuh"

// Common: sum up 1 block and get 1 element
// iteration 0: Using atomic adds (all in global memory) - not tried

// iteration 3: iteration 2 + Double blockDim: 34-35 milliseconds
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
    int idx = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    int output_address = blockIdx.x;
    int tid = threadIdx.x;

    if(idx >= n) return;

    extern __shared__ float sdata[];
    sdata[tid] = g_idata[idx];
    if(idx + blockDim.x < n) sdata[tid] += g_idata[idx + blockDim.x];
    __syncthreads();
    
    for(unsigned int active_threads = blockDim.x/2; active_threads > 0; active_threads >>= 1){
        if(tid < active_threads && tid + active_threads < n){
            sdata[tid] += sdata[tid + active_threads];
        }
        __syncthreads();
    }

    if(tid == 0) g_odata[output_address] = sdata[0];
}


// iteration 2: For half threads (tid < blockDim/2): 70 milliseconds
// output[tid] = shmem[tid] + shmem[tid + blockDim/2]
__global__ void reduce_kernel2(float *g_idata, float *g_odata, unsigned int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_address = blockIdx.x;
    int tid = threadIdx.x;

    if(idx >= n) return;

    extern __shared__ float sdata[];
    sdata[tid] = g_idata[idx];
    __syncthreads();
    
    for(unsigned int active_threads = blockDim.x/2; active_threads > 0; active_threads >>= 1){
        if(tid < active_threads && tid + active_threads < n){
            sdata[tid] += sdata[tid + active_threads];
        }
        __syncthreads();
    }

    if(tid == 0) g_odata[output_address] = sdata[0];
}

// iteration 1: Using atomic adds (all in shared memory) -  12.5 - 13 seconds
__global__ void reduce_kernel1(float *g_idata, float *g_odata, unsigned int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_address = blockIdx.x;
    int tid = threadIdx.x;

    if(idx >= n) return;

    // initialize shared memory: each thread loads one element from global to shared mem. Last element for result
    extern __shared__ float sdata[];
    sdata[tid] = g_idata[idx];
    if(tid == 0) sdata[blockDim.x] = 0;
    __syncthreads();

    // g_odata[output_address] += g_idata[idx];
    atomicAdd(&sdata[blockDim.x], sdata[tid]);
    __syncthreads();

    if(tid == 0) g_odata[output_address] = sdata[blockDim.x];
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block){

    unsigned int blocks;
    unsigned int num_elements = N;

    float *input_arr = *input, *output_arr = *output;

    while(true) {
        blocks = (num_elements + (2 * threads_per_block) - 1) / (2 * threads_per_block);
        unsigned int shared_memory_size = (threads_per_block + 1) * sizeof(float);
        
        reduce_kernel<<<blocks, threads_per_block, shared_memory_size>>>(input_arr, output_arr, num_elements);
        cudaDeviceSynchronize();

        if(blocks == 1) { // reduced to 1 element
            break;
        }
        
        // swap pointers to input and output arrays
        float *temp = input_arr;
        input_arr = output_arr;
        output_arr = temp;

        // 1 block gets converted to 1 element
        num_elements = blocks;
    }

    // if sum not present in output_arr, copy it to output_arr
    if(output_arr == *input){
        cudaMemcpy(*output, *input, sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
}