#include "stencil.cuh"

__global__ 
void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory for the mask, image portion for block and output portion for block
    extern __shared__ float shared_memory[];
    float* shared_mask = shared_memory;
    float* shared_image = &shared_mask[2*R + 1];
    float* shared_output = &shared_image[blockDim.x + 2*R];

    // 1. first 2R + 1 threads store the mask to shared memory
    // 2. All threads store image to shared memory
    // 3. First R threads store outsides of the image region to shared memory

    // copy mask to shared memory
    if(threadIdx.x < 2*R + 1) {
        shared_mask[threadIdx.x] = mask[threadIdx.x];
    }

    // copy image to shared memory
    if(idx < n) {
        shared_image[R + threadIdx.x] = image[idx];
        // copy previous and next image values to shared memory which are used in convolution
        if(threadIdx.x < R) {
            float prev_image_ele = 1.0, next_image_ele = 1.0;
            int prev_idx = idx - (int)R, next_idx = idx + blockDim.x;
            // handle out of bounds cases for initialization
            if(prev_idx >= 0) prev_image_ele = image[prev_idx];
            if(next_idx < n) next_image_ele = image[next_idx];
            
            shared_image[threadIdx.x] = prev_image_ele;

            // If n doesn't end in this block, then put the next block elements after R + Blockdim
            // either n - blockDim.x*blockIdx.x or blockDim.x
            if(n - blockDim.x * blockIdx.x < blockDim.x) {
                int img_idx = R + n - blockDim.x * blockIdx.x + threadIdx.x;
                shared_image[img_idx] = next_image_ele;
            } else {
                shared_image[blockDim.x + R + threadIdx.x] = next_image_ele;
            }
        }
    }

    __syncthreads();

    if(idx >= n) return;

    float sum = 0.0;
    for(int i = 0; i <= 2*R ; i++) {
        sum += shared_mask[i] * shared_image[threadIdx.x + i];
    }

    // write output to shared memory
    shared_output[threadIdx.x] = sum;

    // write from shared memory to global memory
    output[idx] = shared_output[threadIdx.x];
}

__host__ 
void stencil(const float* image, const float* mask, float* output, 
    unsigned int n, unsigned int R, unsigned int threads_per_block) {
    unsigned int blocks = (n + threads_per_block - 1) / threads_per_block;
    unsigned int mask_size = (2*R + 1);
    unsigned int image_size_in_shared_mem = (threads_per_block + 2*R);
    unsigned int output_size_in_shared_mem = threads_per_block;
    unsigned int shared_mem_size = (mask_size + image_size_in_shared_mem + output_size_in_shared_mem) * sizeof(float);
    
    stencil_kernel<<<blocks, threads_per_block, shared_mem_size>>>(image, mask, output, n, R);
}