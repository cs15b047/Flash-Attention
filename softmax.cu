#include "softmax.cuh"
#include <stdio.h>

void print_matrix1(const float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

__global__
void exp_kernel(float *x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * n) {
        out[i] = exp(x[i]);
    }
}

__global__
void reduce(float* input, float* output, int n) {
    // Each block reduces a row of elements
    int row = blockIdx.x;
    int elements_per_thread = (n + blockDim.x - 1) / blockDim.x;
    int start = threadIdx.x * elements_per_thread;
    int end = min(start + elements_per_thread, n);
    start += row * n;
    end += row * n;

    // Load row into shared memory
    extern __shared__ float shMem[];
    float sum = 0;
    for (int i = start; i < end; i++) {
        sum += input[i];
    }

    shMem[threadIdx.x] = sum;
    __syncthreads();

    // Reduce shared memory in single thread and write to output in global memory
    if(threadIdx.x > 0) return;
    float rowsum = 0;
    for (int i = 0; i < blockDim.x; i++) {
        rowsum += shMem[i];
    }
    output[row] = rowsum;
}

__global__
void normalize(float* input, float* output, float* rowsums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float shMem[];

    // Load rowsums into shared memory
    if(idx < n) shMem[idx] = rowsums[idx];
    __syncthreads();

    // Normalize each element with rowsums
    if (idx < n * n) {
        output[idx] = input[idx] / rowsums[idx / n];
    }
}

/*
Iteration 1:
a) exp kernel, b) Reduce kernel, c) Divide kernel
*/

__host__ void softmax(float *input, float *output, int N) {
    int threads_per_block = 1024;
    int num_blocks_reduce = N;
    int num_blocks_exp = (N + threads_per_block - 1) / threads_per_block, num_blocks_normalize = num_blocks_exp;

    // Allocate memory for rowsums
    float *rowsums, *input_exp;
    cudaMallocManaged(&rowsums, N * sizeof(float));
    cudaMallocManaged(&input_exp, N * N * sizeof(float));

    // a) exp kernel
    exp_kernel<<<num_blocks_exp, threads_per_block>>>(input, input_exp, N);

    cudaDeviceSynchronize();
    print_matrix1(input_exp, N, N);

    // b) Reduce kernel
    reduce<<<num_blocks_reduce, threads_per_block, threads_per_block * sizeof(float)>>>(input_exp, rowsums, N);
    cudaDeviceSynchronize();
    print_matrix1(rowsums, N, 1);

    // c) Divide kernel
    normalize<<<num_blocks_normalize, threads_per_block, N * sizeof(float)>>>(input_exp, output, rowsums, N);
    cudaDeviceSynchronize();
    print_matrix1(output, N, N);
}