#include "softmax.cuh"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace std;

void print_matrix1(const float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

__device__  void warpReduce(volatile float* sdata, int tid) {
    if(blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
    if(blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
    if(blockDim.x >= 16) sdata[tid] += sdata[tid +  8];
    if(blockDim.x >=  8) sdata[tid] += sdata[tid +  4];
    if(blockDim.x >=  4) sdata[tid] += sdata[tid +  2];
    if(blockDim.x >=  2) sdata[tid] += sdata[tid +  1];

}

__global__ void exp_kernel(float *x, float* out, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * dim) {
        out[i] = exp(x[i]);
    }
}

__global__ void reduce1(float* input, float* output, int n) {
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
    for(int i = 0; i < blockDim.x; i++) {
        rowsum += shMem[i];
    }
    output[row] = rowsum;
}

__global__ void reduce2(float* input, float* output,  int dim) {
	extern __shared__ float sdata[];

	// each thread loads one element from global to shared mem
    int row = blockIdx.x;
	int tid = threadIdx.x;
    int idx = row * dim + tid;

	sdata[tid] = 0;
    if (tid < dim)
	{
		sdata[tid] = input[idx];
	}
    
	__syncthreads();

	// do reduction in shared mem
	// Sequential addressing. This solves the bank conflicts as
	//  the threads now access shared memory with a stride of one
	//  32-bit word (unsigned int) now, which does not cause bank 
	//  conflicts
	for (unsigned int s = dim/2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
        __syncthreads();
	}

    // write result for this block to global mem
	if (tid == 0)
		output[blockIdx.x] = sdata[0];
}

__global__ void reduce3(float* input, float* output,  int dim) {
	extern __shared__ float sdata[];

	// each thread loads one element from global to shared mem
    int row = blockIdx.x;
	int tid = threadIdx.x;
    int idx = row * dim + tid;

	sdata[tid] = 0;
    if (tid < dim)
	{
		sdata[tid] = input[idx];
	}
    
	__syncthreads();

	// do reduction in shared mem
	// Sequential addressing. This solves the bank conflicts as
	//  the threads now access shared memory with a stride of one
	//  32-bit word (unsigned int) now, which does not cause bank 
	//  conflicts
	if (dim >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (dim >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (dim >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
    
 
    if (tid < 32)
        warpReduce(sdata, tid);

    // write result for this block to global mem
	if (tid == 0)
		output[blockIdx.x] = sdata[0];
}

__global__ void reduce5(float* input, float* output,  int dim, int row_per_block) {
	extern __shared__ float sdata[]; 

	// each thread loads one element from global to shared mem
    int row = row_per_block *blockIdx.x;
	int tid = threadIdx.x;
    
    #pragma unroll
	for(int i=0;i<row_per_block;i++){
            sdata[tid+i*dim] = 0;
        }
    
    if (tid < dim)
	{   
        #pragma unroll
		for(int i=0;i<row_per_block;i++){
            sdata[tid+i*dim] = input[(row+i)*dim + tid];
        }
	}

    
    
	__syncthreads();

	// do reduction in shared mem
	// Sequential addressing. This solves the bank conflicts as
	//  the threads now access shared memory with a stride of one
	//  32-bit word (unsigned int) now, which does not cause bank 
	//  conflicts
    
	if (dim >= 512) { 
        if (tid < 256) { 
            #pragma unroll
            for(int i=0;i<row_per_block;i++){
                sdata[tid+i*dim] += sdata[(tid+i*dim) + 256];}}
        __syncthreads(); }

   
    if (dim >= 256) { if (tid < 128) { 
        #pragma unroll
        for(int i=0;i<row_per_block;i++){
            sdata[tid+i*dim] += sdata[(tid+i*dim) + 128]; 
        } __syncthreads(); }}

    //print dim
    
   

    if (dim >= 128) { if (tid <  64) { 
        #pragma unroll
        for(int i=0;i<row_per_block;i++){
            sdata[tid+i*dim] += sdata[(tid+i*dim) + 64]; 
        } 
        
        __syncthreads(); }}

      
    if (tid < 32)
        #pragma unroll
        for(int i=0;i<row_per_block;i++){
            warpReduce(sdata, tid+i*dim);
        }
    

    // write result for this block to global mem
	if (tid == 0)
        #pragma unroll
		for(int i=0;i<row_per_block;i++){
            output[row+i] = sdata[i*dim];
        }

}

__device__ float cu_reduce_warp(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__global__ void reduce4(float const *inputs, float *outputs, int dim) {
    float sum = 0;
    // for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    //         i < input_size; 
    //         i += blockDim.x * gridDim.x)
    //     sum += inputs[i];

    int row = blockIdx.x;
    int i = row * dim + threadIdx.x;
    sum = inputs[i];


    __shared__ float shared[32];
    unsigned int lane = threadIdx.x % warpSize;
    unsigned int wid = threadIdx.x / warpSize;

    sum = cu_reduce_warp(sum);
    if (lane == 0)
        shared[wid] = sum;

    // Wait for all partial reductions
    __syncthreads();

    sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0)
        sum = cu_reduce_warp(sum);

    if (threadIdx.x == 0)
        outputs[blockIdx.x] = sum;
}

__global__ void normalize(float* input, float* output, float* rowsums, int n, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shMem[];
    int row = idx / dim;
    int rowsum_val = rowsums[row];
    
    // Normalize each element with rowsums
    if (idx < n * dim) {
        output[idx] = input[idx] / rowsum_val;
    }
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

/*
Iteration 1:
a) exp kernel, b) Reduce kernel, c) Divide kernel
*/

__host__ void fused_softmax(float *input, float *output, int N, int dim,int row_per_block) {
    int threads_per_block = dim;
    int num_blocks_reduce = N;

    fused_softmax_kernel<<<num_blocks_reduce/row_per_block, threads_per_block, row_per_block * 2 * threads_per_block * sizeof(float)>>>(input, output, N, dim,row_per_block);
    cudaDeviceSynchronize();
    
    
}

__host__ void softmax1(float *input, float *output, int N, int dim) {
    int threads_per_block = dim;
    int num_blocks_reduce = N;
    int num_blocks_exp = (N * dim + threads_per_block - 1) / threads_per_block;
    int num_blocks_normalize = num_blocks_reduce;

    // Allocate memory for rowsums
    float *rowsums, *input_exp;
    cudaMallocManaged(&rowsums, N * sizeof(float));
    cudaMallocManaged(&input_exp, N * dim * sizeof(float));

    // a) exp kernel
    exp_kernel<<<num_blocks_exp, threads_per_block>>>(input, input_exp, N, dim);

    cudaDeviceSynchronize();
    // print_matrix1(input_exp, N, dim);

    // b) Reduce kernel
    // int row_per_block = 24;
    // reduce5<<<num_blocks_reduce/row_per_block, threads_per_block, row_per_block * threads_per_block * sizeof(float)>>>(input_exp, rowsums, dim,row_per_block);
    // cudaDeviceSynchronize();
    // b) Reduce kernel
    reduce2<<<num_blocks_reduce, threads_per_block, threads_per_block * sizeof(float)>>>(input_exp, rowsums, dim);
    cudaDeviceSynchronize();
    // print_matrix1(rowsums, N, 1);
    
    // c) Divide kernel
    normalize<<<num_blocks_normalize, threads_per_block>>>(input_exp, output, rowsums, N,dim);
    cudaDeviceSynchronize();


    // print_matrix1(output, N, dim);
    

}

__host__ void softmax2(float *input, float *output, int N, int dim) {
    int threads_per_block = dim;
    int num_blocks_reduce = N;
    int num_blocks_exp = (N * dim + threads_per_block - 1) / threads_per_block;
    int num_blocks_normalize = num_blocks_reduce;

    // Allocate memory for rowsums
    float *rowsums, *input_exp;
    cudaMallocManaged(&rowsums, N * sizeof(float));
    cudaMallocManaged(&input_exp, N * dim * sizeof(float));

    // a) exp kernel
    exp_kernel<<<num_blocks_exp, threads_per_block>>>(input, input_exp, N, dim);

    cudaDeviceSynchronize();
    // print_matrix1(input_exp, N, dim);

    // b) Reduce kernel
    // int row_per_block = 24;
    // reduce5<<<num_blocks_reduce/row_per_block, threads_per_block, row_per_block * threads_per_block * sizeof(float)>>>(input_exp, rowsums, dim,row_per_block);
    // cudaDeviceSynchronize();
    // b) Reduce kernel
    reduce3<<<num_blocks_reduce, threads_per_block, threads_per_block * sizeof(float)>>>(input_exp, rowsums, dim);
    cudaDeviceSynchronize();
    // print_matrix1(rowsums, N, 1);
    
    // c) Divide kernel
    normalize<<<num_blocks_normalize, threads_per_block>>>(input_exp, output, rowsums, N,dim);
    cudaDeviceSynchronize();

    // print_matrix1(output, N, dim);
    
}


__host__ void softmax3(float *input, float *output, int N, int dim) {
    int threads_per_block = dim;
    int num_blocks_reduce = N;
    int num_blocks_exp = (N * dim + threads_per_block - 1) / threads_per_block;
    int num_blocks_normalize = num_blocks_reduce;

    // Allocate memory for rowsums
    float *rowsums, *input_exp;
    cudaMallocManaged(&rowsums, N * sizeof(float));
    cudaMallocManaged(&input_exp, N * dim * sizeof(float));

    // a) exp kernel
    exp_kernel<<<num_blocks_exp, threads_per_block>>>(input, input_exp, N, dim);

    cudaDeviceSynchronize();
    // print_matrix1(input_exp, N, dim);

    // b) Reduce kernel
    // int row_per_block = 24;
    // reduce5<<<num_blocks_reduce/row_per_block, threads_per_block, row_per_block * threads_per_block * sizeof(float)>>>(input_exp, rowsums, dim,row_per_block);
    // cudaDeviceSynchronize();
    // b) Reduce kernel
    reduce4<<<num_blocks_reduce, threads_per_block, threads_per_block * sizeof(float)>>>(input_exp, rowsums, dim);
    cudaDeviceSynchronize();
    // print_matrix1(rowsums, N, 1);
    
    // c) Divide kernel
    normalize<<<num_blocks_normalize, threads_per_block>>>(input_exp, output, rowsums, N,dim);
    cudaDeviceSynchronize();

    
}




