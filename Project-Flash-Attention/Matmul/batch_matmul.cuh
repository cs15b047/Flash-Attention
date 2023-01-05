#ifndef BATCH_MATMUL_H
#define BATCH_MATMUL_H
#include <cuda.h>
#include <cuda_runtime_api.h>


void batch_matmul(const float *A, const float *B, float *C, int batchsize, int M, int N, int K);
void batch_matmul_single_kernel(const float *A, const float *B, float *C, int batchsize, int M, int N, int K);

#endif