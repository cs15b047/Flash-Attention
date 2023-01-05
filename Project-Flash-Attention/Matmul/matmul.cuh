#ifndef MATMUL_H
#define MATMUL_H
#include <cuda.h>
#include <cuda_runtime_api.h>


void matmul(const float *A, const float *B, float *C, int M, int N, int K);

void matmul_tiled(const float *A, const float *B, float *C, int M, int N, int K);

void matmul_tiled_coalesced(const float *A, const float *B, float *C, int M, int N, int K);

void matmul_tiled_coalesced_one_to_many(const float *A, const float *B, float *C, int M, int N, int K);

#endif