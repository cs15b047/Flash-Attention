
#include <cublas_v2.h>
#include <stdio.h>
#include "self_attention.cuh"
#include "softmax_cublas.cuh"


void print_matrix(const float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

__host__ void self_attention(const float *Q, const float *K, const float *V, float *intermediate, float *softmax_result, float *O, int N, int dim){
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                N, N, dim, &alpha, 
                K, dim, Q, dim, &beta, intermediate, N);
    
    softmax_cublas(intermediate, softmax_result, N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                dim, N, N, &alpha, 
                V, dim, softmax_result, N, &beta, O, dim);
    cublasDestroy(handle);
    cudaDeviceSynchronize();
}


