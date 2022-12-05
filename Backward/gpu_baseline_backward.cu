
#include <cublas_v2.h>
#include <stdio.h>
#include "self_attention_backward.cuh"


__host__ void self_attention_backward(const float *Q, const float *K, const float *V, const float *dO, const float *P, 
                                 float* dP, float* dQ, float* dV, float* dK, float* dS, int N, int dim) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    // dV = P^T * dO
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                dim, N, N, &alpha,
                dO, dim, P, N, &beta, dV, dim);
    

    // // dP = dO * V^T
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                N, N, dim, &alpha, 
                V, dim, dO, dim, &beta, dP, N);
    
    // dS = softmax backward
    softmax_backward(P, dP, dS, N);

    // dQ = dS * K
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                dim, N, N, &alpha, 
                K, dim, dS, N, &beta, dQ, dim);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                dim, N, N, &alpha,
                Q, dim, dS, N, &beta, dK, dim);
    
    cublasDestroy(handle);
}
