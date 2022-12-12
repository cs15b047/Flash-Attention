
#include <cublas_v2.h>
#include <stdio.h>
#include "self_attention_backward.cuh"


__host__ void self_attention_backward(const float *Q, const float *K, const float *V, const float *dO, const float *P, 
                                 float* dP, float* dQ, float* dV, float* dK, float* dS, float* rowsums, int N, int dim, int batch_size, int num_heads) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    // dV = P^T * dO
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                dim, N, N, &alpha,
                dO, dim, N*dim, P, N, N*N, &beta, dV, dim, N*dim,
                batch_size * num_heads);

    // // dP = dO * V^T
    cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                N, N, dim, &alpha, 
                V, dim, N*dim, dO, dim, N*dim, &beta, dP, N, N*N,
                batch_size * num_heads);
    
    // dS = softmax backward
    softmax_backward(P, dP, dS, rowsums, N, batch_size, num_heads);

    // dQ = dS * K
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                dim, N, N, &alpha, 
                K, dim, N*dim, dS, N, N*N, &beta, dQ, dim, N*dim,
                batch_size * num_heads);

    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                dim, N, N, &alpha,
                Q, dim, N*dim, dS, N, N*N, &beta, dK, dim, N*dim,
                batch_size * num_heads);
    
    cublasDestroy(handle);
}
