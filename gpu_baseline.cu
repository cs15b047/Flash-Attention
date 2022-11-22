
#include <cublas_v2.h>
#include <stdio.h>
#include "self_attention.cuh"


__host__ void self_attention(const float *Q, const float *K, const float *V, float *O, int N, int dim){
    float *intermediate;
    cudaMallocManaged(&intermediate, N * N  * sizeof(float));
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                N, N, dim, &alpha, 
                K, N, Q, N, &beta, intermediate, N);
    cublasDestroy(handle);

    

    
    
    

}
