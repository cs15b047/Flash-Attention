#include "mmul.h"
#include <stdio.h>

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n) {
    cublasStatus_t status;
    float alpha = 1.0f;
    float beta = 0.0f;
    status = cublasSgemm(handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        n, n, n, 
        &alpha, A, n, B, n, &beta, C, n);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!! kernel execution error !!!\n");
    }
}