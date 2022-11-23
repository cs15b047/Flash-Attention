
#include <cublas_v2.h>
#include <stdio.h>
#include "self_attention.cuh"
#include "softmax.cuh"

void print_matrix(const float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void matmul_serial(const float* Q, const float* K, int N, int dim) {
    float* output;
    cudaMallocManaged(&output, N * N * sizeof(float));
    cudaMemset(output, 0, N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < dim; k++) {
                output[i * N + j] += Q[i * dim + k] * K[j * dim + k];
            }
        }
    }
    print_matrix(output, N, N);
}


__host__ void self_attention(const float *Q, const float *K, const float *V, float *O, int N, int dim){
    float *intermediate, *softmax_result;
    cudaMallocManaged(&intermediate, N * N  * sizeof(float));
    cudaMemset(intermediate, 0, N * N  * sizeof(float));
    cudaMallocManaged(&softmax_result, N * N * sizeof(float));
    cudaMemset(softmax_result, 0, N * N * sizeof(float));

    matmul_serial(Q, K, N, dim);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                N, N, dim, &alpha, 
                K, dim, Q, dim, &beta, intermediate, N);
    cublasDestroy(handle);

    printf("Q:\n");
    print_matrix(Q, N, dim);
    printf("K:\n");
    print_matrix(K, N, dim);
    
    printf("Intermediate matrix: \n");
    print_matrix(intermediate, N, N);
    
    softmax(intermediate, softmax_result, N);
    
    printf("Softmax result: \n");
    print_matrix(softmax_result, N, N);

    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                dim, N, N, &alpha, 
                V, dim, softmax_result, N, &beta, O, dim);
    cublasDestroy(handle);

    printf("V:\n");
    print_matrix(V, N, dim);

    printf("Output matrix: \n");
    print_matrix(O, N, dim);
    

    
    
    

}
