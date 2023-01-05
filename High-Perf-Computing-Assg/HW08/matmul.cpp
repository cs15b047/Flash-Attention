#include "matmul.h"

void mmul(const float* A, const float* B, float* C, const size_t n_){
    int n = n_;
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < n; i++) {
        for(int k = 0; k < n; k++) {
            for(int j = 0; j < n; j++) {
                // C(i, j) = C(i, j) + A(i, k) * B(k, j)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}