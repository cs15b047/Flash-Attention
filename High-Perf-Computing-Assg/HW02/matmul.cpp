#include "matmul.h"

// Assume C is initialized to 0
void mmul1(const double* A, const double* B, double* C, const unsigned int dim){
    int n = dim;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            for(int k = 0; k < n; k++) {
                // C(i, j) = C(i, j) + A(i, k) * B(k, j)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}

void mmul2(const double* A, const double* B, double* C, const unsigned int dim){
    int n = dim;
    for(int i = 0; i < n; i++) {
        for(int k = 0; k < n; k++) {
            for(int j = 0; j < n; j++) {
                // C(i, j) = C(i, j) + A(i, k) * B(k, j)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}

void mmul3(const double* A, const double* B, double* C, const unsigned int dim){
    int n = dim;
    for(int j = 0; j < n; j++) {
        for(int k = 0; k < n; k++) {
            for(int i = 0; i < n; i++) {
                // C(i, j) = C(i, j) + A(i, k) * B(k, j)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int dim){
    int n = dim;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            for(int k = 0; k < n; k++) {
                // C(i, j) = C(i, j) + A(i, k) * B(k, j)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}