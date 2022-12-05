#include "self_attention_backward.h"
#include <bits/stdc++.h>

using namespace std;

//generate matrix multiplication kernel for genereic sizes with transpose in cpu
void matrix_mult(const float* A, const float* B, float* C, int M, int N, int K, bool transA,bool transB) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                float term1 = transA ? A[k * M + i] : A[i * K + k];
                float term2 = transB ? B[j * K + k] : B[k * N + j];
                sum += term1 * term2;
            }
            C[i * N + j] = sum;
        }
    }
}

void softmax_backward(const float* P, const float* dP, int N, float* dS) {
    for (int i = 0; i < N; i++) {
        float sum = 0;
        for(int k = 0; k < N; k++) {
            sum += dP[i * N + k] * P[i*N + k];
        }
        for (int j = 0; j < N; j++) {
            
            dS[i * N + j] = P[i * N + j] * (dP[i * N + j] - sum);
        }
    }
}

void self_attention_backward_cpu(const float *Q, const float *K, const float *V, const float *dO, const float *P, float* dP, float* dQ, float* dV, float* dK, float* dS, int N, int dim){
    matrix_mult(P, dO, dV, N, dim, N, true, false);
    matrix_mult(dO, V, dP, N, N, dim, false, true);
    softmax_backward(P, dP, N, dS);
    matrix_mult(dS, K, dQ, N, dim, N, false, false);
    matrix_mult(dS, Q, dK, N, dim, N, true, false);
}


