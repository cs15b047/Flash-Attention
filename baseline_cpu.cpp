#include "self_attention.h"
#include <bits/stdc++.h>

using namespace std;

void self_attention_cpu(const float *Q, const float *K, const float *V, float *O, int N, int dim) {
    float* QKt = new float[N * N];
    float* QKt_softmax = new float[N * N];

    memset(QKt, 0, N * N * sizeof(float));
    memset(O, 0, N * dim * sizeof(float));

    // Step 1: Q * transpose(K)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < dim; k++) {
            for (int j = 0; j < N; j++) {
                QKt[i * N + j] += Q[i * dim + k] * K[j * dim + k];
            }
        }
    }

    // Step 2: Softmax(Q * transpose(K))
    for (int i = 0; i < N; i++) {
        float sum = 0;
        for (int j = 0; j < N; j++) {
            QKt_softmax[i * N + j] = exp(QKt[i * N + j]);
            sum += QKt_softmax[i * N + j];
        }
        for (int j = 0; j < N; j++) {
            QKt_softmax[i * N + j] /= sum;
        }
    }

    // Step 3: Softmax(Q * transpose(K)) * V
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < dim; k++) {
                O[i * dim + k] += QKt_softmax[i * N + j] * V[j * dim + k];
            }
        }
    }

    delete[] QKt;
    delete[] QKt_softmax;
}