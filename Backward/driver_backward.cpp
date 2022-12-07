#include "self_attention_backward.h"
#include "self_attention_backward.cuh"
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


using namespace std;

float rand_float() {
    return rand() % 4;
    // float r = 0.1 * (float)rand() / RAND_MAX;
    // return 2 * r - 1;
}

void print_matrix(const float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int N = stoi(argv[1]);
    int dim = stoi(argv[2]);

    float *Q, *K, *V, *O, *P, *O_cpu;
    float *dQ, *dK, *dV, *dS, *dO, *dP;
    cudaMallocManaged((void **)&Q, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&K, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&V, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&O, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&P, sizeof(float) * N * N);
    cudaMallocManaged((void **)&dQ, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&dK, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&dV, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&dS, sizeof(float) * N * N);
    cudaMallocManaged((void **)&dO, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&dP, sizeof(float) * N * N);

    O_cpu = new float[N * dim];

    generate(Q, Q + N * dim, rand_float);
    generate(K, K + N * dim, rand_float);
    generate(V, V + N * dim, rand_float);
    generate(dO, dO + N * dim, rand_float);
    generate(P, P + N * N, rand_float);

    float cpu_time_ms = 0.0, gpu_time_ms = 0.0;

    int num_iters = 50;

    for(int i = 0; i < 50; i++) {
        auto cpu_start = chrono::high_resolution_clock::now();
        self_attention_backward_cpu(Q, K, V, dO, P, dP, dQ, dV, dK, dS, N, dim);
        auto cpu_end = chrono::high_resolution_clock::now();
        auto cpu_time = chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start).count();

        if(i > 0) cpu_time_ms += cpu_time / 1000.0; // count time excluding 1st iteration
    }

    cudaMemset(dQ, 0, sizeof(float) * N * dim);
    cudaMemset(dK, 0, sizeof(float) * N * dim);
    cudaMemset(dV, 0, sizeof(float) * N * dim);
    cudaMemset(dS, 0, sizeof(float) * N * N);
    cudaMemset(dP, 0, sizeof(float) * N * N);
    

    for(int i = 0; i < num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        self_attention_backward(Q, K, V, dO, P, dP, dQ, dV, dK, dS, N, dim);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();

        if(i > 0) gpu_time_ms += gpu_time / 1000.0; // count time excluding 1st iteration
    }

    float avg_cpu_time_ms = cpu_time_ms / (num_iters - 1), avg_gpu_time_ms = gpu_time_ms / (num_iters - 1);
    cout << "CPU time: " << avg_cpu_time_ms << " ms" << endl;
    cout << "GPU time: " << avg_gpu_time_ms << " ms" << endl;

    return 0;
}
