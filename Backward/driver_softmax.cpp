#include "self_attention_backward.cuh"
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


using namespace std;

float rand_float() {
    float r = 0.1 * (float)rand() / RAND_MAX;
    return 2 * r - 1;
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
    int batch_size = stoi(argv[3]);
    int num_heads = stoi(argv[4]);

    float *P, *dP;
    float *dS, *rowsums;
    cudaMallocManaged((void **)&P, sizeof(float) * N * N * batch_size * num_heads);
    cudaMallocManaged((void **)&dS, sizeof(float) * N * N * batch_size * num_heads);
    cudaMallocManaged((void **)&dP, sizeof(float) * N * N * batch_size * num_heads);
    cudaMallocManaged((void **)&rowsums, sizeof(float) * N * batch_size * num_heads);

    generate(P, P + N * N * batch_size * num_heads, rand_float);
    generate(dP, dP + N * N * batch_size * num_heads, rand_float);


    float gpu_time_ms = 0.0;
    int num_iters = 2;

    for(int i = 0; i < num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        softmax_backward(P, dP, dS, rowsums, N, batch_size, num_heads);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();

        if(i > 0) gpu_time_ms += gpu_time / 1000.0; // count time excluding 1st iteration
    }

    float avg_gpu_time_ms = gpu_time_ms / (num_iters - 1);
    cout << "GPU time: " << avg_gpu_time_ms << " ms" << endl;

    cudaFree(rowsums);
    cudaFree(dS);
    cudaFree(dP);
    cudaFree(P);

    return 0;
}
