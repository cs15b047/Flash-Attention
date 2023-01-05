#include "self_attention.cuh"
#include "self_attention.h"
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


using namespace std;

float rand_float() {
    float r = 0.1 * (float)rand() / RAND_MAX;
    return 2 * r - 1;
}

void print_mat(float* O, int N, int dim) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < dim; j++) {
            cout << O[i * dim + j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char **argv) {
    int N = stoi(argv[1]);
    int dim = stoi(argv[2]);

    float *Q, *K, *V, *O, *O_cpu,*O1;
    cudaMallocManaged((void **)&Q, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&K, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&V, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&O, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&O1, sizeof(float) * N * dim);
    O_cpu = new float[N * dim];

    generate(Q, Q + N * dim, rand_float);
    generate(K, K + N * dim, rand_float);
    generate(V, V + N * dim, rand_float);

    //calculate time
    float gpu_time_ms = 0.0;
    int num_iters = 2;

    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        self_attention_cpu(Q, K, V, O_cpu, N, dim);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_ms += gpu_time / 1000.0;
    }
    cout << "CPU Time: " << gpu_time_ms / num_iters << " ms" << endl;

    gpu_time_ms = 0.0;

    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        self_attention(Q, K, V, O1, N, dim);
        cudaDeviceSynchronize();
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_ms += gpu_time / 1000.0;
    }
    cout << "GPU Time: " << gpu_time_ms / num_iters << " ms" << endl;

    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        fused_self_attention(Q, K, V, O, N, dim);
        cudaDeviceSynchronize();
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_ms += gpu_time / 1000.0;
    }
    cout << "GPU Time: " << gpu_time_ms / num_iters << " ms" << endl;
    

    

    return 0;
}
