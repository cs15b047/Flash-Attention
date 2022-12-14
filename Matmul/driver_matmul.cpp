#include "matmul.cuh"
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

void matmul_serial(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void calculate_error(float* C_cpu, float* C_gpu, int M, int N) {
    double max_error = 0.0, total_error = 0.0;
    for(long long int i = 0; i < (long long int)(M * N); i++) {
        double error = fabs(C_cpu[i] - C_gpu[i]);
        if(error > max_error) max_error = error;
        total_error += error;
    }
    cout << "Max error: " << max_error << endl;
    cout << "Total error: " << total_error << endl;
}


int main(int argc, char* argv[]) {
    int M = stoi(argv[1]);
    int N = stoi(argv[2]);
    int K = stoi(argv[3]);

    float *A, *B, *C1, *C2, *C3, *C4, *C_cpu;
    cudaMallocManaged((void **)&A, sizeof(float) * M * K);
    cudaMallocManaged((void **)&B, sizeof(float) * K * N);
    cudaMallocManaged((void **)&C1, sizeof(float) * M * N);
    cudaMallocManaged((void **)&C2, sizeof(float) * M * N);
    cudaMallocManaged((void **)&C3, sizeof(float) * M * N);
    cudaMallocManaged((void **)&C4, sizeof(float) * M * N);
    C_cpu = new float[M * N];

    generate(A, A + M * K, rand_float);
    generate(B, B + K * N, rand_float);

    float cpu_time_ms = 0, gpu_time_ms = 0, gpu_time_tiled_ms = 0, gpu_time_tiled_coalesced_ms = 0, gpu_time_tiled_coalesced_one_to_many_ms = 0;
    int num_iters = 2;

    // for(int i = 0; i <= num_iters; i++) {
    //     auto cpu_start = chrono::high_resolution_clock::now();
    //     matmul_serial(A, B, C_cpu, M, N, K);
    //     auto cpu_end = chrono::high_resolution_clock::now();
    //     auto cpu_time = chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start).count();
    //     if(i > 0) cpu_time_ms += cpu_time / 1000.0;
    // }

    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        matmul(A, B, C1, M, N, K);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_ms += gpu_time / 1000.0;
    }

    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        matmul_tiled(A, B, C2, M, N, K);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_tiled_ms += gpu_time / 1000.0;
    }

    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        matmul_tiled_coalesced(A, B, C3, M, N, K);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_tiled_coalesced_ms += gpu_time / 1000.0;
    }

    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        matmul_tiled_coalesced_one_to_many(A, B, C4, M, N, K);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_tiled_coalesced_one_to_many_ms += gpu_time / 1000.0;
    }


    float avg_cpu_ms = cpu_time_ms / (num_iters - 1);
    float avg_gpu_ms = gpu_time_ms / (num_iters - 1);
    float avg_gpu_tiled_ms = gpu_time_tiled_ms / (num_iters - 1);
    float avg_gpu_tiled_coalesced_ms = gpu_time_tiled_coalesced_ms / (num_iters - 1);
    float avg_gpu_tiled_coalesced_one_to_many_ms = gpu_time_tiled_coalesced_one_to_many_ms / (num_iters - 1);

    cout << "CPU time: " << avg_cpu_ms << " ms" << endl;
    cout << "GPU time: " << avg_gpu_ms << " ms" << endl;
    cout << "GPU time (tiled): " << avg_gpu_tiled_ms << " ms" << endl;
    cout << "GPU time (tiled, coalesced): " << avg_gpu_tiled_coalesced_ms << " ms" << endl;
    cout << "GPU time (tiled, coalesced, one to many): " << avg_gpu_tiled_coalesced_one_to_many_ms << " ms" << endl;


    calculate_error(C1, C2, M, N);
    calculate_error(C1, C3, M, N);
    calculate_error(C1, C4, M, N);
}