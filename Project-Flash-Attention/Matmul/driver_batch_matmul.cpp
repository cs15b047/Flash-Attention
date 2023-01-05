#include "batch_matmul.cuh"
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
using namespace std;

float rand_float() {
    float r = 0.1 * (float)rand() / RAND_MAX;
    return 2 * r - 1;
}

void print_mat(float* O, int N, int dim,int batch_size) {
    for(int b = 0; b < batch_size; b++) {
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < dim; j++) {
                cout << O[b * N * dim + i * dim + j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void matmul_serial(const float *A, const float *B, float *C, int batch_size, int M, int N, int K) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float* c = &C[b*M*N];
                const float* a = &A[b*M*K];
                const float* b_ = &B[b*K*N];
                c[i * N + j] = 0;
                for (int k = 0; k < K; k++) {
                    c[i * N + j] += a[i * K + k] * b_[k * N + j];
                }
            }
        }
    }
}

void calculate_error(float* C_cpu, float* C_gpu, int batch_size, int M, int N) {
    double max_error = 0.0, total_error = 0.0;
    for(long long int b = 0; b < batch_size; b++) {
        for(long long int i = 0; i < (long long int)(M * N); i++) {
            double error = fabs(C_cpu[b * M * N + i] - C_gpu[b * M * N + i]);
            if(error > max_error) max_error = error;
            total_error += error;
        }
    }
    cout << "Max error: " << max_error << endl;
    cout << "Total error: " << total_error << endl;
}
    


int main(int argc, char* argv[]) {
    int batch_size = atoi(argv[1]);
    int M = atoi(argv[2]);
    int N = atoi(argv[3]);
    int K = atoi(argv[4]);


    float *A, *B, *C1, *C2, *C_cpu;
    cudaMallocManaged((void **)&A, sizeof(float) * batch_size *M * K);
    cudaMallocManaged((void **)&B, sizeof(float) * batch_size * K * N);
    cudaMallocManaged((void **)&C1, sizeof(float) * batch_size * M * N);



    C_cpu = new float[batch_size * M * N];
    cudaMallocManaged((void **)&C2, sizeof(float) * M * N);
    // cudaMallocManaged((void **)&C3, sizeof(float) * M * N);
    // cudaMallocManaged((void **)&C4, sizeof(float) * M * N);
    
    generate(A, A + batch_size * M * K, rand_float);
    generate(B, B + batch_size * K * N, rand_float);

    float cpu_time_ms = 0, gpu_time_ms = 0, gpu_single_kernel_time_ms = 0;
    int num_iters = 2;

    for(int i = 0; i <= num_iters; i++) {
        auto cpu_start = chrono::high_resolution_clock::now();
        matmul_serial(A, B, C_cpu, batch_size, M, N, K);
        auto cpu_end = chrono::high_resolution_clock::now();
        auto cpu_time = chrono::duration_cast<chrono::microseconds>(cpu_end - cpu_start).count();
        if(i > 0) cpu_time_ms += cpu_time / 1000.0;
    }

    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        batch_matmul(A, B, C1, batch_size, M, N, K);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_ms += gpu_time / 1000.0;
    }

    // for(int i = 0; i <= num_iters; i++) {
    //     auto gpu_start = chrono::high_resolution_clock::now();
    //     batch_matmul_single_kernel(A, B, C2, batch_size, M, N, K);
    //     auto gpu_end = chrono::high_resolution_clock::now();
    //     auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
    //     if(i > 0) gpu_single_kernel_time_ms += gpu_time / 1000.0;
    // }






    float avg_cpu_ms = cpu_time_ms / (num_iters - 1);
    float avg_gpu_ms = gpu_time_ms / (num_iters - 1);
    float avg_gpu_single_kernel_ms = gpu_single_kernel_time_ms / (num_iters - 1);
    
    cout << "CPU time: " << avg_cpu_ms << " ms" << endl;
    cout << "GPU time: " << avg_gpu_ms << " ms" << endl;
    cout<<"Single kernel time: "<<avg_gpu_single_kernel_ms<<" ms"<<endl;

    
    
    calculate_error(C1, C_cpu, batch_size, M, N);
    
}