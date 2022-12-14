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

void softmax_grad_serial(float* P_, float* dP_, float* dS_, float* rowsums_, int N, int batch_size, int num_heads) {
    for(int idx = 0; idx < batch_size * num_heads; idx++) {
        float* P = P_ + idx * N * N, *dP = dP_ + idx * N * N, *dS = dS_ + idx * N * N, *rowsums = rowsums_ + idx * N;
        // calculate rowsums
        for(int i = 0; i < N; i++) {
            float sum = 0.0;
            for(int j = 0; j < N; j++) {
                sum += P[i * N + j] * dP[i * N + j];
            }
            rowsums[i] = sum;
        }

        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                // dS = PdP - P rowsum(PdP)
                dS[i * N + j] = P[i * N + j] * dP[i * N + j] - P[i * N + j] * rowsums[i];
            }
        }
    }
}

void calculate_error(float* dS_cpu, float* dS_gpu, float* rowsums_cpu, float* rowsums_gpu, int N, int batch_size, int num_heads) {
    double max_error = 0.0, total_error = 0.0;
    for(long long int i = 0; i < (long long int)(N * N) * (long long int)batch_size * num_heads; i++) {
        double error = fabs(dS_cpu[i] - dS_gpu[i]);
        if(error > max_error) max_error = error;
        total_error += error;
    }
    cout << "Max error: " << max_error << endl;
    cout << "Total error: " << total_error << endl;
}

int main(int argc, char **argv) {
    int N = stoi(argv[1]);
    int dim = stoi(argv[2]);
    int batch_size = stoi(argv[3]);
    int num_heads = stoi(argv[4]);

    float *P, *dP;
    float *dS1, *dS2, *dS3, *rowsums1, *rowsums2, *rowsums3;
    cudaMallocManaged((void **)&P, sizeof(float) * N * N * batch_size * num_heads);
    cudaMallocManaged((void **)&dS1, sizeof(float) * N * N * batch_size * num_heads);
    cudaMallocManaged((void **)&dS2, sizeof(float) * N * N * batch_size * num_heads);
    cudaMallocManaged((void **)&dP, sizeof(float) * N * N * batch_size * num_heads);
    cudaMallocManaged((void **)&rowsums1, sizeof(float) * N * batch_size * num_heads);
    cudaMallocManaged((void **)&rowsums2, sizeof(float) * N * batch_size * num_heads);

    generate(P, P + N * N * batch_size * num_heads, rand_float);
    generate(dP, dP + N * N * batch_size * num_heads, rand_float);


    float gpu_time_ms_naive = 0.0, gpu_time_ms_optimized1 = 0.0, gpu_time_ms_optimized2 = 0.0;
    int num_iters = 10;

    // softmax_grad_serial(P, dP, dS1, rowsums1, N, batch_size, num_heads);

    // for(int i = 0; i < num_iters; i++) {
    //     auto gpu_start = chrono::high_resolution_clock::now();
    //     softmax_backward1(P, dP, dS1, rowsums1, N, batch_size, num_heads);
    //     auto gpu_end = chrono::high_resolution_clock::now();
    //     auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();

    //     if(i > 0) gpu_time_ms_naive += gpu_time / 1000.0; // count time excluding 1st iteration
    // }

    for(int i = 0; i < num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        softmax_backward2(P, dP, dS2, rowsums2, N, batch_size, num_heads);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();

        if(i > 0) gpu_time_ms_optimized1 += gpu_time / 1000.0; // count time excluding 1st iteration
    }

    for(int i = 0; i < num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        softmax_backward3(P, dP, dS2, rowsums2, N, batch_size, num_heads);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();

        if(i > 0) gpu_time_ms_optimized2 += gpu_time / 1000.0; // count time excluding 1st iteration
    }

    // calculate_error(dS1, dS2, rowsums1, rowsums2, N , batch_size, num_heads);
    
    float avg_gpu_time_ms_naive = gpu_time_ms_naive / (num_iters - 1);
    float avg_gpu_time_ms_optimized1 = gpu_time_ms_optimized1 / (num_iters - 1);
    float avg_gpu_time_ms_optimized2 = gpu_time_ms_optimized2 / (num_iters - 1);
    // cout << "GPU time (naive): " << avg_gpu_time_ms_naive << " ms" << endl;
    cout << "GPU time (optimized-1): " << avg_gpu_time_ms_optimized1 << " ms" << endl;
    cout << "GPU time (optimized-2): " << avg_gpu_time_ms_optimized2 << " ms" << endl;

    cudaFree(rowsums1);
    cudaFree(rowsums2);
    cudaFree(dS1);
    cudaFree(dS2);
    cudaFree(dP);
    cudaFree(P);

    return 0;
}
