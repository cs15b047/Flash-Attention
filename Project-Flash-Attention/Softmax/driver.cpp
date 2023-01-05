#include "softmax.cuh"
#include "softmax_cublas.cuh"
#include "softmax.h"
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
    int seq_len = stoi(argv[2]);
    
    float *input,*output4, *output_cpu, *output1, *output2, *output3, *output5;
    cudaMallocManaged((void **)&input, sizeof(float) * N * seq_len);
    cudaMallocManaged((void **)&output1, sizeof(float) * N * seq_len);
    cudaMallocManaged((void **)&output2, sizeof(float) * N * seq_len);
    cudaMallocManaged((void **)&output3, sizeof(float) * N * seq_len);
    cudaMallocManaged((void **)&output4, sizeof(float) * N * seq_len);
    cudaMallocManaged((void **)&output5, sizeof(float) * N * seq_len);


    output_cpu = new float[N * seq_len];

    generate(input, input + N * seq_len, rand_float);
    
    float gpu_time_ms = 0.0;
    int num_iters = 2;

    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        softmax_cpu(input, output_cpu, N, seq_len);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_ms += gpu_time / 1000.0;
    }
    cout << "CPU Time: " << gpu_time_ms / num_iters << " ms" << endl;

    gpu_time_ms = 0.0;
    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        softmax1(input, output1, N, seq_len);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_ms += gpu_time / 1000.0;
    }
    cout << "GPU1 Time: " << gpu_time_ms / num_iters << " ms" << endl;

    gpu_time_ms = 0.0;
    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        softmax2(input, output2, N, seq_len);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_ms += gpu_time / 1000.0;
    }
    cout << "GPU2 Time: " << gpu_time_ms / num_iters << " ms" << endl;

    gpu_time_ms = 0.0;
    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        softmax3(input, output3, N, seq_len);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_ms += gpu_time / 1000.0;
    }

    cout << "GPU3 Time: " << gpu_time_ms / num_iters << " ms" << endl;

    gpu_time_ms = 0.0;
    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        fused_softmax(input, output4, N, seq_len,1);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_ms += gpu_time / 1000.0;
    }

    cout << "GPU4 Time: " << gpu_time_ms / num_iters << " ms" << endl;

    gpu_time_ms = 0.0;
    for(int i = 0; i <= num_iters; i++) {
        auto gpu_start = chrono::high_resolution_clock::now();
        fused_softmax(input, output5, N, seq_len,30);
        auto gpu_end = chrono::high_resolution_clock::now();
        auto gpu_time = chrono::duration_cast<chrono::microseconds>(gpu_end - gpu_start).count();
        if(i > 0) gpu_time_ms += gpu_time / 1000.0;
    }

    cout << "GPU5 Time: " << gpu_time_ms / num_iters << " ms" << endl;




    // Calculate error
    double error = 0;
    for (int i = 0; i < N ; i++) {
        double sum = 0;
        for (int j = 0; j < seq_len; j++) {
            sum += output5[i * seq_len + j];
        }
        error += fabs(sum - 1.0);
        // cout<<sum<<endl;
        
    }
    // cout << "Error: " << error << endl;
    

    return 0;
}
