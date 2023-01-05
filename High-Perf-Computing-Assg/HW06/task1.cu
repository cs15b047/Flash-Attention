#include "mmul.h"
#include <iostream>
#include <vector>
using namespace std;

float inline generateRandomNumber() {
    float r = (float)rand() / (float)RAND_MAX;
    return 2 * r - 1;
}

void initMatrices(float *A, float *B, float *C, int n) {
    for(int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            // init (i,j)th element
            A[j * n + i] = generateRandomNumber();
            B[j * n + i] = generateRandomNumber();
            C[j * n + i] = 0;
        }
    }
}

void initialize_arrays_for_mm(float **A, float **B, float **C, int n) {
    cudaMallocManaged(A, n * n * sizeof(float));
    cudaMallocManaged(B, n * n * sizeof(float));
    cudaMallocManaged(C, n * n * sizeof(float));
    initMatrices(*A, *B, *C, n);
}

void free_mem(float *A, float *B, float *C) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

int main(int argc, char *argv[]) {
    int n = stoi(argv[1]);
    int n_tests = stoi(argv[2]);

    float *A, *B, *C;
    initialize_arrays_for_mm(&A, &B, &C, n);

    float total_exec_time = 0;
    vector<float> exec_times;
    for (int i = 0; i < n_tests; i++) {
        float time_in_ms = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        cublasHandle_t handle;
        cublasCreate(&handle);

        mmul(handle, A, B, C, n);

        cublasDestroy(handle);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time_in_ms, start, stop);
        total_exec_time += time_in_ms;
        exec_times.push_back(time_in_ms);
    }

    float average_time = total_exec_time / (float)n_tests;
    cout << average_time << endl;
}