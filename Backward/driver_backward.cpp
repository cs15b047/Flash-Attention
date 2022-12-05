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
    cudaMallocManaged((void **)&dS, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&dO, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&dP, sizeof(float) * N * dim);

    O_cpu = new float[N * dim];

    generate(Q, Q + N * dim, rand_float);
    generate(K, K + N * dim, rand_float);
    generate(V, V + N * dim, rand_float);
    generate(dO, dO + N * dim, rand_float);
    generate(P, P + N * N, rand_float);

    self_attention_backward_cpu(Q, K, V, dO, P, dP, dQ, dV, dK, dS, N, dim);

    printf("dK: \n");
    print_matrix(dK, N, dim);
    printf("dQ: \n");
    print_matrix(dQ, N, dim);

    cudaMemset(dQ, 0, sizeof(float) * N * dim);
    cudaMemset(dK, 0, sizeof(float) * N * dim);
    cudaMemset(dV, 0, sizeof(float) * N * dim);
    cudaMemset(dS, 0, sizeof(float) * N * N);
    cudaMemset(dP, 0, sizeof(float) * N * N);
    

    self_attention_backward(Q, K, V, dO, P, dP, dQ, dV, dK, dS, N, dim);
    // cudaDeviceSynchronize();

    printf("dK: \n");
    print_matrix(dK, N, dim);
    printf("dQ: \n");
    print_matrix(dQ, N, dim);

    // self_attention_cpu(Q, K, V, O_cpu, N, dim);

    // cout << "GPU output:" << endl;
    // print_mat(O, N, dim);
    // cout << endl;

    // cout << "CPU output:" << endl;
    // print_mat(O_cpu, N, dim);
    // cout << endl;

    // // Calculate error
    // double error = 0;
    // for (int i = 0; i < N * dim; i++) {
    //     error += fabs(O[i] - O_cpu[i]);
    // }
    // cout << "Error: " << error << endl;

    

    

    return 0;
}
