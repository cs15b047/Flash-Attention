#include "self_attention.cuh"
#include "self_attention.h"
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


using namespace std;

float rand_float() {
    // float r = (float)rand() / RAND_MAX;
    return rand()%5;
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

    float *Q, *K, *V, *O, *O_cpu;
    cudaMallocManaged((void **)&Q, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&K, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&V, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&O, sizeof(float) * N * dim);
    O_cpu = new float[N * dim];

    generate(Q, Q + N * dim, rand_float);
    generate(K, K + N * dim, rand_float);
    generate(V, V + N * dim, rand_float);

    print_mat(Q, N, dim);
    cout << endl;
    print_mat(K, N, dim);
    cout << endl;
    print_mat(V, N, dim);
    cout << endl;

    self_attention(Q, K, V, O, N, dim);
    cudaDeviceSynchronize();

    self_attention_cpu(Q, K, V, O_cpu, N, dim);

    cout << "GPU output:" << endl;
    print_mat(O, N, dim);
    cout << endl;

    cout << "CPU output:" << endl;
    print_mat(O_cpu, N, dim);
    cout << endl;

    

    

    return 0;
}
