#include "self_attention.cuh"
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


using namespace std;

float rand_float() {
    // float r = (float)rand() / RAND_MAX;
    return rand()%5;
}

void print_output(float* O, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << O[i * N + j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char **argv) {
    int N = stoi(argv[1]);
    int dim = stoi(argv[2]);

    // float *Q = new float[N * dim];
    // float *K = new float[N * dim];
    // float *V = new float[N * dim];
    // float *O = new float[N * dim];
    float *Q, *K, *V, *O;
    cudaMallocManaged((void **)&Q, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&K, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&V, sizeof(float) * N * dim);
    cudaMallocManaged((void **)&O, sizeof(float) * N * N);

    generate(Q, Q + N * dim, rand_float);
    generate(K, K + N * dim, rand_float);
    generate(V, V + N * dim, rand_float);

    // auto start = chrono::high_resolution_clock::now();
    // self_attention(Q, K, V, O, N, dim);
    // auto end = chrono::high_resolution_clock::now();

    // auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // cout << duration.count()/1000.0 << endl;


    self_attention(Q, K, V, O, N, dim);

    

    return 0;
}
