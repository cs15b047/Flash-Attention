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

    float *Q, *K, *V, *O, *O_cpu;
    Q = new float[N*dim];
    K = new float[N*dim];
    V = new float[N*dim];
    O = new float[N*dim];
    O_cpu = new float[N * dim];

    generate(Q, Q + N * dim, rand_float);
    generate(K, K + N * dim, rand_float);
    generate(V, V + N * dim, rand_float);

    //calculate time 
    clock_t start, end;
    start = clock();
    self_attention_cpu(Q, K, V, O_cpu, N, dim);
    end = clock();
    double time_taken = double(end - start)*1000 / double(CLOCKS_PER_SEC);
    cout << "Time taken by GPU is : " << fixed
         << time_taken << setprecision(5);
    cout << " sec(ms) " << endl;


    
    
    
    

    

    return 0;
}
