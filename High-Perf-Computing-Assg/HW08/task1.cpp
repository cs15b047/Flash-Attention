#include<bits/stdc++.h>
#include "matmul.h"

using namespace std;

inline float get_random_float() {
    return ((float)rand() / (float)RAND_MAX) * 2 - 1;
}

float* generate_random_matrix(int n) {
    float* matrix = new float[n*n];
    generate(matrix, matrix + n*n, get_random_float);
    return matrix;
}

void clear_matrix(float* matrix, int n) {
    memset(matrix, 0, n*n*sizeof(float));
}

void matmul_serial(const float* A, const float* B, float* C, const int n) {
    for(int i = 0; i < n; i++) {
        for(int k = 0; k < n; k++) {
            for(int j = 0; j < n; j++) {
                // C(i, j) = C(i, j) + A(i, k) * B(k, j)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    srand(time(nullptr));
    int n = stoi(argv[1]);
    int num_threads = stoi(argv[2]);

    omp_set_num_threads(num_threads);
    
    // Matrix init
    float* A = generate_random_matrix(n);
    float* B = generate_random_matrix(n);
    float* C = (float*)calloc(n*n, sizeof(float));

    float* C_serial = (float*)calloc(n*n, sizeof(float));
    auto start_serial = chrono::high_resolution_clock::now();
    matmul_serial(A, B, C_serial, n);
    auto end_serial = chrono::high_resolution_clock::now();
    auto duration_serial = chrono::duration_cast<chrono::microseconds>(end_serial - start_serial);

    // Matrix multiplication using order of loops
    auto start1 = chrono::high_resolution_clock::now();
    mmul(A, B, C, n);
    auto end1 = chrono::high_resolution_clock::now();

    double error = 0;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            error += fabs(C[i*n + j] - C_serial[i*n + j]);
        }
    }

    // cout << "Error: " << error << endl;
    
    auto duration1 = chrono::duration_cast<chrono::microseconds>(end1 - start1);
    cout << C[0] << endl;
    cout << C[n * n - 1] << endl;
    cout << duration1.count()/1000.0 << endl;
}