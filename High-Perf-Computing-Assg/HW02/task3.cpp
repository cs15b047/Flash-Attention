#include<bits/stdc++.h>
#include "matmul.h"

using namespace std;

inline double get_random_double() {
    return ((double)rand() / (double)RAND_MAX) * 2 - 1;
}

double* generate_random_matrix(int n) {
    double* matrix = new double[n*n];
    generate(matrix, matrix + n*n, get_random_double);
    return matrix;
}

void clear_matrix(double* matrix, int n) {
    memset(matrix, 0, n*n*sizeof(double));
}

int main() {
    srand(time(nullptr));
    int n = 1024;
    cout << n << endl;
    
    // Matrix init
    double* A = generate_random_matrix(n);
    double* B = generate_random_matrix(n);
    double* C = (double*)calloc(n*n, sizeof(double));

    // Matrix multiplication using order of loops
    auto start1 = chrono::high_resolution_clock::now();
    mmul1(A, B, C, n);
    auto end1 = chrono::high_resolution_clock::now();
    
    auto duration1 = chrono::duration_cast<chrono::microseconds>(end1 - start1);
    cout << duration1.count()/1000.0 << endl;
    cout << C[n * n - 1] << endl;
    clear_matrix(C, n);

    auto start2 = chrono::high_resolution_clock::now();
    mmul2(A, B, C, n);
    auto end2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::microseconds>(end2 - start2);
    cout << duration2.count()/1000.0 << endl;
    cout << C[n * n - 1] << endl;
    clear_matrix(C, n);

    auto start3 = chrono::high_resolution_clock::now();
    mmul3(A, B, C, n);
    auto end3 = chrono::high_resolution_clock::now();
    auto duration3 = chrono::duration_cast<chrono::microseconds>(end3 - start3);
    cout << duration3.count()/1000.0 << endl;
    cout << C[n * n - 1] << endl;
    clear_matrix(C, n);

    // copy data from array to vector
    vector<double> vecA(A, A + n*n), vecB(B, B + n*n);

    auto start4 = chrono::high_resolution_clock::now();
    mmul4(vecA, vecB, C, n);
    auto end4 = chrono::high_resolution_clock::now();
    auto duration4 = chrono::duration_cast<chrono::microseconds>(end4 - start4);
    cout << duration4.count()/1000.0 << endl;
    cout << C[n * n - 1] << endl;
    clear_matrix(C, n);
}