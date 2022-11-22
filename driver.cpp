#include "self_attention.h"
#include <bits/stdc++.h>

using namespace std;

float rand_float() {
    float r = (float)rand() / RAND_MAX;
    return 2 * r - 1;
}

int main(int argc, char **argv) {
    int N = stoi(argv[1]);
    int dim = stoi(argv[2]);

    float *Q = new float[N * dim];
    float *K = new float[N * dim];
    float *V = new float[N * dim];
    float *O = new float[N * dim];

    generate(Q, Q + N * dim, rand_float);
    generate(K, K + N * dim, rand_float);
    generate(V, V + N * dim, rand_float);

    auto start = chrono::high_resolution_clock::now();
    self_attention(Q, K, V, O, N, dim);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << duration.count()/1000.0 << endl;

    return 0;
}
