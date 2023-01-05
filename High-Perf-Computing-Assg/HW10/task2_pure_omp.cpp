#include "reduce.h"
#include<bits/stdc++.h>

using namespace std;

float rand_float() {
    float r = (float)rand() / RAND_MAX;
    return 2*r - 1;
}

int main(int argc, char* argv[]) {
    int n = stoi(argv[1]);
    int threads = stoi(argv[2]);

    float *arr = new float[n];
    generate(arr, arr + n, rand_float);

    omp_set_num_threads(threads);

    auto start = chrono::high_resolution_clock::now();
    int start_idx = 0, end_idx = start_idx + n;
    float res = reduce(arr, start_idx, end_idx);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << res << endl;
    cout << duration.count()/1000.0 << endl;

    return 0;
}