#include<bits/stdc++.h>
#include "scan.h"

using namespace std;

inline float get_rand_number() {
    float num = (float)rand() / (float)RAND_MAX;// number between 0 and 1
    num = -1 + 2 * num; // number between -1 and 1
    return num;
}

int main(int argc, char const *argv[]) {
    srand(time(nullptr));
    int n = stoi(argv[1]);
    float* arr = new float[n];
    float* output = new float[n];
    generate(arr, arr + n, get_rand_number);

    auto start = chrono::high_resolution_clock::now();
    scan(arr, output, n);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds> (end - start);
    double time_taken_in_ms = (double)duration.count() / 1000.0;
    cout << time_taken_in_ms << endl;
    cout << output[0] << endl;
    cout << output[n - 1] << endl;

    delete[] arr;
    delete[] output;
}