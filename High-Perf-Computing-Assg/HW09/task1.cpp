#include "cluster.h"
#include <bits/stdc++.h>

using namespace std;

float random_float(int n) {
    float f = (float)rand() / RAND_MAX;
    f = f * n;
    return f;
}

void print_array(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    int n = stoi(argv[1]);
    int t = stoi(argv[2]);

    omp_set_num_threads(t);

    // sorted array of random floats
    float* arr = new float[n];
    for(int i = 0; i < n; i++) {
        arr[i] = random_float(n);
    }
    sort(arr, arr + n);

    // cout << "Array of points: " << endl;
    // print_array(arr, n);

    // centers
    float* centers = new float[t];
    float multiplier = (float)n / (float)(2 * t);
    for(int i = 0; i < t; i++) {
        centers[i] = multiplier * (2 * i + 1);
    }
    // cout << "Centers: " << endl;
    // print_array(centers, t);


    // distances
    float* dists = new float[t];
    for(int i = 0; i < t; i++) {
        dists[i] = 0.0;
    }

    auto start = chrono::high_resolution_clock::now();
    cluster(n, t, arr, centers, dists);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // cout << "Distances: " << endl;
    // print_array(dists, t);

    int max_index = 0;
    float max_dist = dists[0];

    for(int i = 1; i < t; i++) {
        if(dists[i] > max_dist) {
            max_dist = dists[i];
            max_index = i;
        }
    }

    cout << max_dist << endl;
    cout << max_index << endl;
    cout << duration.count()/1000.0 << endl;

}