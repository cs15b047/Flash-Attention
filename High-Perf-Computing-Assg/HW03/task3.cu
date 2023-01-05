#include <iostream>
#include "vscale.cuh"
using namespace std;

float generate_a() {
    float r = (float)rand() / (float)RAND_MAX;
    r = r * 20 - 10;
    return r;
}

float generate_b() {
    return (float)rand() / (float)RAND_MAX;
}

int main(int argc, char *argv[]) {
    srand(time(nullptr));
    int n = stoi(argv[1]);
    int threads = 512, blocks = (n + threads - 1) / threads;

    float* h_a = new float[n];
    float* h_b = new float[n];

    // init data
    for(int i = 0; i < n; i++) {
        h_a[i] = generate_a();
        h_b[i] = generate_b();
    }

    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    // data reached device

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    vscale<<<blocks, threads>>>(d_a, d_b, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);

    // printing: time, 1st ele and last ele
    cout << ms << endl;
    cout << h_b[0] << endl;
    cout << h_b[n - 1] << endl;

    return 0;
}