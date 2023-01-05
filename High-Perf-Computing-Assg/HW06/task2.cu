#include "scan.cuh"
#include <iostream>

using namespace std;

float generate_random_float() {
    float r = (float)rand() / (float)RAND_MAX;
    r = 2 * r - 1;
    return r;
}

void print_array(double* array, int size) {
    for (int i = 0; i < size; i++) {
        cout << array[i] << " ";
    }
    cout << endl;
}

void serial_prefix_sum(const float* input, double* output, int size) {
    output[0] = (double)input[0];
    for (int i = 1; i < size; i++) {
        output[i] = output[i - 1] + (double)input[i];
    }
}

int main(int argc, char* argv[]) {
    srand(time(nullptr));
    int threads_per_block = stoi(argv[2]);
    int n = stoi(argv[1]);

    if(n > threads_per_block * threads_per_block){
        cout << "n must be less than threads_per_block * threads_per_block" << endl;
        return 0;
    }

    float* arr;
    cudaMallocManaged(&arr, n * sizeof(float));
    for(int i = 0; i < n; i++){
        arr[i] = generate_random_float();
    }

    float* result;
    cudaMallocManaged(&result, n * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    scan(arr, result, (unsigned int)n, (unsigned int)threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cout << result[n - 1] << endl;
    cout << milliseconds << endl;
    
    return 0;
}