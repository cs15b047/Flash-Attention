#include "reduce.cuh"
#include <iostream>

using namespace std;

float generateRandomNumber(){
    float r = (float)rand() / (float)RAND_MAX;
    return 2*r - 1;
}

void generate_input_data(float **p_input, int size){
    float* input = *p_input;
    for(int i = 0; i < size; i++){
        input[i] = generateRandomNumber();
    }
}

unsigned int get_output_size(int input_size, int threads_per_block) {
    // strategy: 1 block --> 1 sum
    unsigned int output_size = (input_size + threads_per_block - 1)/ threads_per_block;

    return output_size;
}

void allocate_device_arrays(float **d_input, float **d_output, int size, int threads_per_block){
    unsigned int output_size = get_output_size(size, threads_per_block);
    cudaMalloc((void **)d_input, size * sizeof(float));
    cudaMalloc((void **)d_output, output_size * sizeof(float));
}

void copy_data_to_device(float *input, float **device_input, int size){
    cudaMemcpy(*device_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
}

void print_array(float *array, int size){
    for(int i = 0; i < size; i++){
        cout << array[i] << " ";
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    int N = stoi(argv[1]);
    int threads_per_block = stoi(argv[2]);

    float *h_input = new float[N];
    float *d_input, *d_output;
    
    generate_input_data(&h_input, N);
    allocate_device_arrays(&d_input, &d_output, N, threads_per_block);
    copy_data_to_device(h_input, &d_input, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce(&d_input, &d_output, (unsigned int)N, (unsigned int)threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float* sum = new float[1];

    cudaMemcpy(sum, d_output, 1*sizeof(float), cudaMemcpyDeviceToHost);

    cout << *sum << endl;
    cout << milliseconds << endl;
}