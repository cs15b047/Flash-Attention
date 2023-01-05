#include "stencil.cuh"
#include<iostream>

using namespace std;

float generate_random_number(){
    float r = (float)rand() / (float)RAND_MAX;
    r = 2*r-1;
    return r;
}

void allocate_host_arrays(float** pimage, float** pmask, float** poutput, unsigned int n, unsigned int R){
    *pimage = new float[n];
    *pmask = new float[2*R + 1];
    *poutput = new float[n];

    float* image = *pimage;
    float* mask = *pmask;
    float* output = *poutput; 

    for(unsigned int i = 0; i < 2*R + 1; i++){
        mask[i] = generate_random_number();
    }

    for(unsigned int i=0; i<n; i++){
        image[i] = generate_random_number();
        output[i] = 0;
    }
}

void allocate_device_arrays(float** d_image, float** d_mask, float** d_output, unsigned int n, unsigned int R){
    cudaMalloc((void**)d_image, n*sizeof(float));
    cudaMalloc((void**)d_mask, (2*R + 1)*sizeof(float));
    cudaMalloc((void**)d_output, n*sizeof(float));
}

void copy_host_to_device(float* image, float* mask, float* output, float* d_image, float* d_mask, float* d_output,
     unsigned int n, unsigned int R){
    cudaMemcpy(d_image, image, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, (2*R + 1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, n*sizeof(float), cudaMemcpyHostToDevice);
}

void print_array(float* a, int n){
    for(int i=0; i<n; i++){
        cout << a[i] << endl;
    }
}


int main(int argc, char* argv[]) {
    unsigned int n = stoi(argv[1]);
    unsigned int R = stoi(argv[2]);
    unsigned int threads_per_block = stoi(argv[3]);

    srand(time(nullptr));

    float *image, *mask, *output;
    float *d_image, *d_mask, *d_output;

    allocate_host_arrays(&image, &mask, &output, n, R);
    allocate_device_arrays(&d_image, &d_mask, &d_output, n, R);
    copy_host_to_device(image, mask, output, d_image, d_mask, d_output, n, R);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    stencil(d_image, d_mask, d_output, n, R, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(output, d_output, n*sizeof(float), cudaMemcpyDeviceToHost);

    cout << output[n - 1] << endl;
    cout << milliseconds << endl;
}