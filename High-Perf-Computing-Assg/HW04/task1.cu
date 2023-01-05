#include "matmul.cuh"
#include<iostream>

using namespace std;

float generate_random_number(){
    float r = (float)rand() / (float)RAND_MAX;
    r = 2*r-1;
    return r;
}

void allocate_host_arrays(float** pa, float** pb, float** pc, int n){
    *pa = new float[n*n];
    *pb = new float[n*n];
    *pc = new float[n*n];

    float* a = *pa;
    float* b = *pb;
    float* c = *pc; 

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            int idx = i*n + j;
            a[idx] = generate_random_number();
            b[idx] = generate_random_number();
            c[idx] = 0;
        }
    }
}

void allocate_device_arrays(float** d_a, float** d_b, float** d_c, size_t n){
    cudaMalloc((void**)d_a, n*n*sizeof(float));
    cudaMalloc((void**)d_b, n*n*sizeof(float));
    cudaMalloc((void**)d_c, n*n*sizeof(float));
}

void copy_host_to_device(float* a, float* b, float* c, float* d_a, float* d_b, float* d_c, size_t n){
    cudaMemcpy(d_a, a, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, n*n*sizeof(float), cudaMemcpyHostToDevice);
}

void print_array(float* a, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            int idx = i*n + j;
            cout << a[idx] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char *argv[]) {
    size_t n = stol(argv[1]);
    unsigned int threads_per_block = stoi(argv[2]);

    srand(time(nullptr));

    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    allocate_device_arrays(&d_a, &d_b, &d_c, n);
    allocate_host_arrays(&a, &b, &c, n);

    copy_host_to_device(a, b, c, d_a, d_b, d_c, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul(d_a, d_b, d_c, n, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(c, d_c, n*n*sizeof(float), cudaMemcpyDeviceToHost);

    cout << c[n*n - 1] << endl;
    cout << milliseconds << endl;

    return 0;
}