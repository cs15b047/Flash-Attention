#include "matmul.cuh"
#include <iostream>
#include <vector>

using namespace std;

vector<int> matrix_a, matrix_b;

int generate_random_number(){
    return (rand() % 5);
}

void generate_random_data(int n){
    matrix_a.clear();
    matrix_b.clear();
    for(int i = 0; i < n * n; i++){
        matrix_a.push_back(generate_random_number());
        matrix_b.push_back(generate_random_number());
    }
}

template<typename T>
void allocate_host_arrays(T** pa, T** pb, T** pc, int n){
    *pa = new T[n*n];
    *pb = new T[n*n];
    *pc = new T[n*n];

    T* a = *pa;
    T* b = *pb;
    T* c = *pc;

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            int idx = i*n + j;
            a[idx] = (T)matrix_a[idx];
            b[idx] = (T)matrix_b[idx];
            c[idx] = 0;
        }
    }
}

template<typename T>
void allocate_device_arrays(T** d_a, T** d_b, T** d_c, size_t n){
    cudaMalloc((void**)d_a, n*n*sizeof(T));
    cudaMalloc((void**)d_b, n*n*sizeof(T));
    cudaMalloc((void**)d_c, n*n*sizeof(T));
}

template<typename T>
void copy_host_to_device(T* a, T* b, T* c, T* d_a, T* d_b, T* d_c, size_t n){
    cudaMemcpy(d_a, a, n*n*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*n*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, n*n*sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void print_array(T* a, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            int idx = i*n + j;
            cout << a[idx] << " ";
        }
        cout << endl;
    }
}

void print_data(int n) {
    cout << "Matrix A:" << endl;
    print_array(&matrix_a[0], n);
    cout << "Matrix B:" << endl;
    print_array(&matrix_b[0], n);
}

template<typename T>
void init_data_and_run(size_t n, unsigned int block_dim) {
    T *a, *b, *c;
    T *d_a, *d_b, *d_c;

    allocate_device_arrays<T>(&d_a, &d_b, &d_c, n);
    allocate_host_arrays<T>(&a, &b, &c, n);
    copy_host_to_device<T>(a, b, c, d_a, d_b, d_c, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (is_same<int, T>::value) matmul_1((const int*)d_a, (const int*)d_b, (int*)d_c, n, block_dim);
    else if (is_same<float, T>::value) matmul_2((const float *)d_a, (const float *)d_b, (float *)d_c, n, block_dim);
    else if (is_same<double, T>::value) matmul_3((const double *)d_a, (const double *)d_b, (double *)d_c, n, block_dim);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(c, d_c, n*n*sizeof(T), cudaMemcpyDeviceToHost);

    // print_array(c, n); cout << endl;
    cout << c[0] << endl;
    cout << c[n * n - 1] << endl;
    cout << milliseconds << endl;
}

int main(int argc, char *argv[]) {
    size_t n = stol(argv[1]);
    unsigned int block_dim = stoi(argv[2]);
    srand(time(nullptr));

    generate_random_data(n);
    // print_data(n);

    init_data_and_run<int>(n, block_dim);
    init_data_and_run<float>(n, block_dim);
    init_data_and_run<double>(n, block_dim);

    return 0;
}