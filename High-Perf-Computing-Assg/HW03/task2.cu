#include<iostream>
using namespace std;

__global__
void saxpy(int a, int* dA) {
    int x = threadIdx.x, y = blockIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dA[idx] = a * x + y;
}

int main() {
    int blocks = 2, threads = 8;

    int a = rand() % 10 + 1;
    int *dA, *hA;
    int size = 16 * sizeof(int);
    
    hA = (int*)malloc(size);
    cudaMalloc(&dA, size);

    saxpy<<<blocks, threads>>>(a, dA);
    
    cudaMemcpy(hA, dA, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16; i++) {
        cout << hA[i] << " ";
    }
    cout << endl;

    cudaFree(dA);
    return 0;
}