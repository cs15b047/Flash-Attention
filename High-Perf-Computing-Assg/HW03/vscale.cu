#include "vscale.cuh"

__global__ 
void vscale(const float *a, float *b, unsigned int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        b[idx] *= a[idx];
    }
}