#include<iostream>
using namespace std;

__global__
void factorial() {
    int n = threadIdx.x + 1;
    int fact = 1;
    for(int i = 1; i <= n; i++) {
        fact *= i;
    }
    printf("%d!=%d\n", n, fact);
}

int main() {
    int n = 8;
    factorial<<<1, n>>>();
    cudaDeviceSynchronize();
    return 0;
}