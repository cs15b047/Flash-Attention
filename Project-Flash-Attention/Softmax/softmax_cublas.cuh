#ifndef SOFTMAX_CUBLAS_H
#define SOFTMAX_CUBLAS_H

/* Softmax: 
    Both input and output are in managed memory
    input: n x n, output: n x 1 
*/
void softmax_cublas(float *input, float *output, int n);

#endif // SOFTMAX_H