#ifndef SOFTMAX_H
#define SOFTMAX_H

/* Softmax: 
    Both input and output are in managed memory
    input: n x n, output: n x 1 
*/
__host__ void softmax(float *input, float *output, int n);

#endif // SOFTMAX_H