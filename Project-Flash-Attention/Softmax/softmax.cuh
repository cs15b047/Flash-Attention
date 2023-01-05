#ifndef SOFTMAX_H
#define SOFTMAX_H

/* Softmax: 
    Both input and output are in managed memory
    input: n x n, output: n x 1 
*/
void softmax1(float *input, float *output, int N,int dim);
void softmax2(float *input, float *output, int N,int dim);
void softmax3(float *input, float *output, int N,int dim);



void fused_softmax(float *input, float *output, int N,int dim,int row_per_block);

#endif // SOFTMAX_H