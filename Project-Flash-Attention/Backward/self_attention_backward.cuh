#ifndef SELF_ATTN_BACKWARD_CUH
#define SELF_ATTN_BACKWARD_CUH


void self_attention_backward(const float *Q, const float *K, const float *V, const float *dO, const float *P, 
    float* dP, float* dQ, float* dV, float* dK, float* dS, float* rowsums, int N, int dim, int batch_size, int num_heads);

void softmax_backward(const float *P, const float* dP, float* dS, float* rowsums, int N, int batch_size, int num_heads);

void softmax_backward1(const float *P, const float* dP, float* dS, float* rowsums, int N, int batch_size, int num_heads);
void softmax_backward2(const float *P, const float* dP, float* dS, float* rowsums, int N, int batch_size, int num_heads);
void softmax_backward3(const float *P, const float* dP, float* dS, float* rowsums, int N, int batch_size, int num_heads);

#endif // SELF_ATTN_BACKWARD_CUH