#ifndef SELF_ATTN_BACKWARD_CUH
#define SELF_ATTN_BACKWARD_CUH


void self_attention_backward(const float *Q, const float *K, const float *V, const float *dO, const float *P, 
                                 float* dP, float* dQ, float* dV, float* dK, float* dS, int N, int dim);

void softmax_backward(const float *P, const float* dP, float* dS, int N);

#endif // SELF_ATTN_BACKWARD_CUH