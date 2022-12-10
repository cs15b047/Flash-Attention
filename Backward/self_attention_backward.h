#ifndef SELF_ATTN_BACKWARD_H
#define SELF_ATTN_BACKWARD_H

void self_attention_backward_cpu(const float *Q, const float *K, const float *V, const float *dO, const float *P, 
                                 float* dP, float* dQ, float* dV, float* dK, float* dS, int N, int dim, int batch_size, int num_heads);

#endif // SELF_ATTN_H