#ifndef SELF_ATTN_H
#define SELF_ATTN_H

void self_attention(const float *Q, const float *K, const float *V, float *O, int N, int dim);

#endif // SELF_ATTN_H