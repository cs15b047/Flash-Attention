#ifndef SELF_ATTN_CUH
#define SELF_ATTN_CUH


void self_attention(const float *Q, const float *K, const float *V, float *intermediate, float *softmax_result, float *O, int N, int dim);

#endif // SELF_ATTN_CUH