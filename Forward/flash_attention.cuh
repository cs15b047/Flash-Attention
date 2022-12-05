#ifndef FLA_ATTN_CUH
#define FLA_ATTN_CUH


void flash_attention(const float *Q, const float *K, const float *V, float *O, int N, int dim);

#endif // FLA_ATTN_CUH