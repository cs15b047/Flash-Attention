#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n_, const float *mask, std::size_t m_) {
    int n = n_, m = m_;
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            float sum = 0;
            for(int k = 0; k < m; k++) {
                for(int l = 0; l < m; l++) {
                    int x = i + k - (m-1)/2;
                    int y = j + l - (m-1)/2;
                    if(x >= 0 && x < n && y >= 0 && y < n) {
                        sum += image[x*n + y] * mask[k*m + l];
                    } else if((x >= 0 && x < n) || (y >= 0 && y < n)) {// only y or x co-ordinate is out of scope
                        sum += mask[k*m + l];
                    }
                }
            }
            output[i*n + j] = sum;
        }
    }
}