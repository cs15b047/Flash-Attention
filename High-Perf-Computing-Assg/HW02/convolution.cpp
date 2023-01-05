#include "convolution.h"
using namespace std;

void convolve(const float *image, float *output, size_t img_sz, const float *mask, size_t mask_sz) {
    int n = img_sz, m = mask_sz;
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