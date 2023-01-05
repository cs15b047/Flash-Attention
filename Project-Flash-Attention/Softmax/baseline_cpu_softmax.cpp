#include "softmax.h"
#include <bits/stdc++.h>

using namespace std;

void softmax_cpu(const float *input, float *output, int n, int dim){
    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int j = 0; j < dim; j++) {
            sum += exp(input[i * dim + j]);
        }
        for (int j = 0; j < dim; j++) {
            output[i * dim + j] = exp(input[i * dim + j]) / sum;
        }
    }
}


