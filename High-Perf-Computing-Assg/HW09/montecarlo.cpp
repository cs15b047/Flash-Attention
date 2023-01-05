#include "montecarlo.h"
#include <bits/stdc++.h>

using namespace std;

int montecarlo(const size_t n_, const float *x, const float *y, const float radius){
    int threads = omp_get_max_threads();
    int n = n_;
    int thread_inside = 0;

#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        
        #pragma omp for simd reduction(+:thread_inside)
        for (int i = 0; i < n; i++){
            double x_ = x[i], y_ = y[i], r_ = radius;
            thread_inside += ( (x_ * x_ + y_ * y_) <= (r_ * r_) );
        }
    }

    return thread_inside;
}