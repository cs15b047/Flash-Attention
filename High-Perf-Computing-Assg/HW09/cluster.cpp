#include "cluster.h"
#include <cmath>
#include <iostream>

void cluster(const size_t n, const size_t t, const float *arr, const float *centers, float *dists) {
#pragma omp parallel num_threads(t)
    {
    unsigned int tid = omp_get_thread_num();
    float thread_dist = 0.0;
#pragma omp for
        for (size_t i = 0; i < n; i++) {
            thread_dist += std::fabs(arr[i] - centers[tid]);
        }
        dists[tid] = thread_dist;
    }
}
