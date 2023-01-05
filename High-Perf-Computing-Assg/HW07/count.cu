#include "count.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <iostream>


void count(const thrust::device_vector<int>& d_in_const, thrust::device_vector<int>& values, thrust::device_vector<int>& counts) {
    thrust::device_vector<int> d_in = d_in_const;
    thrust::sort(d_in.begin(), d_in.end());
    int size = thrust::reduce_by_key(thrust::device, d_in.begin(), d_in.end(), thrust::constant_iterator<int>(1), 
                    values.begin(), counts.begin()).first - values.begin();
    values.resize(size);
    counts.resize(size);
}