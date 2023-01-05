#include "reduce.h"

using namespace std;

float reduce(const float* arr, const size_t l, const size_t r) {
    float sum = 0;
    #pragma omp parallel for simd reduction(+:sum)
    for (size_t i = l; i < r; i++) {
        sum += arr[i];
    }
    return sum;
}