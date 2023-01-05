#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
using namespace cub;
using namespace std;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

float generate_float() {
    float r = rand() / (float)RAND_MAX;
    return 2 * r - 1;
}

int main(int argc, char* argv[]) {
    int num_items = stoi(argv[1]);
    // Set up host arrays
    float *h_in = new float[num_items];
    for (int i = 0; i < num_items; i++) {
        h_in[i] = generate_float();
    }

    float sum = 0;
    for (int i = 0; i < num_items; i++)
        sum += h_in[i];

    // Set up device arrays
    float* d_in = NULL;
    (g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) * num_items));
    // Initialize device input
    (cudaMemcpy(d_in, h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice));
    // Setup device output array
    float* d_sum = NULL;
    (g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1));
    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    (DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items));
    (g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Do the actual reduce operation
    (DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_in_ms;
    cudaEventElapsedTime(&time_in_ms, start, stop);

    float gpu_sum;
    (cudaMemcpy(&gpu_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost));
    // Check for correctness
    bool close = (fabs(gpu_sum - sum) < 10);
    const char *msg = (close ? "Test passed." : "Test falied.");
    // printf("%s %f %f\n", msg, gpu_sum, sum);
    printf("%f\n", gpu_sum);
    printf("%f\n", time_in_ms);

    // Cleanup
    if (d_in) (g_allocator.DeviceFree(d_in));
    if (d_sum) (g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) (g_allocator.DeviceFree(d_temp_storage));
    
    return 0;
}