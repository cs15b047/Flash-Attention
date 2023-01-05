#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

using namespace std;

float generate_float() {
    float r = rand() / (float)RAND_MAX;
    return 2 * r - 1;
}

int main(int argc, char *argv[]) {
    int n = stoi(argv[1]);

    thrust::host_vector<float> h_vec(n);
    thrust::generate(h_vec.begin(), h_vec.end(), generate_float);

    thrust::device_vector<float> d_vec = h_vec;

    auto start = chrono::high_resolution_clock::now();

    float result = thrust::reduce(thrust::device, d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
    
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();

    cout << result << endl;
    cout << duration / 1000.0 << endl;

}