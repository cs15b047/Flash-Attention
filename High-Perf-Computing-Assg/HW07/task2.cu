#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

#include "count.cuh"

using namespace std;

int generate_num() {
    return (rand() % 501);
}

int main(int argc, char *argv[]) {
    srand(time(nullptr));
    int n = stoi(argv[1]);

    thrust::host_vector<int> h_vec(n);

    thrust::generate(h_vec.begin(), h_vec.end(), generate_num);
    thrust::device_vector<int> d_vec = h_vec;
    thrust::device_vector<int> values(n), counts(n);

    auto start = chrono::high_resolution_clock::now();

    count(d_vec, values, counts);
    
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();

    int last_value = values[values.size() - 1];
    int last_count = counts[counts.size() - 1];

    cout << last_value << endl;
    cout << last_count << endl;
    cout << duration / 1000.0 << endl;

}