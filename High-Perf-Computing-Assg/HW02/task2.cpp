#include<bits/stdc++.h>
#include "convolution.h"

using namespace std;
typedef long long int ll;

float generate_random_image() {
    float r = (float)rand()/(float)RAND_MAX;
    r = -10 + 20*r;
    return r;
}

float generate_random_mask() {
    float r = (float)rand()/(float)RAND_MAX;
    r = -1 + 2*r;
    return r;
}

int main(int argc, char* argv[]) {
    int n = stoi(argv[1]);
    int m = stoi(argv[2]);

    srand(time(nullptr));

    float* image = new float[n*n];
    float* mask = new float[m*m];
    float* output = new float[n*n];

    // randomly generate image and mask
    generate(image, image + n*n, generate_random_image);
    generate(mask, mask + m*m, generate_random_mask);

    auto start = chrono::high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    double time_taken_in_ms = (double)duration.count() / 1000.0;
    cout << time_taken_in_ms << endl;
    cout << output[0] << endl;
    cout << output[n*n - 1] << endl;

    delete[] image;
    delete[] mask;
    delete[] output;

}