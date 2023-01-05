#include "montecarlo.h"
#include <bits/stdc++.h>

using namespace std;

float radius = 100;

float generate_float(){
    float r = (float)rand() / (float)RAND_MAX; // 0,1
    r = (2 * r - 1); // -1, 1
    r = r * radius; // -radius, radius
    return r;
}

int main(int argc, char *argv[]) {
    int num_points = stoi(argv[1]);
    int threads = stoi(argv[2]);

    omp_set_num_threads(threads);

    float* x = new float[num_points];
    float* y = new float[num_points];

    generate(x, x + num_points, generate_float);
    generate(y, y + num_points, generate_float);

    auto start = chrono::high_resolution_clock::now();
    
    int points_inside_circle = montecarlo(num_points, x, y, radius);

    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    float pi = 4 * (float)points_inside_circle / (float)num_points;

    cout << pi << endl;
    cout << duration.count()/1000.0 << endl;
}