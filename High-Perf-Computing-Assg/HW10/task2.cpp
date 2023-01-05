#include "reduce.h"
#include<bits/stdc++.h>
#include <mpi.h>

using namespace std;

float rand_float() {
    float r = (float)rand() / RAND_MAX;
    return 2*r - 1;
}

int main(int argc, char* argv[]) {
    int n = stoi(argv[1]);
    int threads = stoi(argv[2]);

    float *arr = new float[n];
    generate(arr, arr + n, rand_float);

    int world_size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    omp_set_num_threads(threads);

    auto start = chrono::high_resolution_clock::now();
    int start_idx = 0, end_idx = start_idx + n;
    float res = reduce(arr, start_idx, end_idx);

    float global_res = 0;
    MPI_Reduce(&res, &global_res, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    if (rank == 0) {
        cout << global_res << endl;
        cout << duration.count()/1000.0 << endl;
    }

    MPI_Finalize();
    return 0;
}