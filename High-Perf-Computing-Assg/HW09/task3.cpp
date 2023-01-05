#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

float rand_float() {
    float r = rand() / (float) RAND_MAX; // [0, 1]
    return r;
}

int main(int argc, char* argv[]) {
    int nums = stoi(argv[1]);

    int nodes, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);

    float* buffer1 = new float[nums];
    float* buffer2 = new float[nums];
    generate(buffer1, buffer1 + nums, rand_float);
    generate(buffer2, buffer2 + nums, rand_float);

    if(rank == 0) { // rank - 0
        int dst_rank = 1, tag = 0;
        auto start = chrono::high_resolution_clock::now();
        MPI_Send(buffer1, nums, MPI_FLOAT, dst_rank, tag, MPI_COMM_WORLD);
        MPI_Recv(buffer2, nums, MPI_FLOAT, dst_rank, tag + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        float t0 = duration.count() / 1000.0;

        float t1;
        MPI_Recv(&t1, 1, MPI_FLOAT, dst_rank, tag + 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        float total_time = t0 + t1;
        cout << total_time << endl;

    } else if(rank == 1) { // rank - 1
        int dst_rank = 0, tag = 0;
        auto start = chrono::high_resolution_clock::now();
        MPI_Recv(buffer1, nums, MPI_FLOAT, dst_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(buffer2, nums, MPI_FLOAT, dst_rank, tag + 1, MPI_COMM_WORLD);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        float t1 = duration.count() / 1000.0;

        // Send t1 to node 0
        MPI_Send(&t1, 1, MPI_FLOAT, dst_rank, tag + 2, MPI_COMM_WORLD);
    }

    delete[] buffer1;
    delete[] buffer2;

    MPI_Finalize();
    return 0;
}