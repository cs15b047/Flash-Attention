#include<bits/stdc++.h>
#include "msort.h"

using namespace std;

int generateInt() {
    int r = rand() % 1001;
    r = 2 * r - 1000;
    return r;
}

void print_array(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    int n = stoi(argv[1]);
    int num_threads = stoi(argv[2]);
    int threshold = stoi(argv[3]);

    omp_set_num_threads(num_threads);
    omp_set_nested(1);

    int* arr = new int[n];
    generate(arr, arr + n, generateInt);

    auto start = chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        #pragma omp single
        {
            msort(arr, n, threshold);
        }
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    vector<int> v(arr, arr + n);
    sort(v.begin(), v.end());

    for (int i = 0; i < n; i++) {
        if (arr[i] != v[i]) {
            cout << "Error at index " << i << endl;
            break;
        }
    }

    cout << arr[0] << endl;
    cout << arr[n - 1] << endl;
    cout << duration.count()/1000.0 << endl;

    delete[] arr;
    v.clear();
}