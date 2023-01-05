#include "msort.h"
#include <bits/stdc++.h>

using namespace std;

void merge(const int* arr1, const int* arr2, int* dest, int n1, int n2) {
    int i = 0, j = 0, k = 0;
    while (i < n1 && j < n2) {
        if (arr1[i] < arr2[j]) {
            dest[k] = arr1[i];
            i++;
        } else {
            dest[k] = arr2[j];
            j++;
        }
        k++;
    }
    while (i < n1) {
        dest[k] = arr1[i];
        i++;
        k++;
    }
    while (j < n2) {
        dest[k] = arr2[j];
        j++;
        k++;
    }
}

void insertion_sort(int* arr, int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void msort(int* arr, const std::size_t n, const size_t threshold){
    if(n < threshold) {
        insertion_sort(arr, n);
        return;
    }
    
    

    int *arr1 = new int[n/2], *arr2 = new int[n - n/2];
    int size1 = n/2, size2 = n - n/2;

    #pragma omp task
    {
        memcpy(arr1, arr, size1 * sizeof(int));
        msort(arr1, size1, threshold);
    }
    #pragma omp task
    {
        memcpy(arr2, arr + size1, size2 * sizeof(int));
        msort(arr2, size2, threshold);
    }
    #pragma omp taskwait

    merge(arr1, arr2, arr, size1, size2);

    delete[] arr1;
    delete[] arr2;
}
