#include <omp.h>
#include <iostream>

using namespace std;

int factorial(int num) {
    if(num == 0) return 1;
    int result = 1;
    for (int i = 1; i <= num; i++) {
        result *= i;
    }
    return result;
}

int main() {
    omp_set_num_threads(4);

    int n = 8;

    #pragma omp parallel
    {
        #pragma omp single
        cout << "Number of threads: " << omp_get_num_threads() << endl;
        cout << "I am thread " << omp_get_thread_num() << endl;
    }

    #pragma omp parallel for
    for(int i = 1; i <= n; i++) {
        cout << i << "!=" << factorial(i) << endl;
    }
    
}