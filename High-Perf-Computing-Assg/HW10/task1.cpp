#include <bits/stdc++.h>
#include "optimize.h"

using namespace std;

float measure(void (func)(vec*, data_t*), vec *v) {
    int64_t avg_time = 0;
    for(int i = 0; i < 10; i++) {
        data_t dest = IDENT;
        auto start = chrono::high_resolution_clock::now();
        func(v, &dest);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        avg_time += duration.count();
        
        if(i == 9) cout << dest << endl;
    }
    avg_time /= 10;

    float time_in_ms = avg_time / 1000.0;

    cout << time_in_ms << endl;

    return time_in_ms;
}

int main(int argc, char* argv[]) {
    int n = stoi(argv[1]);
    vec v(n);
    v.data = new data_t[n];
    generate(v.data, v.data + n, []() { return (data_t)(1); });

    measure(optimize1, &v);
    measure(optimize2, &v);
    measure(optimize3, &v);
    measure(optimize4, &v);
    measure(optimize5, &v);

    return 0;
}