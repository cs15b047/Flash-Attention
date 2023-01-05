#include<bits/stdc++.h>

using namespace std;

int main(int argc, char const *argv[]) {
    int n = stoi(argv[1]);
    for (int i = 0; i <= n; i++) {
        printf("%d", i);
        if(i < n) {
            printf(" ");
        } else {
            printf("\n");
        }
    }
    for (int i = 0; i <= n; i++) {
        cout <<  n - i;
        if(i < n) {
            cout << " ";
        } else {
            cout << endl;
        }
    }
    
    return 0;
}