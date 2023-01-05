#include "optimize.h"
#include <immintrin.h>
#include <omp.h>

using namespace std;

data_t *get_vec_start(vec *v) {
    return v->data;
}

void optimize1(vec *v, data_t *dest) {
    int length = v->len;
    data_t *d = get_vec_start(v);
    data_t temp = IDENT;
    for(int i = 0; i < length; i++) {
        temp = temp OP d[i];
    }
    *dest= temp;
}

void optimize2(vec *v, data_t *dest) {
    int length = v->len;
    data_t *d = get_vec_start(v);
    data_t temp = IDENT;
    int i;
    for(i = 0; i < length - 1; i+=2) {
        temp = (temp OP d[i]) OP d[i+1];
    }
    for(; i < length; i++) {
        temp = temp OP d[i];
    }

    *dest= temp;
}

void optimize3(vec *v, data_t *dest) {
    int length = v->len;
    data_t *d = get_vec_start(v);
    data_t temp = IDENT;
    int i;
    for(i = 0; i < length - 1; i+=2) {
        temp = temp OP (d[i] OP d[i+1]);
    }
    for(; i < length; i++) {
        temp = temp OP d[i];
    }

    *dest= temp;
}

void optimize4(vec *v, data_t *dest) {
    int length = v->len;
    data_t *d = get_vec_start(v);
    data_t temp0 = IDENT, temp1 = IDENT;
    int i;
    for(i = 0; i < length - 1; i+=2) {
        temp0 = temp0 OP d[i];
        temp1 = temp1 OP d[i+1];
    }
    for(; i < length; i++) {
        temp0 = temp0 OP d[i];
    }

    *dest= temp0 OP temp1;
}

void optimize5(vec *v, data_t *dest) {
    int length = v->len;
    data_t *d = get_vec_start(v);
    data_t temp0 = IDENT, temp1 = IDENT, temp2 = IDENT;
    int i;
    for(i = 0; i < length - 2; i+=3) {
        temp0 = temp0 OP d[i];
        temp1 = temp1 OP d[i+1];
        temp2 = temp2 OP d[i+2];
    }
    for(; i < length; i++) {
        temp0 = temp0 OP d[i];
    }

    *dest= temp0 OP temp1 OP temp2;
}
