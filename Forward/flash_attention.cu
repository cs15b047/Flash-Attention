
#include <cublas_v2.h>
#include <stdio.h>
#include "flash_attention.cuh"


void print_matrix(const float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

__host__ void flash_attention(const float *Q, const float *K, const float *V, float *O, int N, int dim){

    

}
    

