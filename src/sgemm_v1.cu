#include<stdio.h>
#include<stdlib.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]

// naive version
__global__  __launch_bounds__(1024)
void mysgemm_v1(int M, int N, int K, float* A, float* B, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;

    float sum = 0.0f;
    for (int k_count = 0; k_count<K; k_count++){
        sum += A(tx, k_count) * B(k_count, ty);
    }
    C(tx,ty) = sum;
}