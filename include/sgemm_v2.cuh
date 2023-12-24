#ifndef __SGEMM_V2__
#define __SGEMM_V2__
#include<stdio.h>
#include<stdlib.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]


// shared memory tiling, each thread block computes a tile of C
template<int BLOCK_SIZE = 32>
__global__  __launch_bounds__(1024)
void mysgemm_v2(int M, int N, int K, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  int lda = M, ldb = K, ldc = M;
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;

  // 计算该线程所负责的C的元素坐标
  int row = by * BLOCK_SIZE + ty;
  int col = bx * BLOCK_SIZE + tx;

  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  float Csub = 0.0f;
  for (int t = 0; t < K; t += BLOCK_SIZE) {
    As[ty][tx] = A(row, t + tx);
    Bs[ty][tx] = B(t + ty, col);
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }
  C(row, col) = Csub;
}

#endif