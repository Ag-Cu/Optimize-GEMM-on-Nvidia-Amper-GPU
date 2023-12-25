#ifndef __SGEMM_V3__
#define __SGEMM_V3__
#include <stdio.h>
#include <stdlib.h>
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// handle bank conflict
template <const int BM = 128, const int BK = 8, const int BN = 128,
          const int TM = 8, const int TN = 8>
__global__ void mysgemm_v4(int M, int N, int K, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C)
{
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int bx = blockIdx.x, by = blockIdx.y;
  const int tid = ty * blockDim.x + tx;

  const int subTM = TM / 2;   // default: 4
  const int subTN = TN / 2;   // default: 4

  __shared__ float As[BK][BM];    // 加载进shared_memory时对A的tile进行转置
  __shared__ float Bs[BK][BN];

  float Csub[TM][TN] = {0.0f};
  float reg_a[2][subTM];
  float reg_b[2][subTN];

  // 在每个128 * 8的tile中的坐标
  int load_a_smem_m = tid / 2;
  int load_a_smem_k = (tid % 2) * 4;
  int load_b_smem_k = tid / 32;        // b tile一行有128个元素，所以y坐标是tid>>5
  int load_b_smem_n = (tid % 32) * 4; 

  const int load_a_gmem_m = by * BM + load_a_smem_m;
  const int load_b_gmem_n = bx * BN + load_b_smem_n;


  for (int bk = 0; bk < (K + BK-1) / BK; ++bk)
  {
    // load A and B tile into shared memory
    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    FLOAT4(As[load_a_smem_k][load_a_smem_m]) = FLOAT4(A[load_gmem_addr]);
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    load_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
    FLOAT4(Bs[load_b_smem_k][load_b_smem_n]) = FLOAT4(B[load_gmem_addr]);

    __syncthreads();

    // sub matrix index in 128*128 C tile
    int first_sub_m = tid / 16;
    int first_sub_n = tid % 16;

    // load into register
    FLOAT4(reg_a[0][0]) = FLOAT4(As[first_sub_m][first_sub_n]);

    // compute Csub, 做外积
    for (int k = 0; k < BK; ++k) {
      for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
          int comp_a_smem_m = ty * TM + m;
          int comp_b_smem_n = tx * TN + n;
          Csub[m][n] += As[k][comp_a_smem_m] * Bs[k][comp_b_smem_n];
        }
      }
    }

    __syncthreads();
  }

  for (int i = 0; i < TM; ++i) {
    int comp_c_gmem_m = by * BM + ty * TM + i;
    for (int j = 0; j < TN; j+=4) {
      int comp_c_gmem_n = bx * BN + tx * TN + j;
      int store_gmem_addr = OFFSET(comp_c_gmem_m, comp_c_gmem_n, N);
      FLOAT4(C[store_gmem_addr]) = FLOAT4(Csub[i][j]);
    }
  }
  __syncthreads();
}
#endif