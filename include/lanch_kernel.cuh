#ifndef _SGEMM_H_
#define _SGEMM_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "sgemm_v1.cuh"
#include "sgemm_v2.cuh"
#include "sgemm_v3.cuh"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

void call_my_kernel(int kernel_id, float *dev_a, float *dev_b, float *dev_c, int m, int k, int n) {
  dim3 block1(32, 32);
  dim3 grid1(CEIL_DIV(m, block1.x), CEIL_DIV(n, block1.y));

  const int BM = 128, BN = 128, TM = 8, TN = 8;
  dim3 block3(BN/TN, BM/TM);
  dim3 grid3(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
  switch (kernel_id)
  {
  case 1:
    mysgemm_v1<<<grid1, block1>>>(m, k, n, dev_a, dev_b, dev_c);
    break;
  case 2:
    mysgemm_v2<<<grid1, block1>>>(m, k, n, dev_a, dev_b, dev_c);
    break;
  case 3:
    mysgemm_v3<<<grid3, block3>>>(m, k, n, dev_a, dev_b, dev_c);
    break;
  
  default:
    break;
  }
}

void lanch_kernel(int kernel_id, float *dev_a, float *dev_b, float *dev_c, int m, int k, int n) {

  float alpha = 1.0f;
  float beta = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop); 

  cudaEventRecord(start, 0);
  if (kernel_id == 0) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_a, m, dev_b, k, &beta, dev_c, m);
    cublasDestroy(handle);
  } else {
    call_my_kernel(kernel_id, dev_a, dev_b, dev_c, m, k, n);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf( "Size:  %d * %d \t\tTime: %f ms \t\tPerformance: %f GFLOPS.\n", m, m, elapsedTime, 2.0*m*m*m/(elapsedTime*1e6));
}


#endif