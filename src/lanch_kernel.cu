#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "lanch_kernel.cuh"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

void call_my_kernel(int kernel_id, float *dev_a, float *dev_b, float *dev_c, int m, int k, int n) {
  dim3 block(32, 32);
  dim3 grid(CEIL_DIV(m, block.x), CEIL_DIV(n, block.y));
  switch (kernel_id)
  {
  case 1:
    mysgemm_v1<<<grid, block>>>(m, k, n, dev_a, dev_b, dev_c);
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