#include <iostream>
#include <cublas_v2.h>
#include <vector>

#include "cuda_helper.h"
#include "lanch_kernel.cuh"

using namespace std;

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    cout << "Usage: ./sgemm <kernel_id>" << endl;
    exit(1);
  }

  vector<int> test_cases = {1024, 2048, 4096};
  int max_size = test_cases[test_cases.size() - 1];
  int element_size = sizeof(float);
  int kernel_id = atoi(argv[1]);

  cublasHandle_t err;
  cublasCreate(&err);
  float alpha = 1.0f;
  float beta = 0.0f;

  float *a = (float *)malloc(max_size * max_size * element_size);
  float *b = (float *)malloc(max_size * max_size * element_size);
  float *c = (float *)malloc(max_size * max_size * element_size);
  float *c_ref = (float *)malloc(max_size * max_size * element_size);

  memset(a, 0, max_size * max_size * element_size);
  memset(b, 0, max_size * max_size * element_size);
  memset(c, 0, max_size * max_size * element_size);
  memset(c_ref, 0, max_size * max_size * element_size);

  float *dev_a, *dev_b, *dev_c, *dev_c_ref;
  CHECK(cudaMalloc((void **)&dev_a, max_size * max_size * element_size));
  CHECK(cudaMalloc((void **)&dev_b, max_size * max_size * element_size));
  CHECK(cudaMalloc((void **)&dev_c, max_size * max_size * element_size));
  CHECK(cudaMalloc((void **)&dev_c_ref, max_size * max_size * element_size));

  initialData(a, max_size * max_size);
  initialData(b, max_size * max_size);

  CHECK(cudaMemcpy(dev_a, a, max_size * max_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_b, b, max_size * max_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dev_c, 0, max_size * max_size * element_size));
  CHECK(cudaMemset(dev_c_ref, 0, max_size * max_size * element_size));
  cudaDeviceSynchronize();

  if (kernel_id != 0)
  {
    for (auto size : test_cases)
    {
      cublasSgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, dev_a, size, dev_b, size, &beta, dev_c_ref, size);
      cudaDeviceSynchronize();
      CHECK(cudaMemcpy(c_ref, dev_c_ref, size * size, cudaMemcpyDeviceToHost));
      cudaDeviceSynchronize();
      lanch_kernel(kernel_id, dev_a, dev_b, dev_c, size, size, size);
      cudaDeviceSynchronize();
      CHECK(cudaMemcpy(c, dev_c, size * size, cudaMemcpyDeviceToHost));
      cudaDeviceSynchronize();
      checkResult(c_ref, c, size * size);
    }
  }
  else
  {
    for (auto size : test_cases)
    {
      lanch_kernel(0, dev_a, dev_b, dev_c, size, size, size);
      cudaDeviceSynchronize();
    }
  }

  CHECK(cudaFree(dev_a));
  CHECK(cudaFree(dev_b));
  CHECK(cudaFree(dev_c));
  CHECK(cudaFree(dev_c_ref));

  free(a);
  free(b);
  free(c);
  free(c_ref);

  return 0;
}
