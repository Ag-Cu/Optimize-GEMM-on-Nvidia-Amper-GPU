void lanch_kernel(int kernel_id, float *dev_a, float *dev_b, float *dev_c, int m, int k, int n);

__global__ void mysgemm_v1(int M, int N, int K, float* A, float* B, float* C);