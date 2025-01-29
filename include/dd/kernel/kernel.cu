#include "kernel.cuh"
#include <stdio.h>

__global__ void mergeKernel(
  double* z_real_dev, 
  double* z_real_new_dev, 
  double* z_imag_dev, 
  double* z_imag_new_dev
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double z_real_shr[512];
  __shared__ double z_imag_shr[512];

  z_real_shr[threadIdx.x] = z_real_dev[tid];
  z_imag_shr[threadIdx.x] = z_imag_dev[tid];
  __syncthreads();

  z_real_shr[threadIdx.x] += z_real_new_dev[tid];
  z_imag_shr[threadIdx.x] += z_imag_new_dev[tid];
  __syncthreads();

  z_real_dev[tid] = z_real_shr[threadIdx.x];
  z_imag_dev[tid] = z_imag_shr[threadIdx.x];
}

void check(cudaError_t err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    // We don't exit when we encounter CUDA errors in this example.
    // std::exit(EXIT_FAILURE);
  }
}


void call_kernel(
  size_t nDim,
  double* z_real_dev, 
  double* z_real_new_dev, 
  double* z_imag_dev, 
  double* z_imag_new_dev
) {
  mergeKernel<<<nDim / 512, 512>>>(z_real_dev, z_real_new_dev,
    z_imag_dev, z_imag_new_dev
  );

  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();

  CHECK_CUDA_ERROR(err);
}