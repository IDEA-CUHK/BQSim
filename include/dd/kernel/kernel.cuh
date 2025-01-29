#include <cuda.h>
#include "cuda_runtime.h"  
#include <iostream>
#include <cassert>
#include <limits>

void call_kernel(
  size_t nDim,
  double* z_real_dev, 
  double* z_real_new_dev, 
  double* z_imag_dev, 
  double* z_imag_new_dev
);


__global__ void mergeKernel(
  double* z_real_dev, 
  double* z_real_new_dev, 
  double* z_imag_dev, 
  double* z_imag_new_dev
);

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char *const func, const char *const file,
           const int line);
