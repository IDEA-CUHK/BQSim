#pragma once  
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc. 
#include <cuComplex.h> // cuDoubleComplex 
#include <custatevec.h> // custatevecApplyMatrix
#include <iostream>	
#include <cmath>

bool CheckEquivalent(cuDoubleComplex* a, cuDoubleComplex* b, size_t arr_size) { 
  double eps = 1e-3;
  for (size_t i = 0;i < arr_size; i++ ) {
  // std::cout <<a[i].x<<" "<<a[i].y<<" "<<b[i].x<<" "<<b[i].y<<std::endl;

    if (abs(a[i].x-b[i].x) > eps*abs(a[i].x) || abs(a[i].y-b[i].y) > eps*abs(a[i].y) ) {
      return false;
    }
  }	
  return true;
}

/*
  M and N are for the old matrix
*/
void Transpose(cuDoubleComplex* in_arr, cuDoubleComplex* out_arr, int M, int N) {
  // cuDoubleComplex* out_arr;
  // cudaMallocHost((void**)&out_arr, M * N * sizeof(cuDoubleComplex));

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      out_arr[i + j * M] = in_arr[i * N + j];
    }
  }

}