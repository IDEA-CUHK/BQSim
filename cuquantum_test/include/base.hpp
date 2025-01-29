#pragma once

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf

#include <stdlib.h>           // EXIT_FAILURE
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "qpp/qpp.h"

class Base
{
protected:
  int n_qubit;
  int nSvSize;
  int batchSize;
  int n_batch;
  std::vector<cuDoubleComplex *> matrix_vec;
  std::vector<int> ctrl_vec;
  std::vector<std::vector<int>> target_vec;
  cuDoubleComplex *input_arr;
  cuDoubleComplex *output_arr;
  std::string filename;
public:
  Base(  
    int _n_qubit,
    int _nSvSize,
    int _batchSize,
    std::vector<qpp::QCircuit::double2 *> _mat_vec,
    std::vector<int> _ctrl_vec,
    std::vector<std::vector<int>> _target_vec,
    int _n_batch
  );
  Base(  
    int _n_qubit,
    int _nSvSize,
    int _batchSize,
    std::vector<cuDoubleComplex *> _mat_vec,
    std::vector<int> _ctrl_vec,
    std::vector<std::vector<int>> _target_vec,
    int _n_batch
  );
  virtual void ReadInputs() = 0;
  virtual void BatchSim() = 0;
  virtual cuDoubleComplex * FetchOutput() {
    return output_arr;
  }
  ~Base();
};

Base::Base(
  int _n_qubit,
  int _nSvSize,
  int _batchSize,
  std::vector<qpp::QCircuit::double2 *> _mat_vec,
  std::vector<int> _ctrl_vec,
  std::vector<std::vector<int>> _target_vec, 
  int _n_batch
) : n_qubit(_n_qubit), nSvSize(_nSvSize), batchSize(_batchSize), 
  ctrl_vec(_ctrl_vec), target_vec(_target_vec), n_batch(_n_batch)
{
  cudaMallocHost((void**)&input_arr, _nSvSize * _batchSize * sizeof(cuDoubleComplex));
  // std::ifstream file;
  filename = "../../input_batch/n"+std::to_string(_n_qubit)+".txt";

  for (int gid = 0; gid < _mat_vec.size(); gid++) {
    cuDoubleComplex *matrix;
    cudaMallocHost((void**)&matrix, 4* sizeof(cuDoubleComplex));
    for (size_t mid = 0; mid < 4; mid++)
    {
      matrix[mid] = {_mat_vec[gid][mid].x, _mat_vec[gid][mid].y};
    }
    
    matrix_vec.push_back(matrix);
  }
}

Base::Base(
  int _n_qubit,
  int _nSvSize,
  int _batchSize,
  std::vector<cuDoubleComplex *> _mat_vec,
  std::vector<int> _ctrl_vec,
  std::vector<std::vector<int>> _target_vec, 
  int _n_batch
) : n_qubit(_n_qubit), nSvSize(_nSvSize), batchSize(_batchSize), 
  ctrl_vec(_ctrl_vec), target_vec(_target_vec), n_batch(_n_batch), matrix_vec(_mat_vec)
{
  cudaMallocHost((void**)&input_arr, _nSvSize * _batchSize * sizeof(cuDoubleComplex));
  // std::ifstream file;
  filename = "../../input_batch/n"+std::to_string(_n_qubit)+".txt";

  // for (int gid = 0; gid < _mat_vec.size(); gid++) {
  //   cuDoubleComplex *matrix;
  //   cudaMallocHost((void**)&matrix, 4* sizeof(cuDoubleComplex));
  //   for (size_t mid = 0; mid < 4; mid++)
  //   {
  //     matrix[mid] = {_mat_vec[gid][mid].x, _mat_vec[gid][mid].y};
  //   }
    
  //   matrix_vec.push_back(matrix);
  // }
}



Base::~Base()
{
}
