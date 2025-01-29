cmake_minimum_required(VERSION 2.8)
project(kernels)

find_package(CUDA REQUIRED)

# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
SET(CUDA_NVCC_FLAGS "-lineinfo -Xptxas -v --gpu-architecture=sm_80")
# SET(CUDA_NVCC_FLAGS "-g -G")
cuda_add_library(kernels STATIC kernel.cuh kernel.cu)
