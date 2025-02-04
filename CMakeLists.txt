# set required cmake version
cmake_minimum_required(VERSION 3.19)

# This avoids googletest complaining that this (IPO) policy is not set
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

project(
  flatdd
  LANGUAGES CXX CUDA
  DESCRIPTION "BQSim - GPU-accelerated Batch Quantum Circuit Simulation using Decision Diagram")

if (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
    SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
    SET(CUDA_NVCC_FLAGS "-g -G")
    set(CMAKE_BUILD_TYPE Debug)
    # SET(CUDA_NVCC_FLAGS "-lineinfo -Xptxas -v --gpu-architecture=sm_80")
endif()
message("Build type: " ${CMAKE_BUILD_TYPE})
set(CMAKE_CXX_FLAGS "-mavx")
# check whether `modulename` is correctly cloned in the `extern` directory.
macro(CHECK_SUBMODULE_PRESENT modulename)
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/${modulename}/CMakeLists.txt")
    message(
      FATAL_ERROR
        "${modulename} submodule not cloned properly. \
        Please run `git submodule update --init --recursive` \
        from the main project directory")
  endif()
endmacro()

set(CUSTATEVEC_LIBRARY $ENV{CUSTATEVEC_LIBRARY})
set(CUQUANTUM_ROOT $ENV{CUQUANTUM_ROOT})
message("-- Found custatevec library: ${CUSTATEVEC_LIBRARY}")
message("-- Found custatevec include: ${CUQUANTUM_ROOT}/include")
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
# check_submodule_present(cxxopts)
check_submodule_present(taskflow)

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(cuquantum_test)
# add_subdirectory(extern/MemoryAllocator.KanameShiki)
# add_subdirectory(extern/cxxopts)
# add_subdirectory(extern/taskflow)
if(NOT BUILD_DDSIM_TESTS)
  set(BUILD_MQT_CORE_TESTS
      OFF
      CACHE BOOL "")
endif()

# if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#   check_submodule_present(cxxopts)
#   add_subdirectory(apps)
# endif()
