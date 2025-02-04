cmake_minimum_required(VERSION 3.15)
project(cuquantum LANGUAGES CXX CUDA
DESCRIPTION "Batch quantum circuit simulation using cuQuantum")
set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
    SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
    SET(CUDA_NVCC_FLAGS "-g -G")
endif()
message("Build type: " ${CMAKE_BUILD_TYPE})
# If the Quantum++ installation path was non-standard, i.e., specified by
#
# cmake -B build -DCMAKE_INSTALL_PREFIX=/path/to/installed/qpp
#
# then uncomment the following line and replace the installation path with yours
find_package(CUDA REQUIRED)
# set(CMAKE_PREFIX_PATH "/path/to/installed/qpp")
include_directories(util)
include_directories(include)
# add_compile_options(-lcustatevec)
find_package(qpp REQUIRED)
add_executable(cuquantum main.cu)
target_link_libraries(cuquantum ${QPP_LINK_DEPS} libqpp)
# target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
#                        -lcustatevec
#                        >)

# find_library(CUSTATEVEC_LIBRARY
#     REQUIRED
# )
# message("Found custatevec root: ${CUQUANTUM_ROOT}")
include_directories(${CUQUANTUM_ROOT}/include)
target_link_libraries(cuquantum ${CUSTATEVEC_LIBRARY})
# target_link_libraries(cuquantum -lcustatevec)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(cuquantum PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
endif()
# target_compile_options(main PUBLIC "-lcustatevec")
