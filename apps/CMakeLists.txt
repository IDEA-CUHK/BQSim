add_subdirectory("${PROJECT_SOURCE_DIR}/extern/cxxopts" "extern/cxxopts"
                 EXCLUDE_FROM_ALL)
# the following sets the SYSTEM flag for the include dirs of the cxxopts libs to
# cmake-lint: disable=C0307

set_target_properties(
  cxxopts PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                     $<TARGET_PROPERTY:cxxopts,INTERFACE_INCLUDE_DIRECTORIES>)

# macro to add a executable with the given libraries
macro(ADD_SIM_EXECUTABLE appname)
  add_executable(${appname}
                 ${CMAKE_CURRENT_SOURCE_DIR}/${appname}.cpp)
  target_link_libraries(${appname} PRIVATE ${PROJECT_NAME}
                                                           ${ARGN})
  target_link_libraries(${appname} PRIVATE OpenMP::OpenMP_CXX)
endmacro()

macro(ADD_CUSIM_EXECUTABLE appname)
  add_executable(${appname}
                 ${CMAKE_CURRENT_SOURCE_DIR}/${appname}.cu)
  target_link_libraries(${appname} PRIVATE ${PROJECT_NAME}
                                                           ${ARGN})
  target_link_libraries(${appname} PRIVATE -lcustatevec)
endmacro()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)
link_libraries(Threads::Threads)

find_package(OpenMP REQUIRED)
add_compile_options(-fopenmp)

add_sim_executable(DDSIM cxxopts::cxxopts)
add_sim_executable(FlatDD cxxopts::cxxopts)
add_cusim_executable(BQSim cxxopts::cxxopts)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(BQSim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
endif()
