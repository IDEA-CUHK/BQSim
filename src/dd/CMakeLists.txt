if(NOT TARGET ${PROJECT_NAME}-dd)
  file(GLOB_RECURSE DD_HEADERS ${PROJECT_SOURCE_DIR}/include/dd/*.hpp)
  add_subdirectory(${PROJECT_SOURCE_DIR}/include/dd/kernel kernel_bin_dir)
  # add DD Package library
  add_library(
    ${PROJECT_NAME}-dd
    ${DD_HEADERS}
    Complex.cpp
    ComplexNumbers.cpp
    ComplexValue.cpp
    Edge.cpp
    MemoryManager.cpp
    Node.cpp
    NoiseFunctionality.cpp
    Operations.cpp
    RealNumber.cpp
    RealNumberUniqueTable.cpp
    UniqueTableStatistics.cpp)

  target_link_libraries(${PROJECT_NAME}-dd PUBLIC kernels)
  target_link_libraries(${PROJECT_NAME}-dd PUBLIC ${PROJECT_NAME})
  add_library(MQT::CoreDD ALIAS ${PROJECT_NAME}-dd)
  add_library(MQT::${OLD_PROJECT_NAME}_dd ALIAS ${PROJECT_NAME}-dd)
endif()
