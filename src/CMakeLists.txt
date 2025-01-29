if(NOT TARGET nlohmann_json)
  # cmake-lint: disable=C0103
  set(JSON_BuildTests
      OFF
      CACHE INTERNAL "")
  # cmake-lint: disable=C0103
  set(JSON_MultipleHeaders
      OFF
      CACHE INTERNAL "")
  # add nlohmann::json library
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/json" "extern/json" EXCLUDE_FROM_ALL)

  # the following sets the SYSTEM flag for the include dirs of the json libs to suppress warnings
  set_target_properties(
    nlohmann_json PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                             $<TARGET_PROPERTY:nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
  )# cmake-lint: disable=C0307
endif()

if(NOT TARGET ${PROJECT_NAME})
  # main project library
  add_library(
    ${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/include/CircuitOptimizer.hpp
    ${PROJECT_SOURCE_DIR}/include/Definitions.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/Expression.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/ClassicControlledOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/CompoundOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/Control.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/NonUnitaryOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/Operation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/OpType.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/StandardOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/SymbolicOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/parsers/qasm_parser/Parser.hpp
    ${PROJECT_SOURCE_DIR}/include/parsers/qasm_parser/Scanner.hpp
    ${PROJECT_SOURCE_DIR}/include/parsers/qasm_parser/Token.hpp
    ${PROJECT_SOURCE_DIR}/include/Permutation.hpp
    ${PROJECT_SOURCE_DIR}/include/QuantumComputation.hpp
    ${PROJECT_SOURCE_DIR}/include/DDSimulator.hpp
    ${PROJECT_SOURCE_DIR}/include/SwitchSimulator.hpp
    ${PROJECT_SOURCE_DIR}/include/QBatchSimulator.hpp
    # CircuitOptimizer.cpp
    operations/Expression.cpp
    operations/NonUnitaryOperation.cpp
    operations/Operation.cpp
    operations/StandardOperation.cpp
    operations/SymbolicOperation.cpp
    parsers/GRCSParser.cpp
    parsers/QASMParser.cpp
    parsers/QCParser.cpp
    parsers/RealParser.cpp
    parsers/TFCParser.cpp
    parsers/qasm_parser/Parser.cpp
    parsers/qasm_parser/Scanner.cpp
    QuantumComputation.cpp
    DDSimulator.cpp
    SwitchSimulator.cpp)

  # set include directories
  target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include
                                                    ${PROJECT_BINARY_DIR}/include
                                                    ${CUQUANTUM_ROOT}/include)

  target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json)
  target_link_libraries(${PROJECT_NAME} PUBLIC MQT::CoreDD)

  # # add options and warnings to the library
  # target_link_libraries(${PROJECT_NAME} PUBLIC project_options project_warnings)

  # add MQT alias
  add_library(MQT::Core ALIAS ${PROJECT_NAME})
  add_library(MQT::${OLD_PROJECT_NAME} ALIAS ${PROJECT_NAME})
endif()

set(JSON_MultipleHeaders
    OFF
    CACHE INTERNAL "")
set(TF_BUILD_TESTS
    OFF
    CACHE BOOL "")
set(TF_BUILD_EXAMPLES
    OFF
    CACHE BOOL "")
set(TF_BUILD_PROFILER
    OFF
    CACHE BOOL "")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/taskflow" "extern/taskflow")
target_link_libraries(${PROJECT_NAME} PUBLIC Taskflow)
# the following sets the SYSTEM flag for the include dirs of the taskflow libs
# cmake-lint: disable=C0307
set_target_properties(
  Taskflow PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                      $<TARGET_PROPERTY:Taskflow,INTERFACE_INCLUDE_DIRECTORIES>)
# add DD package library
add_subdirectory(dd)
