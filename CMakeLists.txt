cmake_minimum_required(VERSION 3.18)
project(sofie LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Location of the standalone SOFIE sources (https://github.com/ML4EP/SOFIE).
# Defaults to the git submodule SOFIE/ of this repository.
set(SOFIE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SOFIE" CACHE PATH "Path to SOFIE source tree")
if(NOT EXISTS "${SOFIE_SOURCE_DIR}/core/CMakeLists.txt")
  message(FATAL_ERROR "SOFIE sources not found in ${SOFIE_SOURCE_DIR}. "
                      "Run: git submodule update --init   (or pass -DSOFIE_SOURCE_DIR=<path>).")
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

include(GNUInstallDirs)
add_subdirectory("${SOFIE_SOURCE_DIR}/core" "${CMAKE_BINARY_DIR}/sofie_core" EXCLUDE_FROM_ALL)
add_subdirectory("${SOFIE_SOURCE_DIR}/parsers" "${CMAKE_BINARY_DIR}/sofie_parsers" EXCLUDE_FROM_ALL)

set(SOFIE_BINDINGS_SOURCES
  src/bindings.cpp
  src/types.cpp
  src/roperator.cpp
  src/rmodel.cpp
  src/rmodelparser_onnx.cpp
  src/operators/gemm.cpp
  src/operators/conv.cpp
  src/operators/conv_transpose.cpp
  src/operators/relu.cpp
  src/operators/sigmoid.cpp
  src/operators/tanh.cpp
  src/operators/identity.cpp
  src/operators/swish.cpp
  src/operators/selu.cpp
  src/operators/leaky_relu.cpp
  src/operators/elu.cpp
  src/operators/transpose.cpp
  src/operators/reshape.cpp
  src/operators/concat.cpp
  src/operators/softmax.cpp
  src/operators/pool.cpp
  src/operators/batch_normalization.cpp
  src/operators/layer_normalization.cpp
  src/operators/rnn.cpp
  src/operators/gru.cpp
  src/operators/lstm.cpp
  src/operators/erf.cpp
  src/operators/hard_swish.cpp
  src/operators/not.cpp
  src/operators/gelu.cpp
  src/operators/hard_sigmoid.cpp
  src/operators/where.cpp
  src/operators/clip.cpp
  src/operators/cast.cpp
  src/operators/expand.cpp
  src/operators/gather.cpp
  src/operators/split.cpp
  src/operators/slice.cpp
  src/operators/group_norm.cpp
  src/operators/sdpa.cpp
  src/operators/tile.cpp
  src/operators/pad.cpp
  src/operators/cumsum.cpp
  src/operators/instance_normalization.cpp
  src/operators/basic_unary.cpp
  src/operators/basic_nary.cpp
  src/operators/reduce.cpp
  src/operators/comparison.cpp
  src/operators/basic_binary.cpp
)
pybind11_add_module(_sofie ${SOFIE_BINDINGS_SOURCES})
target_include_directories(_sofie PRIVATE src)
target_link_libraries(_sofie PRIVATE SOFIE_core SOFIE_parsers)

# The generation of the portable code for heterogeneous architectures via alpaka is opt-in,
# and names the backend that the build is for: none (the default), cuda (nvidia) or hip (amd). 
# The generator itself is the same for all of them (the generated Session is templated on 
# the accelerator): the backend is chosen when the
# generated code is compiled. It is recorded in the build (sofie.alpaka_backend).
#   pip install . -C alpaka=cuda   (see packaging/sofie_build.py)
#   SOFIE_ALPAKA_BACKEND=cuda pip install .
set(PYSOFIE_ALPAKA_BACKEND "none" CACHE STRING "alpaka backend of this build: none, cuda or hip")
set_property(CACHE PYSOFIE_ALPAKA_BACKEND PROPERTY STRINGS none cuda hip)
if(PYSOFIE_ALPAKA_BACKEND STREQUAL "none" AND DEFINED ENV{SOFIE_ALPAKA_BACKEND})
  set(PYSOFIE_ALPAKA_BACKEND "$ENV{SOFIE_ALPAKA_BACKEND}")
endif()
set(_pysofie_alpaka_backends none cuda hip)
if(NOT PYSOFIE_ALPAKA_BACKEND IN_LIST _pysofie_alpaka_backends)
  message(FATAL_ERROR "PYSOFIE_ALPAKA_BACKEND must be none, cuda or hip, not '${PYSOFIE_ALPAKA_BACKEND}'")
endif()
if(NOT PYSOFIE_ALPAKA_BACKEND STREQUAL "none")
  target_compile_definitions(_sofie PRIVATE PYSOFIE_ENABLE_ALPAKA PYSOFIE_ALPAKA_BACKEND="${PYSOFIE_ALPAKA_BACKEND}")
endif()
message(STATUS "alpaka backend: ${PYSOFIE_ALPAKA_BACKEND}")

set_target_properties(_sofie SOFIE_core SOFIE_parsers PROPERTIES
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "$ORIGIN")
install(TARGETS _sofie SOFIE_core SOFIE_parsers LIBRARY DESTINATION sofie)
