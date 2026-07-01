cmake_minimum_required(VERSION 3.20)
project(triton-perlane-retire LANGUAGES CXX)

# Out-of-tree Triton plugin pass. Point these at a Triton source/build tree
# (in-source builds have both in one place) and its MLIR/LLVM distribution:
#   -DTRITON_SRC_DIR=/path/to/triton
#   -DTRITON_BUILD_DIR=/path/to/triton-build   (defaults to TRITON_SRC_DIR)
#   -DTRITON_LLVM_DIR=/path/to/llvm-dist       (the ~/.triton/llvm/... dir)
set(TRITON_SRC_DIR "" CACHE PATH "Triton source tree")
set(TRITON_BUILD_DIR "${TRITON_SRC_DIR}" CACHE PATH "Triton build tree")
set(TRITON_LLVM_DIR "" CACHE PATH "LLVM/MLIR distribution used to build Triton")

if(NOT TRITON_SRC_DIR OR NOT TRITON_LLVM_DIR)
  message(FATAL_ERROR "Set -DTRITON_SRC_DIR and -DTRITON_LLVM_DIR")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_PREFIX_PATH "${TRITON_LLVM_DIR}")
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${TRITON_SRC_DIR}/include
  ${TRITON_SRC_DIR}            # for nvidia/... third_party include forms
  ${TRITON_BUILD_DIR}/include  # generated headers (triton/Version.h)
  ${CMAKE_CURRENT_BINARY_DIR}  # local Passes.h.inc
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name PerLaneRetire)
add_public_tablegen_target(PerLaneRetireIncGen)

add_library(PerLaneLoopRetirement SHARED PerLaneLoopRetirement.cpp)
add_dependencies(PerLaneLoopRetirement PerLaneRetireIncGen)
target_compile_options(PerLaneLoopRetirement PRIVATE
  -fvisibility=hidden -fno-exceptions -fno-rtti)

# Link against libtriton.so so the plugin carries a DT_NEEDED edge and its
# MLIR/LLVM + plugin-glue symbols resolve at dlopen (the same library is
# already loaded in the host process).
set(TRITON_LIB "${TRITON_BUILD_DIR}/python/triton/_C/libtriton.so"
    CACHE FILEPATH "Path to libtriton.so (must be built with TRITON_EXT_ENABLED)")
target_link_libraries(PerLaneLoopRetirement PRIVATE "${TRITON_LIB}")
