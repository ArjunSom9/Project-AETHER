# Copyright 2023 The Project AETHER Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CUDA C++ kernels of Project AETHER (Hopper TMA pipeline, CUTLASS GEMMs).
#
# Toolchain (all inside the June 5 - August 25, 2023 window):
#   CMake >= 3.26 (3.26.4, 2023-06), CUDA 12.2 Update 1 (12.2.128,
#   2023-07-31), CUTLASS commit 4575443d ("CUTLASS 3.2 (#1024)", 2023-08-08).
#
# Build from project_aether/:
#   cmake -S src/custom_kernels -B src/custom_kernels/build \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build src/custom_kernels/build -j
#   ./src/custom_kernels/build/dsmc_collide_tma_test
#   ./src/custom_kernels/build/block_sparse_gemm_benchmark

cmake_minimum_required(VERSION 3.26)
project(aether_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# WGMMA and TMA need the architecture-specific sm_90a target.
set(CMAKE_CUDA_ARCHITECTURES OFF)
add_compile_options(
  "$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a>"
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

find_package(CUDAToolkit 12.2 REQUIRED)

include(FetchContent)
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG 4575443d446807b8e9d7f77185452899020ac8c9  # 2023-08-08
  GIT_SHALLOW FALSE)
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
  # Header-only use: populate without configuring CUTLASS's own build.
  FetchContent_Populate(cutlass)
endif()

# Sources include project headers as "src/custom_kernels/...".
get_filename_component(AETHER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_library(aether_cuda SHARED
  cuda/dsmc_collide_tma.cu
  cutlass/block_sparse_gemm.cu)
target_include_directories(aether_cuda PUBLIC
  ${AETHER_ROOT}
  ${cutlass_SOURCE_DIR}/include)
target_link_libraries(aether_cuda PUBLIC CUDA::cudart CUDA::cuda_driver)
set_target_properties(aether_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)

add_executable(dsmc_collide_tma_test cuda/dsmc_collide_tma_test.cu)
target_link_libraries(dsmc_collide_tma_test PRIVATE aether_cuda)

add_executable(block_sparse_gemm_benchmark
  cutlass/block_sparse_gemm_benchmark.cu)
target_link_libraries(block_sparse_gemm_benchmark PRIVATE
  aether_cuda CUDA::cublas)
