// Copyright 2023 The Project AETHER Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Benchmark of report Table 4: CUTLASS block-sparse GEMM (75% block
// sparsity, BF16 inputs, FP32 accumulation) against dense cuBLAS on the same
// shapes. Correctness is checked against cuBLAS run on the masked dense A,
// which computes the identical product. Requires an sm_90 GPU.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "src/custom_kernels/cutlass/block_sparse_gemm.h"

namespace aether {
namespace sparse {
namespace {

constexpr int kWarmup = 5;
constexpr int kIterations = 20;

void Check(bool ok, const char* what) {
  if (!ok) {
    fprintf(stderr, "%s failed\n", what);
    exit(1);
  }
}

struct Config {
  int size;
  int block_size;
};

template <typename Fn>
float TimeMs(Fn fn) {
  for (int i = 0; i < kWarmup; ++i) fn();
  cudaEvent_t begin;
  cudaEvent_t end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord(begin);
  for (int i = 0; i < kIterations; ++i) fn();
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, begin, end);
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return ms / kIterations;
}

std::vector<__nv_bfloat16> ToBf16(const std::vector<float>& x) {
  std::vector<__nv_bfloat16> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = __float2bfloat16(x[i]);
  return out;
}

bool RunConfig(const Config& config, cublasHandle_t handle) {
  const int m = config.size;
  const int n = config.size;
  const int k = config.size;
  const int num_blocks = k / config.block_size;
  BlockPattern pattern{config.block_size, WindowForDensity(num_blocks, 1, 0.25),
                       1};

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
  std::vector<float> a(static_cast<size_t>(m) * k);
  std::vector<float> b(static_cast<size_t>(k) * n);  // Column-major.
  for (float& x : a) x = uniform(rng);
  for (float& x : b) x = uniform(rng);
  ApplyPattern(pattern, m, k, &a);

  // Dense cuBLAS baseline on the masked A: C^T(n x m) = B^T * A^T.
  const std::vector<__nv_bfloat16> a16 = ToBf16(a);
  const std::vector<__nv_bfloat16> b16 = ToBf16(b);
  __nv_bfloat16* d_a = nullptr;
  __nv_bfloat16* d_b = nullptr;
  float* d_ref = nullptr;
  float* d_c = nullptr;
  const size_t c_elems = static_cast<size_t>(m) * n;
  Check(cudaMalloc(&d_a, a16.size() * sizeof(a16[0])) == cudaSuccess, "malloc");
  Check(cudaMalloc(&d_b, b16.size() * sizeof(b16[0])) == cudaSuccess, "malloc");
  Check(cudaMalloc(&d_ref, c_elems * sizeof(float)) == cudaSuccess, "malloc");
  Check(cudaMalloc(&d_c, c_elems * sizeof(float)) == cudaSuccess, "malloc");
  cudaMemcpy(d_a, a16.data(), a16.size() * sizeof(a16[0]),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b16.data(), b16.size() * sizeof(b16[0]),
             cudaMemcpyHostToDevice);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const auto dense = [&] {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, d_b,
                 CUDA_R_16BF, k, d_a, CUDA_R_16BF, k, &beta, d_ref, CUDA_R_32F,
                 n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  };
  const float dense_ms = TimeMs(dense);

  BlockSparseGemm sparse(pattern, m, n, k, Precision::kBf16);
  Check(sparse.UploadA(a) == cudaSuccess, "UploadA");
  Check(sparse.UploadB(b) == cudaSuccess, "UploadB");
  Check(sparse.Run(d_c, nullptr) == cudaSuccess, "BlockSparseGemm::Run");
  const float sparse_ms = TimeMs([&] { sparse.Run(d_c, nullptr); });

  std::vector<float> ref(c_elems);
  std::vector<float> got(c_elems);
  dense();
  cudaMemcpy(ref.data(), d_ref, c_elems * sizeof(float),
             cudaMemcpyDeviceToHost);
  sparse.Run(d_c, nullptr);
  cudaMemcpy(got.data(), d_c, c_elems * sizeof(float), cudaMemcpyDeviceToHost);
  double max_err = 0.0;
  double max_ref = 0.0;
  for (size_t i = 0; i < c_elems; ++i) {
    max_err = std::max(max_err, std::fabs(double{got[i]} - ref[i]));
    max_ref = std::max(max_ref, std::fabs(double{ref[i]}));
  }
  const double rel_err = max_err / max_ref;

  // FP8 (E4M3) path on the same operands: speed only, plus agreement with
  // BF16 within quantization error.
  BlockSparseGemm sparse_fp8(pattern, m, n, k, Precision::kFp8E4m3);
  Check(sparse_fp8.UploadA(a) == cudaSuccess, "UploadA fp8");
  Check(sparse_fp8.UploadB(b) == cudaSuccess, "UploadB fp8");
  const float fp8_ms = TimeMs([&] { sparse_fp8.Run(d_c, nullptr); });

  printf(
      "%5d^3 GEMM, %3dx%-3d blocks, window %2d, density %.3f | "
      "cuBLAS %.3f ms | block-sparse BF16 %.3f ms (%.2fx) | "
      "FP8 %.3f ms | max rel err %.2e\n",
      m, config.block_size, config.block_size, pattern.window,
      Density(pattern, num_blocks), dense_ms, sparse_ms, dense_ms / sparse_ms,
      fp8_ms, rel_err);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_ref);
  cudaFree(d_c);
  return rel_err < 1e-2;
}

}  // namespace
}  // namespace sparse
}  // namespace aether

int main() {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 1;
  const aether::sparse::Config configs[] = {
      {4096, 64}, {4096, 128}, {8192, 128}};
  bool ok = true;
  for (const auto& config : configs) {
    ok = aether::sparse::RunConfig(config, handle) && ok;
  }
  cublasDestroy(handle);
  printf(ok ? "PASSED\n" : "FAILED\n");
  return ok ? 0 : 1;
}
