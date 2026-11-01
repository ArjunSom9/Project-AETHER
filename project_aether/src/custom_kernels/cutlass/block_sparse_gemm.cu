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

#include "src/custom_kernels/cutlass/block_sparse_gemm.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

#include <cmath>
#include <vector>

// The CUTLASS 3.2 collective builders are not self-contained (they rely on
// cutlass/gemm/gemm.h and cutlass/arch/mma.h being included first), so these
// headers keep the order of CUTLASS's own Hopper examples instead of
// alphabetical order.
// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
// clang-format on

namespace aether {
namespace sparse {
namespace {

// One CUTLASS 3.x GEMM: row-major A (K-major), column-major B (K-major),
// row-major float C/D, FP32 accumulation, TMA + WGMMA warp-specialized.
template <class ElementAB, class TileShape, class KernelSchedule,
          class EpilogueSchedule>
struct HopperGemm {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int kAlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int kAlignmentC = 128 / cutlass::sizeof_bits<float>::value;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, float,
          float, float, LayoutC, kAlignmentC, float, LayoutC, kAlignmentC,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementAB,
          LayoutA, kAlignmentAB, ElementAB, LayoutB, kAlignmentAB, float,
          TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// 64-row tiles for 64 x 64 blocks; 128-row cooperative tiles otherwise.
using Bf16Tile64 = HopperGemm<cutlass::bfloat16_t,
                              cute::Shape<cute::_64, cute::_128, cute::_64>,
                              cutlass::gemm::KernelTmaWarpSpecialized,
                              cutlass::epilogue::TmaWarpSpecialized>;
using Bf16Tile128 =
    HopperGemm<cutlass::bfloat16_t,
               cute::Shape<cute::_128, cute::_128, cute::_64>,
               cutlass::gemm::KernelTmaWarpSpecializedCooperative,
               cutlass::epilogue::TmaWarpSpecializedCooperative>;
using Fp8Tile64 =
    HopperGemm<cutlass::float_e4m3_t,
               cute::Shape<cute::_64, cute::_128, cute::_128>,
               cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum,
               cutlass::epilogue::TmaWarpSpecialized>;
using Fp8Tile128 =
    HopperGemm<cutlass::float_e4m3_t,
               cute::Shape<cute::_128, cute::_128, cute::_128>,
               cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum,
               cutlass::epilogue::TmaWarpSpecializedCooperative>;

// A strided-batched problem: operand i of the batch starts at
// base + i * batch_stride.
template <class Element>
struct Operand {
  const Element* data;
  int64_t leading_dim;
  int64_t batch_stride;
};

template <class Config>
cudaError_t LaunchGemm(int m, int n, int k, int batch,
                       Operand<typename Config::CollectiveMainloop::ElementA> a,
                       Operand<typename Config::CollectiveMainloop::ElementB> b,
                       float* c, int64_t ldc, int64_t batch_stride_c,
                       float beta, cudaStream_t stream) {
  using Gemm = typename Config::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  const StrideA stride_a =
      cute::make_stride(a.leading_dim, cute::Int<1>{}, a.batch_stride);
  const StrideB stride_b =
      cute::make_stride(b.leading_dim, cute::Int<1>{}, b.batch_stride);
  const StrideC stride_c =
      cute::make_stride(ldc, cute::Int<1>{}, batch_stride_c);

  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                     {m, n, k, batch},
                                     {a.data, stride_a, b.data, stride_b},
                                     {{}, c, stride_c, c, stride_c}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = beta;

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  const size_t workspace_bytes = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;
  if (workspace_bytes > 0 &&
      cudaMallocAsync(&workspace, workspace_bytes, stream) != cudaSuccess) {
    return cudaErrorMemoryAllocation;
  }
  cutlass::Status status = gemm.initialize(arguments, workspace, stream);
  if (status == cutlass::Status::kSuccess) status = gemm.run(stream);
  if (workspace != nullptr) cudaFreeAsync(workspace, stream);
  return status == cutlass::Status::kSuccess ? cudaGetLastError()
                                             : cudaErrorLaunchFailure;
}

// The three launches of one C = A * B.
template <class Config>
cudaError_t RunPattern(const BlockPattern& p, int m, int n, int k,
                       int num_block_rows, int band_width, int padded_k,
                       const void* band, const void* global_cols,
                       const void* global_rows, const void* padded_b, float* c,
                       cudaStream_t stream) {
  using Element = typename Config::CollectiveMainloop::ElementA;
  const int bs = p.block_size;
  const auto* b = static_cast<const Element*>(padded_b);
  // Row r of the unpadded B is row r + window * bs of the padded B.
  const Element* b_rows = b + static_cast<int64_t>(p.window) * bs;

  cudaError_t status = LaunchGemm<Config>(
      bs, n, band_width, num_block_rows,
      {static_cast<const Element*>(band), band_width,
       static_cast<int64_t>(bs) * band_width},
      {b, padded_k, bs}, c, n, static_cast<int64_t>(bs) * n, 0.0f, stream);
  if (status != cudaSuccess || p.global_blocks == 0) return status;

  const int global_width = p.global_blocks * bs;
  status = LaunchGemm<Config>(
      m, n, global_width, 1,
      {static_cast<const Element*>(global_cols), global_width, 0},
      {b_rows, padded_k, 0}, c, n, 0, 1.0f, stream);
  if (status != cudaSuccess) return status;
  return LaunchGemm<Config>(global_width, n, k, 1,
                            {static_cast<const Element*>(global_rows), k, 0},
                            {b_rows, padded_k, 0}, c, n, 0, 1.0f, stream);
}

template <class Element>
cudaError_t Upload(const std::vector<float>& host, void** device) {
  std::vector<Element> converted(host.size());
  for (size_t i = 0; i < host.size(); ++i) converted[i] = Element(host[i]);
  if (*device == nullptr &&
      cudaMalloc(device, converted.size() * sizeof(Element)) != cudaSuccess) {
    return cudaErrorMemoryAllocation;
  }
  return cudaMemcpy(*device, converted.data(),
                    converted.size() * sizeof(Element), cudaMemcpyHostToDevice);
}

cudaError_t UploadAs(Precision precision, const std::vector<float>& host,
                     void** device) {
  return precision == Precision::kBf16
             ? Upload<cutlass::bfloat16_t>(host, device)
             : Upload<cutlass::float_e4m3_t>(host, device);
}

}  // namespace

bool BlockActive(const BlockPattern& pattern, int i, int j) {
  return abs(i - j) <= pattern.window || i < pattern.global_blocks ||
         j < pattern.global_blocks;
}

double Density(const BlockPattern& pattern, int num_blocks) {
  int64_t active = 0;
  for (int i = 0; i < num_blocks; ++i) {
    for (int j = 0; j < num_blocks; ++j) active += BlockActive(pattern, i, j);
  }
  return static_cast<double>(active) /
         (static_cast<double>(num_blocks) * num_blocks);
}

int WindowForDensity(int num_blocks, int global_blocks, double target_density) {
  int best = 0;
  double best_error = 2.0;
  for (int window = 0; window < num_blocks; ++window) {
    const BlockPattern pattern{1, window, global_blocks};
    const double error =
        std::fabs(Density(pattern, num_blocks) - target_density);
    if (error < best_error) {
      best = window;
      best_error = error;
    }
  }
  return best;
}

void ApplyPattern(const BlockPattern& pattern, int m, int k,
                  std::vector<float>* matrix) {
  const int bs = pattern.block_size;
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < k; ++c) {
      if (!BlockActive(pattern, r / bs, c / bs)) {
        (*matrix)[static_cast<size_t>(r) * k + c] = 0.0f;
      }
    }
  }
}

BlockSparseGemm::BlockSparseGemm(const BlockPattern& pattern, int m, int n,
                                 int k, Precision precision)
    : pattern_(pattern),
      m_(m),
      n_(n),
      k_(k),
      precision_(precision),
      num_block_rows_(m / pattern.block_size),
      band_width_((2 * pattern.window + 1) * pattern.block_size),
      padded_k_(k + 2 * pattern.window * pattern.block_size) {}

BlockSparseGemm::~BlockSparseGemm() {
  cudaFree(band_);
  cudaFree(global_cols_);
  cudaFree(global_rows_);
  cudaFree(padded_b_);
}

cudaError_t BlockSparseGemm::UploadA(const std::vector<float>& a) {
  const int bs = pattern_.block_size;
  const int w = pattern_.window;
  const int global_width = pattern_.global_blocks * bs;
  const auto in_band = [&](int i, int j) { return abs(i - j) <= w; };
  // Band: [num_block_rows][bs][band_width], columns (i - w) * bs + c.
  std::vector<float> band(
      static_cast<size_t>(num_block_rows_) * bs * band_width_, 0.0f);
  for (int i = 0; i < num_block_rows_; ++i) {
    for (int r = 0; r < bs; ++r) {
      for (int c = 0; c < band_width_; ++c) {
        const int col = (i - w) * bs + c;
        if (col < 0 || col >= k_) continue;
        band[(static_cast<size_t>(i) * bs + r) * band_width_ + c] =
            a[static_cast<size_t>(i * bs + r) * k_ + col];
      }
    }
  }
  std::vector<float> cols(static_cast<size_t>(m_) * global_width, 0.0f);
  for (int r = 0; r < m_; ++r) {
    for (int c = 0; c < global_width; ++c) {
      if (!in_band(r / bs, c / bs)) {
        cols[static_cast<size_t>(r) * global_width + c] =
            a[static_cast<size_t>(r) * k_ + c];
      }
    }
  }
  std::vector<float> rows(static_cast<size_t>(global_width) * k_, 0.0f);
  for (int r = 0; r < global_width; ++r) {
    for (int c = global_width; c < k_; ++c) {
      if (!in_band(r / bs, c / bs)) {
        rows[static_cast<size_t>(r) * k_ + c] =
            a[static_cast<size_t>(r) * k_ + c];
      }
    }
  }
  cudaError_t status = UploadAs(precision_, band, &band_);
  if (status != cudaSuccess || global_width == 0) return status;
  status = UploadAs(precision_, cols, &global_cols_);
  if (status != cudaSuccess) return status;
  return UploadAs(precision_, rows, &global_rows_);
}

cudaError_t BlockSparseGemm::UploadB(const std::vector<float>& b) {
  const int pad = pattern_.window * pattern_.block_size;
  std::vector<float> padded(static_cast<size_t>(padded_k_) * n_, 0.0f);
  for (int col = 0; col < n_; ++col) {
    for (int row = 0; row < k_; ++row) {
      padded[static_cast<size_t>(col) * padded_k_ + pad + row] =
          b[static_cast<size_t>(col) * k_ + row];
    }
  }
  return UploadAs(precision_, padded, &padded_b_);
}

cudaError_t BlockSparseGemm::Run(float* c, cudaStream_t stream) const {
  const bool small_blocks = pattern_.block_size == 64;
  if (precision_ == Precision::kBf16) {
    return small_blocks
               ? RunPattern<Bf16Tile64>(pattern_, m_, n_, k_, num_block_rows_,
                                        band_width_, padded_k_, band_,
                                        global_cols_, global_rows_, padded_b_,
                                        c, stream)
               : RunPattern<Bf16Tile128>(pattern_, m_, n_, k_, num_block_rows_,
                                         band_width_, padded_k_, band_,
                                         global_cols_, global_rows_, padded_b_,
                                         c, stream);
  }
  return small_blocks
             ? RunPattern<Fp8Tile64>(pattern_, m_, n_, k_, num_block_rows_,
                                     band_width_, padded_k_, band_,
                                     global_cols_, global_rows_, padded_b_, c,
                                     stream)
             : RunPattern<Fp8Tile128>(pattern_, m_, n_, k_, num_block_rows_,
                                      band_width_, padded_k_, band_,
                                      global_cols_, global_rows_, padded_b_, c,
                                      stream);
}

double BlockSparseGemm::ExecutedFlops() const {
  const double global_width = pattern_.global_blocks * pattern_.block_size;
  const double band = static_cast<double>(m_) * n_ * band_width_;
  const double cols = static_cast<double>(m_) * n_ * global_width;
  const double rows = global_width * n_ * k_;
  return 2.0 * (band + cols + rows);
}

}  // namespace sparse
}  // namespace aether
