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

// Block-sparse GEMM C = A * B for the surrogate's 75%-sparse (window +
// global) block pattern, built from CUTLASS 3.x Hopper templates (report
// Part V, 5.1).
//
// The pattern is the one of `block_sparse_attention.block_sparse_layout`:
// block (i, j) is active when |i - j| <= window or i < global or j < global.
// Its active blocks are split into three disjoint dense problems, each a
// launch of the same CUTLASS kernel template (TMA loads, WGMMA math,
// warp-specialized mainloop):
//
//   1. band:        a strided-batched GEMM over block rows; batch i
//                   multiplies the packed band of block row i,
//                   [bs x (2w+1)bs], with the matching rows of B,
//   2. global cols: A[:, 0:g*bs] (band blocks zeroed) x B[0:g*bs, :],
//   3. global rows: A[0:g*bs, :] (band and global-col blocks zeroed) x B,
//
// accumulating into C (beta = 0, then 1). Every active block is computed
// exactly once, so work scales with the ~25% density instead of the dense
// problem. The BF16 path matches report Table 4; the FP8 (E4M3) path uses
// CUTLASS 3.2's FP8 warp-specialized fast-accumulation mainloop.
//
// Example:
//   aether::sparse::BlockSparseGemm gemm(pattern, m, n, k, Precision::kBf16);
//   gemm.UploadA(dense_masked_a);    // Row-major M x K.
//   gemm.UploadB(b_column_major);    // Column-major K x N.
//   gemm.Run(device_c, stream);      // Row-major M x N, float.

#ifndef PROJECT_AETHER_SRC_CUSTOM_KERNELS_CUTLASS_BLOCK_SPARSE_GEMM_H_
#define PROJECT_AETHER_SRC_CUSTOM_KERNELS_CUTLASS_BLOCK_SPARSE_GEMM_H_

#include <cuda_runtime.h>
#include <stdint.h>

#include <vector>

namespace aether {
namespace sparse {

// Window + global block pattern (a compile-time constant of the model).
struct BlockPattern {
  int block_size;
  int window;
  int global_blocks;
};

bool BlockActive(const BlockPattern& pattern, int i, int j);

// Fraction of active blocks for a num_blocks x num_blocks matrix.
double Density(const BlockPattern& pattern, int num_blocks);

// Band half-width whose density is closest to `target_density`.
int WindowForDensity(int num_blocks, int global_blocks, double target_density);

// Zeroes every inactive block of a row-major m x k matrix in place.
void ApplyPattern(const BlockPattern& pattern, int m, int k,
                  std::vector<float>* matrix);

enum class Precision { kBf16, kFp8E4m3 };

// Owns the packed sparse A and the zero-padded B on the device.
// Not copyable or movable.
class BlockSparseGemm {
 public:
  // Problem sizes must be multiples of the block size; block_size must be 64
  // or 128.
  BlockSparseGemm(const BlockPattern& pattern, int m, int n, int k,
                  Precision precision);
  BlockSparseGemm(const BlockSparseGemm&) = delete;
  BlockSparseGemm& operator=(const BlockSparseGemm&) = delete;
  ~BlockSparseGemm();

  // Packs a row-major m x k matrix (inactive blocks are ignored) into the
  // band / global-column / global-row operands and uploads them.
  cudaError_t UploadA(const std::vector<float>& a);

  // Uploads a column-major k x n matrix into the zero-padded B operand.
  cudaError_t UploadB(const std::vector<float>& b);

  // Computes C = A * B into a row-major m x n float device buffer.
  // Returns cudaSuccess, or the first failing CUDA / CUTLASS step as
  // cudaErrorLaunchFailure.
  cudaError_t Run(float* c, cudaStream_t stream) const;

  // Multiply-adds actually executed (including band padding), for reporting.
  double ExecutedFlops() const;

 private:
  BlockPattern pattern_;
  int m_;
  int n_;
  int k_;
  Precision precision_;
  int num_block_rows_;
  int band_width_;  // (2 * window + 1) * block_size.
  int padded_k_;    // k + 2 * window * block_size.
  void* band_ = nullptr;
  void* global_cols_ = nullptr;
  void* global_rows_ = nullptr;
  void* padded_b_ = nullptr;
};

}  // namespace sparse
}  // namespace aether

#endif  // PROJECT_AETHER_SRC_CUSTOM_KERNELS_CUTLASS_BLOCK_SPARSE_GEMM_H_
