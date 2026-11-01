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

#include "src/custom_kernels/cuda/dsmc_collide_tma.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "src/custom_kernels/cuda/philox.h"
#include "src/custom_kernels/cuda/sm90_pipeline.h"

namespace aether {
namespace dsmc {
namespace {

// A stage holds kStageParticles consecutive sorted particles as
// kBoxesPerStage TMA boxes of [3][kBoxWidth] floats (box dims are <= 256).
constexpr int kComponents = 3;
constexpr int kBoxWidth = 256;
constexpr int kBoxesPerStage = 8;
constexpr int kStageParticles = kBoxWidth * kBoxesPerStage;
// Cells are grouped by the window their first particle falls in. A stage
// loaded at a group's first particle then covers every cell of the group
// that has at most kMaxStagedCell particles; larger cells take the
// global-memory fallback kernel.
constexpr int kGroupStride = kStageParticles / 2;
constexpr int kMaxStagedCell = kStageParticles - kGroupStride;
constexpr int kStages = 3;
constexpr int kConsumerWarps = 7;
constexpr int kThreads = 32 * (1 + kConsumerWarps);
constexpr uint32_t kStageBytes = kStageParticles * kComponents * sizeof(float);

struct GroupInfo {
  int first_cell;
  int end_cell;
  int origin;  // Sorted index of the first particle in the stage.
};

struct alignas(128) SharedStorage {
  float velocity[kStages][kBoxesPerStage][kComponents][kBoxWidth];
  uint64_t full[kStages];
  uint64_t empty[kStages];
  GroupInfo group[kStages];
};

// Reads velocity component c of stage-local particle i.
struct StagedVelocity {
  const float* stage;
  __device__ float operator()(int c, int i) const {
    return stage[((i / kBoxWidth) * kComponents + c) * kBoxWidth +
                 i % kBoxWidth];
  }
};

// Reads velocity component c of sorted particle i from global memory.
struct GlobalVelocity {
  const float* base;
  int stride;
  __device__ float operator()(int c, int i) const {
    return base[c * stride + i];
  }
};

// First index in [0, size) whose value is >= `value`.
__device__ int LowerBound(const int* data, int size, int value) {
  int lo = 0;
  int hi = size;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (data[mid] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// One warp collides the candidate pairs of one cell. `read(c, local + k)`
// returns the velocity of the cell's k-th particle; results are written to
// the global velocity tensor at sorted index start + k.
template <typename Reader>
__device__ void CollideCellWarp(const Reader& read, int local, int start,
                                int count, int cell, float* velocity,
                                float* sigma_g_max, int* collisions,
                                const CollideParams& params) {
  const int lane = threadIdx.x % 32;
  const float majorant = sigma_g_max[cell];
  const Philox4 cell_draw = CollisionRandom(cell, kCellDraw, params);
  const int selections = NtcSelections(count, majorant, params.selection_factor,
                                       UniformFloat(cell_draw.v[0]));
  float observed = 0.0f;
  int accepted = 0;
  for (int pair = lane; pair < selections; pair += 32) {
    const int k = 2 * pair;
    float v1[3];
    float v2[3];
    for (int c = 0; c < kComponents; ++c) {
      v1[c] = read(c, local + k);
      v2[c] = read(c, local + k + 1);
    }
    float sigma_g = 0.0f;
    if (CollidePair(v1, v2, majorant, params.gas,
                    CollisionRandom(cell, static_cast<uint32_t>(pair), params),
                    &sigma_g)) {
      for (int c = 0; c < kComponents; ++c) {
        velocity[c * params.padded_particles + start + k] = v1[c];
        velocity[c * params.padded_particles + start + k + 1] = v2[c];
      }
      ++accepted;
    }
    observed = fmaxf(observed, sigma_g);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    observed = fmaxf(observed, __shfl_xor_sync(0xffffffffu, observed, offset));
    accepted += __shfl_xor_sync(0xffffffffu, accepted, offset);
  }
  if (lane == 0) {
    sigma_g_max[cell] = fmaxf(majorant, observed);
    collisions[cell] = accepted;
  }
}

// Warp-specialized persistent kernel: warp 0 streams stages in with TMA,
// warps 1..kConsumerWarps collide the cells of each stage.
__global__ void __launch_bounds__(kThreads)
    CollideTmaKernel(const __grid_constant__ CUtensorMap velocity_map,
                     float* velocity, const int* cell_start, float* sigma_g_max,
                     int* collisions, CollideParams params, int num_groups) {
  extern __shared__ __align__(128) char shared_raw[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_raw);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  if (threadIdx.x == 0) {
    for (int s = 0; s < kStages; ++s) {
      sm90::MbarrierInit(&smem.full[s], 1);
      sm90::MbarrierInit(&smem.empty[s], kConsumerWarps);
    }
    sm90::FenceBarrierInit();
  }
  __syncthreads();

  if (warp == 0) {
    if (lane != 0) return;
    sm90::PrefetchTensorMap(&velocity_map);
    int iteration = 0;
    for (int group = blockIdx.x; group < num_groups;
         group += gridDim.x, ++iteration) {
      const int stage = iteration % kStages;
      const uint32_t phase = (iteration / kStages) & 1;
      sm90::MbarrierWait(&smem.empty[stage], phase ^ 1);
      GroupInfo info;
      info.first_cell =
          LowerBound(cell_start, params.num_cells, group * kGroupStride);
      info.end_cell =
          LowerBound(cell_start, params.num_cells, (group + 1) * kGroupStride);
      info.origin = cell_start[info.first_cell];
      smem.group[stage] = info;
      if (info.first_cell < info.end_cell) {
        sm90::MbarrierArriveExpectTx(&smem.full[stage], kStageBytes);
        for (int box = 0; box < kBoxesPerStage; ++box) {
          sm90::TmaLoad2d(&smem.velocity[stage][box][0][0], &velocity_map,
                          &smem.full[stage], info.origin + box * kBoxWidth, 0);
        }
      } else {
        sm90::MbarrierArrive(&smem.full[stage]);
      }
    }
    return;
  }

  const int consumer = warp - 1;
  int iteration = 0;
  for (int group = blockIdx.x; group < num_groups;
       group += gridDim.x, ++iteration) {
    const int stage = iteration % kStages;
    const uint32_t phase = (iteration / kStages) & 1;
    sm90::MbarrierWait(&smem.full[stage], phase);
    const GroupInfo info = smem.group[stage];
    const StagedVelocity staged{&smem.velocity[stage][0][0][0]};
    for (int cell = info.first_cell + consumer; cell < info.end_cell;
         cell += kConsumerWarps) {
      const int start = cell_start[cell];
      const int count = cell_start[cell + 1] - start;
      if (count > kMaxStagedCell) continue;  // Fallback kernel's job.
      CollideCellWarp(staged, start - info.origin, start, count, cell, velocity,
                      sigma_g_max, collisions, params);
    }
    __syncwarp();
    if (lane == 0) sm90::MbarrierArrive(&smem.empty[stage]);
  }
}

// One warp per cell for the (rare) cells too large to stage.
__global__ void __launch_bounds__(kThreads)
    CollideOversizedKernel(float* velocity, const int* cell_start,
                           float* sigma_g_max, int* collisions,
                           CollideParams params) {
  const int cell = blockIdx.x * (kThreads / 32) + threadIdx.x / 32;
  if (cell >= params.num_cells) return;
  const int start = cell_start[cell];
  const int count = cell_start[cell + 1] - start;
  if (count <= kMaxStagedCell) return;
  const GlobalVelocity global{velocity, params.padded_particles};
  CollideCellWarp(global, start, start, count, cell, velocity, sigma_g_max,
                  collisions, params);
}

}  // namespace

cudaError_t CollideTma(float* velocity, const int* cell_start,
                       float* sigma_g_max, int* collisions,
                       const CollideParams& params, cudaStream_t stream) {
  if (params.padded_particles % 4 != 0 ||
      params.padded_particles < params.num_particles) {
    return cudaErrorInvalidValue;
  }
  // TMA descriptor of the [3][padded_particles] float tensor; boxes are
  // [3][kBoxWidth] and out-of-bounds elements read as zero.
  CUtensorMap map;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(params.padded_particles),
                              kComponents};
  const cuuint64_t strides[1] = {
      static_cast<cuuint64_t>(params.padded_particles) * sizeof(float)};
  const cuuint32_t box[2] = {kBoxWidth, kComponents};
  const cuuint32_t element_strides[2] = {1, 1};
  const CUresult encoded = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, velocity, dims, strides, box,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (encoded != CUDA_SUCCESS) return cudaErrorInvalidValue;

  cudaError_t status =
      cudaMemsetAsync(collisions, 0, sizeof(int) * params.num_cells, stream);
  if (status != cudaSuccess) return status;

  const int smem_bytes = static_cast<int>(sizeof(SharedStorage));
  status = cudaFuncSetAttribute(CollideTmaKernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_bytes);
  if (status != cudaSuccess) return status;
  int device = 0;
  int num_sms = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);

  const int num_groups =
      (params.num_particles + kGroupStride - 1) / kGroupStride;
  if (num_groups > 0) {
    const int grid = num_groups < 2 * num_sms ? num_groups : 2 * num_sms;
    CollideTmaKernel<<<grid, kThreads, smem_bytes, stream>>>(
        map, velocity, cell_start, sigma_g_max, collisions, params, num_groups);
  }
  const int warps_per_block = kThreads / 32;
  const int fallback_blocks =
      (params.num_cells + warps_per_block - 1) / warps_per_block;
  if (fallback_blocks > 0) {
    CollideOversizedKernel<<<fallback_blocks, kThreads, 0, stream>>>(
        velocity, cell_start, sigma_g_max, collisions, params);
  }
  return cudaGetLastError();
}

}  // namespace dsmc
}  // namespace aether

extern "C" int AetherDsmcCollideTma(float* velocity, const int* cell_start,
                                    float* sigma_g_max, int* collisions,
                                    int num_particles, int padded_particles,
                                    int num_cells, float selection_factor,
                                    float vhs_coefficient,
                                    float vhs_speed_exponent, uint64_t seed,
                                    uint64_t step, void* stream) {
  aether::dsmc::CollideParams params;
  params.num_particles = num_particles;
  params.padded_particles = padded_particles;
  params.num_cells = num_cells;
  params.selection_factor = selection_factor;
  params.gas = aether::dsmc::VhsGas{vhs_coefficient, vhs_speed_exponent};
  params.seed = seed;
  params.step = step;
  return static_cast<int>(
      aether::dsmc::CollideTma(velocity, cell_start, sigma_g_max, collisions,
                               params, static_cast<cudaStream_t>(stream)));
}
