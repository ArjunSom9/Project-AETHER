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

// DSMC No-Time-Counter collisions on cell-sorted particles, staged through
// shared memory with Hopper TMA bulk-tensor copies (report Part III, 3.3).
//
// Input is the output order of the fused Triton radix sort: particles sorted
// by collision cell and randomly shuffled inside each cell, so the candidate
// pairs of a cell are (start + 2j, start + 2j + 1). Velocities are one
// row-major float tensor of shape [3][padded_particles] (rows vx, vy, vz).
//
// Example:
//   aether::dsmc::CollideParams params = ...;
//   cudaError_t status = aether::dsmc::CollideTma(velocity, cell_start,
//                                                 sigma_g_max, collisions,
//                                                 params, stream);

#ifndef PROJECT_AETHER_SRC_CUSTOM_KERNELS_CUDA_DSMC_COLLIDE_TMA_H_
#define PROJECT_AETHER_SRC_CUSTOM_KERNELS_CUDA_DSMC_COLLIDE_TMA_H_

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "src/custom_kernels/cuda/philox.h"

namespace aether {
namespace dsmc {

// Variable-hard-sphere cross section in the form sigma_T * g = coefficient *
// g^speed_exponent, i.e. coefficient = pi d^2 (2 k T_ref / m_r)^(omega - 1/2)
// / Gamma(5/2 - omega) and speed_exponent = 2 - 2 omega (Bird 1994).
struct VhsGas {
  float coefficient;
  float speed_exponent;
};

struct CollideParams {
  int num_particles;
  int padded_particles;  // Row stride of the velocity tensor; multiple of 4.
  int num_cells;
  float selection_factor;  // F_N * dt / V_cell.
  VhsGas gas;
  uint64_t seed;
  uint64_t step;
};

// Number of NTC candidate pairs of a cell: floor(1/2 N (N - 1) F_N
// (sigma g)_max dt / V + U), capped by the N / 2 disjoint pairs.
AETHER_HOST_DEVICE inline int NtcSelections(int count, float sigma_g_max,
                                            float selection_factor,
                                            float uniform) {
  const float expected = 0.5f * static_cast<float>(count) *
                         static_cast<float>(count - 1) * selection_factor *
                         sigma_g_max;
  const int selections = static_cast<int>(floorf(expected + uniform));
  return selections < count / 2 ? selections : count / 2;
}

// Random words of a cell's selection draw and of candidate pair `pair`.
AETHER_HOST_DEVICE inline Philox4 CollisionRandom(int cell, uint32_t pair,
                                                  const CollideParams& p) {
  const Philox4 counter = {{static_cast<uint32_t>(cell), pair,
                            static_cast<uint32_t>(p.step),
                            static_cast<uint32_t>(p.step >> 32)}};
  return Philox4x32(counter, static_cast<uint32_t>(p.seed),
                    static_cast<uint32_t>(p.seed >> 32));
}

constexpr uint32_t kCellDraw = 0xFFFFFFFFu;

// Tests one candidate pair and, if accepted, scatters it isotropically in
// the center-of-mass frame (equal masses), conserving momentum and energy.
// Writes sigma_T * g of the pair to `sigma_g`; returns whether it collided.
AETHER_HOST_DEVICE inline bool CollidePair(float* v1, float* v2,
                                           float sigma_g_max, const VhsGas& gas,
                                           const Philox4& random,
                                           float* sigma_g) {
  const float gx = v1[0] - v2[0];
  const float gy = v1[1] - v2[1];
  const float gz = v1[2] - v2[2];
  const float g = sqrtf(gx * gx + gy * gy + gz * gz);
  *sigma_g = g > 0.0f ? gas.coefficient * powf(g, gas.speed_exponent) : 0.0f;
  if (UniformFloat(random.v[0]) * sigma_g_max >= *sigma_g) return false;
  const float cos_chi = 2.0f * UniformFloat(random.v[1]) - 1.0f;
  const float sin_chi = sqrtf(fmaxf(1.0f - cos_chi * cos_chi, 0.0f));
  const float azimuth = 6.28318530718f * UniformFloat(random.v[2]);
  const float g_post[3] = {g * sin_chi * cosf(azimuth),
                           g * sin_chi * sinf(azimuth), g * cos_chi};
  for (int c = 0; c < 3; ++c) {
    const float center = 0.5f * (v1[c] + v2[c]);
    v1[c] = center + 0.5f * g_post[c];
    v2[c] = center - 0.5f * g_post[c];
  }
  return true;
}

// Collides every cell. `velocity` ([3][padded_particles], device) is updated
// in place; `sigma_g_max` ([num_cells]) is raised to the largest sigma_T * g
// observed; `collisions` ([num_cells]) receives accepted pairs per cell.
// `cell_start` ([num_cells + 1]) is the exclusive prefix sum of the cell
// counts. Requires an sm_90 device.
cudaError_t CollideTma(float* velocity, const int* cell_start,
                       float* sigma_g_max, int* collisions,
                       const CollideParams& params, cudaStream_t stream);

}  // namespace dsmc
}  // namespace aether

// C ABI for the Python ctypes binding (`tma_collide.py`). Returns a
// cudaError_t value.
extern "C" int AetherDsmcCollideTma(float* velocity, const int* cell_start,
                                    float* sigma_g_max, int* collisions,
                                    int num_particles, int padded_particles,
                                    int num_cells, float selection_factor,
                                    float vhs_coefficient,
                                    float vhs_speed_exponent, uint64_t seed,
                                    uint64_t step, void* stream);

#endif  // PROJECT_AETHER_SRC_CUSTOM_KERNELS_CUDA_DSMC_COLLIDE_TMA_H_
