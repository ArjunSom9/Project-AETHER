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

// GPU test of the TMA collision kernel against a CPU replay of the same
// algorithm (same Philox streams), plus conservation checks. Requires an
// sm_90 GPU. Exit status 0 means every check passed.

#include "src/custom_kernels/cuda/dsmc_collide_tma.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <random>
#include <vector>

#include "src/custom_kernels/cuda/philox.h"

namespace aether {
namespace dsmc {
namespace {

void Check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    exit(1);
  }
}

// CPU replay of CollideCellWarp for every cell.
void CollideReference(std::vector<float>& velocity,
                      const std::vector<int>& cell_start,
                      std::vector<float>& sigma_g_max,
                      std::vector<int>& collisions,
                      const CollideParams& params) {
  const int stride = params.padded_particles;
  for (int cell = 0; cell < params.num_cells; ++cell) {
    const int start = cell_start[cell];
    const int count = cell_start[cell + 1] - start;
    const float majorant = sigma_g_max[cell];
    const int selections = NtcSelections(
        count, majorant, params.selection_factor,
        UniformFloat(CollisionRandom(cell, kCellDraw, params).v[0]));
    float observed = 0.0f;
    int accepted = 0;
    for (int pair = 0; pair < selections; ++pair) {
      const int i = start + 2 * pair;
      float v1[3];
      float v2[3];
      for (int c = 0; c < 3; ++c) {
        v1[c] = velocity[c * stride + i];
        v2[c] = velocity[c * stride + i + 1];
      }
      float sigma_g = 0.0f;
      if (CollidePair(
              v1, v2, majorant, params.gas,
              CollisionRandom(cell, static_cast<uint32_t>(pair), params),
              &sigma_g)) {
        for (int c = 0; c < 3; ++c) {
          velocity[c * stride + i] = v1[c];
          velocity[c * stride + i + 1] = v2[c];
        }
        ++accepted;
      }
      observed = std::max(observed, sigma_g);
    }
    sigma_g_max[cell] = std::max(majorant, observed);
    collisions[cell] = accepted;
  }
}

int Run() {
  std::mt19937 rng(1234);
  const int num_cells = 4096;
  std::poisson_distribution<int> occupancy(300.0);
  std::vector<int> counts(num_cells);
  for (int& c : counts) c = occupancy(rng);
  counts[17] = 3000;  // Exercises the oversized-cell fallback.
  std::vector<int> cell_start(num_cells + 1, 0);
  for (int c = 0; c < num_cells; ++c) {
    cell_start[c + 1] = cell_start[c] + counts[c];
  }
  const int n = cell_start[num_cells];
  const int padded = (n + 3) / 4 * 4;

  std::normal_distribution<float> thermal(0.0f, 800.0f);
  std::vector<float> velocity(3 * static_cast<size_t>(padded), 0.0f);
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < n; ++i) velocity[c * padded + i] = thermal(rng);
  }

  CollideParams params;
  params.num_particles = n;
  params.padded_particles = padded;
  params.num_cells = num_cells;
  params.selection_factor = 2.0e-17f;
  // Hard-sphere-like air: sigma g = 1.7e-18 * g^0.46 (omega = 0.77).
  params.gas = VhsGas{1.7e-18f, 0.46f};
  params.seed = 0x5eedf00dULL;
  params.step = 7;
  std::vector<float> sigma_g_max(num_cells, 3.0e-16f);

  float* d_velocity = nullptr;
  int* d_cell_start = nullptr;
  float* d_sigma = nullptr;
  int* d_collisions = nullptr;
  Check(cudaMalloc(&d_velocity, velocity.size() * sizeof(float)), "malloc");
  Check(cudaMalloc(&d_cell_start, cell_start.size() * sizeof(int)), "malloc");
  Check(cudaMalloc(&d_sigma, num_cells * sizeof(float)), "malloc");
  Check(cudaMalloc(&d_collisions, num_cells * sizeof(int)), "malloc");
  Check(cudaMemcpy(d_velocity, velocity.data(), velocity.size() * sizeof(float),
                   cudaMemcpyHostToDevice),
        "copy");
  Check(cudaMemcpy(d_cell_start, cell_start.data(),
                   cell_start.size() * sizeof(int), cudaMemcpyHostToDevice),
        "copy");
  Check(cudaMemcpy(d_sigma, sigma_g_max.data(), num_cells * sizeof(float),
                   cudaMemcpyHostToDevice),
        "copy");

  cudaEvent_t begin;
  cudaEvent_t end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord(begin);
  Check(CollideTma(d_velocity, d_cell_start, d_sigma, d_collisions, params,
                   nullptr),
        "CollideTma");
  cudaEventRecord(end);
  Check(cudaDeviceSynchronize(), "kernel");
  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, begin, end);

  std::vector<float> gpu_velocity(velocity.size());
  std::vector<int> gpu_collisions(num_cells);
  std::vector<float> gpu_sigma(num_cells);
  Check(cudaMemcpy(gpu_velocity.data(), d_velocity,
                   velocity.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "copy back");
  Check(cudaMemcpy(gpu_collisions.data(), d_collisions, num_cells * sizeof(int),
                   cudaMemcpyDeviceToHost),
        "copy back");
  Check(cudaMemcpy(gpu_sigma.data(), d_sigma, num_cells * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "copy back");

  std::vector<float> ref_velocity = velocity;
  std::vector<float> ref_sigma = sigma_g_max;
  std::vector<int> ref_collisions(num_cells);
  CollideReference(ref_velocity, cell_start, ref_sigma, ref_collisions, params);

  // 1. Same accept decisions (host and device math may differ by a few ulps,
  //    which can flip a borderline acceptance in rare cells).
  int matching_cells = 0;
  long total = 0;
  for (int c = 0; c < num_cells; ++c) {
    matching_cells += gpu_collisions[c] == ref_collisions[c];
    total += gpu_collisions[c];
  }
  // 2. Momentum and energy conserved in every cell.
  double worst_momentum = 0.0;
  double worst_energy = 0.0;
  for (int cell = 0; cell < num_cells; ++cell) {
    double p0[3] = {0, 0, 0};
    double p1[3] = {0, 0, 0};
    double e0 = 0.0;
    double e1 = 0.0;
    for (int i = cell_start[cell]; i < cell_start[cell + 1]; ++i) {
      for (int c = 0; c < 3; ++c) {
        const double before = velocity[c * padded + i];
        const double after = gpu_velocity[c * padded + i];
        p0[c] += before;
        p1[c] += after;
        e0 += before * before;
        e1 += after * after;
      }
    }
    if (e0 == 0.0) continue;
    const double scale = sqrt(e0);
    for (int c = 0; c < 3; ++c) {
      worst_momentum = std::max(worst_momentum, fabs(p1[c] - p0[c]) / scale);
    }
    worst_energy = std::max(worst_energy, fabs(e1 - e0) / e0);
  }

  const double match_fraction = static_cast<double>(matching_cells) / num_cells;
  printf("particles %d, cells %d, accepted %ld, %.3f ms (%.1f Mpairs/s)\n", n,
         num_cells, total, elapsed_ms, total / (elapsed_ms * 1e3));
  printf("cells matching CPU replay: %.4f\n", match_fraction);
  printf("worst relative momentum error %.3e, energy error %.3e\n",
         worst_momentum, worst_energy);

  cudaFree(d_velocity);
  cudaFree(d_cell_start);
  cudaFree(d_sigma);
  cudaFree(d_collisions);
  const bool ok = match_fraction > 0.999 && worst_momentum < 1e-5 &&
                  worst_energy < 1e-5 && total > 0;
  printf(ok ? "PASSED\n" : "FAILED\n");
  return ok ? 0 : 1;
}

}  // namespace
}  // namespace dsmc
}  // namespace aether

int main() { return aether::dsmc::Run(); }
