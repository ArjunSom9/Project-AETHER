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

// Philox4x32-10 counter-based random numbers (J. K. Salmon et al., "Parallel
// Random Numbers: As Easy as 1, 2, 3", SC '11), usable on host and device.
//
// The output is a pure function of (counter, key): any thread draws the
// random numbers of any (particle, step) directly, without stored generator
// state, and the host can reproduce the device stream exactly.

#ifndef PROJECT_AETHER_SRC_CUSTOM_KERNELS_CUDA_PHILOX_H_
#define PROJECT_AETHER_SRC_CUSTOM_KERNELS_CUDA_PHILOX_H_

#include <stdint.h>

#if defined(__CUDACC__)
#define AETHER_HOST_DEVICE __host__ __device__
#else
#define AETHER_HOST_DEVICE
#endif

namespace aether {

struct Philox4 {
  uint32_t v[4];
};

namespace internal_philox {

constexpr uint32_t kMultiplier0 = 0xD2511F53u;
constexpr uint32_t kMultiplier1 = 0xCD9E8D57u;
constexpr uint32_t kWeyl0 = 0x9E3779B9u;
constexpr uint32_t kWeyl1 = 0xBB67AE85u;

AETHER_HOST_DEVICE inline void MulHiLo(uint32_t a, uint32_t b, uint32_t* hi,
                                       uint32_t* lo) {
  const uint64_t product = static_cast<uint64_t>(a) * b;
  *hi = static_cast<uint32_t>(product >> 32);
  *lo = static_cast<uint32_t>(product);
}

}  // namespace internal_philox

// Philox4x32 with 10 rounds: returns four 32-bit random words.
AETHER_HOST_DEVICE inline Philox4 Philox4x32(Philox4 counter, uint32_t key0,
                                             uint32_t key1) {
  using internal_philox::kMultiplier0;
  using internal_philox::kMultiplier1;
  for (int round = 0; round < 10; ++round) {
    if (round > 0) {
      key0 += internal_philox::kWeyl0;
      key1 += internal_philox::kWeyl1;
    }
    uint32_t hi0, lo0, hi1, lo1;
    internal_philox::MulHiLo(kMultiplier0, counter.v[0], &hi0, &lo0);
    internal_philox::MulHiLo(kMultiplier1, counter.v[2], &hi1, &lo1);
    counter = Philox4{
        {hi1 ^ counter.v[1] ^ key0, lo1, hi0 ^ counter.v[3] ^ key1, lo0}};
  }
  return counter;
}

// Maps a 32-bit word to a float uniform in [0, 1) with 24 random bits.
AETHER_HOST_DEVICE inline float UniformFloat(uint32_t bits) {
  return static_cast<float>(bits >> 8) * (1.0f / 16777216.0f);
}

}  // namespace aether

#endif  // PROJECT_AETHER_SRC_CUSTOM_KERNELS_CUDA_PHILOX_H_
