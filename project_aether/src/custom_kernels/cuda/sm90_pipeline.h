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

// Hopper (sm_90) asynchronous-copy primitives: transaction barriers
// (mbarrier) and Tensor Memory Accelerator (TMA) bulk-tensor copies.
//
// CUDA 12.2 exposes TMA only through PTX, so these wrappers issue the same
// inline PTX as CUTLASS 3.x (`cutlass/arch/barrier.h`, ClusterBarrier and
// ClusterTransactionBarrier) and CuTe (`cute/arch/copy_sm90_tma.hpp`,
// SM90_TMA_LOAD_2D), the reference idioms report Part III, 3.3 follows.
//
// Producer/consumer protocol of a multi-stage pipeline:
//   full[s]:  count 1. The producer arms it with ArriveExpectTx(bytes) and
//             issues TMA copies that complete the transaction bytes.
//   empty[s]: count = consumer warps. Each consumer warp arrives once it no
//             longer reads stage s.
// For iteration i, stage s = i % kStages and phase p = (i / kStages) & 1;
// consumers wait on full[s] with parity p, the producer waits on empty[s]
// with parity p ^ 1 (which completes immediately for the first kStages
// iterations, as in CUTLASS's PipelineTmaAsync).

#ifndef PROJECT_AETHER_SRC_CUSTOM_KERNELS_CUDA_SM90_PIPELINE_H_
#define PROJECT_AETHER_SRC_CUSTOM_KERNELS_CUDA_SM90_PIPELINE_H_

#include <cuda.h>
#include <stdint.h>

namespace aether {
namespace sm90 {

// Shared-memory address of a generic pointer, as PTX expects it.
__device__ inline uint32_t SmemAddress(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Initializes an mbarrier that completes a phase after `arrive_count`
// arrivals (plus any expected transaction bytes).
__device__ inline void MbarrierInit(uint64_t* barrier, uint32_t arrive_count) {
  asm volatile("mbarrier.init.shared.b64 [%1], %0;\n"
               :
               : "r"(arrive_count), "r"(SmemAddress(barrier)));
}

// Makes barrier initialization visible to the async proxy (TMA).
__device__ inline void FenceBarrierInit() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

// Arrives once and expects `bytes` more transaction bytes this phase.
__device__ inline void MbarrierArriveExpectTx(uint64_t* barrier,
                                              uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%1], %0;\n"
               :
               : "r"(bytes), "r"(SmemAddress(barrier))
               : "memory");
}

// Plain arrival (release semantics).
__device__ inline void MbarrierArrive(uint64_t* barrier) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n"
               :
               : "r"(SmemAddress(barrier))
               : "memory");
}

// Blocks until the phase with parity `phase` has completed (acquire).
__device__ inline void MbarrierWait(uint64_t* barrier, uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
      "@P1 bra.uni DONE;\n"
      "bra.uni LAB_WAIT;\n"
      "DONE:\n"
      "}\n"
      :
      : "r"(SmemAddress(barrier)), "r"(phase)
      : "memory");
}

// Prefetches a tensor-map descriptor into the descriptor cache.
__device__ inline void PrefetchTensorMap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];\n"
               :
               : "l"(reinterpret_cast<uint64_t>(map))
               : "memory");
}

// Copies the box at element coordinates (x, y) of a 2D tensor map into
// shared memory; completion is signalled on `barrier` as transaction bytes.
// `dst` must be 128-byte aligned.
__device__ inline void TmaLoad2d(void* dst, const CUtensorMap* map,
                                 uint64_t* barrier, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx"
      "::bytes [%0], [%1, {%3, %4}], [%2];\n"
      :
      : "r"(SmemAddress(dst)), "l"(reinterpret_cast<uint64_t>(map)),
        "r"(SmemAddress(barrier)), "r"(x), "r"(y)
      : "memory");
}

}  // namespace sm90
}  // namespace aether

#endif  // PROJECT_AETHER_SRC_CUSTOM_KERNELS_CUDA_SM90_PIPELINE_H_
