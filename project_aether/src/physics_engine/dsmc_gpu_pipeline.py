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
"""GPU DSMC timestep built from the custom kernels (report Parts II-III).

One step on cell-sorted Structure-of-Arrays particles:

  1. move: `fused_particle_push` (Triton, specular walls);
  2. bin + collision-partner search: `fused_radix_sort` (Triton, Philox
     in-cell shuffle, per-cell counts);
  3. gather the particle payload into sorted order (one irregular gather;
     Hopper TMA cannot gather);
  4. collide: the TMA-pipelined NTC kernel (`dsmc_collide_tma.cu`) through
     `tma_collide`, which streams the now-contiguous velocity tiles through
     shared memory.

The particles stay in sorted order between steps, which keeps the next
sort's scatter and the PIC field gathers cache friendly. Particle state is
held as PyTorch CUDA tensors because Triton 2.0 kernels launch on them.

Typical usage example:

  state = GpuDsmcState.from_numpy(positions, velocities, grid, sigma_g_max)
  stats = gpu_dsmc_step(state, library, grid, params, dt, step)
"""

import dataclasses
from typing import NamedTuple

import numpy as np
import torch

from src.custom_kernels.cuda import tma_collide
from src.custom_kernels.triton import fused_particle_push
from src.custom_kernels.triton import fused_radix_sort
from src.custom_kernels.triton import radix_sort_reference


@dataclasses.dataclass
class GpuDsmcState:
    """Particle state on the GPU.

    Attributes:
      positions: (x, y, z) float32 CUDA tensors (N,).
      velocity: float32 CUDA tensor (3, padded) of vx, vy, vz.
      sigma_g_max: Per-cell NTC majorant, float32 (C,).
      num_particles: N.
    """
    positions: list[torch.Tensor]
    velocity: torch.Tensor
    sigma_g_max: torch.Tensor
    num_particles: int

    @classmethod
    def from_numpy(cls, positions: np.ndarray, velocities: np.ndarray,
                   sigma_g_max: np.ndarray) -> 'GpuDsmcState':
        """Uploads (N, 3) positions and velocities and (C,) majorants."""
        n = positions.shape[0]
        padded = -(-n // 4) * 4
        velocity = torch.zeros((3, padded), dtype=torch.float32, device='cuda')
        velocity[:, :n] = torch.as_tensor(velocities.T, dtype=torch.float32)
        pos = [torch.as_tensor(np.ascontiguousarray(positions[:, i]),
                               dtype=torch.float32, device='cuda')
               for i in range(3)]
        return cls(pos, velocity,
                   torch.as_tensor(sigma_g_max, dtype=torch.float32,
                                   device='cuda'), n)


class GpuStepStats(NamedTuple):
    cell_counts: torch.Tensor
    collisions: torch.Tensor


def gpu_dsmc_step(state: GpuDsmcState, library,
                  grid: radix_sort_reference.CellGrid,
                  params: tma_collide.CollisionParams,
                  dt: float) -> GpuStepStats:
    """Advances `state` by one DSMC timestep in place.

    Args:
      state: Particle state; left in cell-sorted order.
      library: The CUDA library from `tma_collide.load_library`.
      grid: Collision-cell grid (also the reflecting box).
      params: Collision parameters; `params.step` keys the Philox streams.
      dt: Time step [s].

    Returns:
      Per-cell particle counts and accepted collisions.
    """
    n = state.num_particles
    vel = [state.velocity[c, :n] for c in range(3)]
    fused_particle_push.fused_particle_push(state.positions, vel, dt,
                                            grid.lo, grid.hi)
    order = fused_radix_sort.sort_particles_by_cell(
        *state.positions, grid, seed=params.seed ^ (params.step << 20))
    index = order.particle_index.long()
    state.positions = [p.index_select(0, index) for p in state.positions]
    state.velocity[:, :n] = state.velocity[:, :n].index_select(1, index)
    collisions = torch.empty_like(order.cell_counts)
    tma_collide.collide(library, state.velocity, order.cell_start,
                        state.sigma_g_max, collisions, n, params)
    return GpuStepStats(order.cell_counts, collisions)
