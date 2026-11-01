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
"""Fused particle push with specular wall reflection (Triton 2.0).

The DSMC/PIC "move" stage advances x += v dt and reflects particles at the
domain walls. Fusing the kinematic update and the six wall tests into one
kernel reads each Structure-of-Arrays component once and writes it once,
instead of one HBM round trip per elementwise operation. With the SoA layout,
every load and store is fully coalesced.

Typical usage example:

  fused_particle_push(pos, vel, dt=1e-7, lo=(0., 0., 0.), hi=(1., 1., 1.))
"""

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _reflect(c, v, lo, hi):
    """Specular reflection of coordinate c (velocity v) into [lo, hi]."""
    below = c < lo
    above = c > hi
    c = tl.where(below, 2.0 * lo - c, c)
    c = tl.where(above, 2.0 * hi - c, c)
    v = tl.where(below | above, -v, v)
    return c, v


@triton.jit
def _push_kernel(
    x_ptr, y_ptr, z_ptr, vx_ptr, vy_ptr, vz_ptr,
    n, dt,
    x_lo, y_lo, z_lo, x_hi, y_hi, z_hi,
    BLOCK: tl.constexpr,
):
    # Runtime scalars are deliberately un-annotated: Triton 2.0 turns every
    # annotated argument into a compile-time constant.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = tl.load(z_ptr + offs, mask=mask)
    vx = tl.load(vx_ptr + offs, mask=mask)
    vy = tl.load(vy_ptr + offs, mask=mask)
    vz = tl.load(vz_ptr + offs, mask=mask)
    x, vx = _reflect(x + vx * dt, vx, x_lo, x_hi)
    y, vy = _reflect(y + vy * dt, vy, y_lo, y_hi)
    z, vz = _reflect(z + vz * dt, vz, z_lo, z_hi)
    tl.store(x_ptr + offs, x, mask=mask)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(z_ptr + offs, z, mask=mask)
    tl.store(vx_ptr + offs, vx, mask=mask)
    tl.store(vy_ptr + offs, vy, mask=mask)
    tl.store(vz_ptr + offs, vz, mask=mask)


def fused_particle_push(positions: Sequence[torch.Tensor],
                        velocities: Sequence[torch.Tensor], dt: float,
                        lo: Sequence[float], hi: Sequence[float]) -> None:
    """Advances particles in place and reflects them at the box walls.

    Args:
      positions: (x, y, z) float32 CUDA tensors of shape (N,), updated in
        place.
      velocities: (vx, vy, vz) float32 CUDA tensors of shape (N,), updated
        in place.
      dt: Time step.
      lo: Lower box corner (x, y, z).
      hi: Upper box corner (x, y, z).
    """
    n = positions[0].numel()
    grid = (triton.cdiv(n, _BLOCK),)
    _push_kernel[grid](*positions, *velocities, n, dt, *lo, *hi,
                       BLOCK=_BLOCK, num_warps=4)


def reference_push(positions: Sequence[torch.Tensor],
                   velocities: Sequence[torch.Tensor], dt: float,
                   lo: Sequence[float], hi: Sequence[float]
                   ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Unfused PyTorch reference of `fused_particle_push` (out of place)."""
    new_pos, new_vel = [], []
    for c, v, l, h in zip(positions, velocities, lo, hi):
        c = c + v * dt
        below, above = c < l, c > h
        c = torch.where(below, 2.0 * l - c, c)
        c = torch.where(above, 2.0 * h - c, c)
        new_pos.append(c)
        new_vel.append(torch.where(below | above, -v, v))
    return new_pos, new_vel
