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
"""Particle-in-cell push for the re-entry plasma sheath (Triton 2.0).

Electrons and ions in the sheath around a re-entry vehicle move under the
local electric and magnetic fields. One fused kernel per step

  1. gathers E and B at every particle with bilinear weights (shared helpers
     from `pic_tiled_interpolation`, reusing the corner indices for all six
     field components),
  2. advances the velocity with the Boris scheme (J. P. Boris, 1970), which
     splits the Lorentz force into two electric half-kicks around an exact
     magnetic rotation and therefore conserves kinetic energy in a pure
     magnetic field,
  3. advances the in-plane position.

Fields are node-centered on a row-major (ny, nx) grid; particles should be
cell-sorted (e.g. by the fused radix sort) so the gathers stay cache local.

Typical usage example:

  boris_push(pos, vel, fields, cell_size=(dx, dy), charge_mass=-1.76e11,
             dt=1e-12)
"""

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from src.custom_kernels.triton import pic_tiled_interpolation as interp


@triton.jit
def _boris_kernel(
    x_ptr, y_ptr, vx_ptr, vy_ptr, vz_ptr,
    ex_ptr, ey_ptr, ez_ptr, bx_ptr, by_ptr, bz_ptr,
    n, nx, ny, inv_dx, inv_dy, half_qm_dt, dt,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    base, w00, w10, w01, w11 = interp.bilinear_weights(x, y, inv_dx, inv_dy,
                                                       nx, ny)
    ex = interp.bilinear_gather(ex_ptr, base, w00, w10, w01, w11, nx, mask)
    ey = interp.bilinear_gather(ey_ptr, base, w00, w10, w01, w11, nx, mask)
    ez = interp.bilinear_gather(ez_ptr, base, w00, w10, w01, w11, nx, mask)
    bx = interp.bilinear_gather(bx_ptr, base, w00, w10, w01, w11, nx, mask)
    by = interp.bilinear_gather(by_ptr, base, w00, w10, w01, w11, nx, mask)
    bz = interp.bilinear_gather(bz_ptr, base, w00, w10, w01, w11, nx, mask)

    # Boris: half electric kick, magnetic rotation, half electric kick.
    vx = tl.load(vx_ptr + offs, mask=mask, other=0.0) + half_qm_dt * ex
    vy = tl.load(vy_ptr + offs, mask=mask, other=0.0) + half_qm_dt * ey
    vz = tl.load(vz_ptr + offs, mask=mask, other=0.0) + half_qm_dt * ez
    tx, ty, tz = half_qm_dt * bx, half_qm_dt * by, half_qm_dt * bz
    s_factor = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz)
    px = vx + (vy * tz - vz * ty)
    py = vy + (vz * tx - vx * tz)
    pz = vz + (vx * ty - vy * tx)
    vx = vx + s_factor * (py * tz - pz * ty) + half_qm_dt * ex
    vy = vy + s_factor * (pz * tx - px * tz) + half_qm_dt * ey
    vz = vz + s_factor * (px * ty - py * tx) + half_qm_dt * ez

    tl.store(vx_ptr + offs, vx, mask=mask)
    tl.store(vy_ptr + offs, vy, mask=mask)
    tl.store(vz_ptr + offs, vz, mask=mask)
    tl.store(x_ptr + offs, x + vx * dt, mask=mask)
    tl.store(y_ptr + offs, y + vy * dt, mask=mask)


def boris_push(positions: Sequence[torch.Tensor],
               velocities: Sequence[torch.Tensor],
               fields: Sequence[torch.Tensor], cell_size: Sequence[float],
               charge_mass: float, dt: float) -> None:
    """Advances charged particles one step in place.

    Args:
      positions: (x, y) float32 CUDA tensors (N,), updated in place.
      velocities: (vx, vy, vz) float32 CUDA tensors (N,), updated in place.
      fields: (Ex, Ey, Ez, Bx, By, Bz), contiguous float32 CUDA tensors of
        shape (ny, nx) each.
      cell_size: Grid spacing (dx, dy).
      charge_mass: Charge-to-mass ratio q/m [C/kg].
      dt: Time step [s].
    """
    ny, nx = fields[0].shape
    n = positions[0].numel()
    block = 1024
    _boris_kernel[(triton.cdiv(n, block),)](
        *positions, *velocities, *(f.contiguous() for f in fields), n, nx, ny,
        1.0 / cell_size[0], 1.0 / cell_size[1], 0.5 * charge_mass * dt, dt,
        BLOCK=block, num_warps=4)


def reference_boris_push(positions: Sequence[torch.Tensor],
                         velocities: Sequence[torch.Tensor],
                         fields: Sequence[torch.Tensor],
                         cell_size: Sequence[float], charge_mass: float,
                         dt: float
                         ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Out-of-place PyTorch reference of `boris_push` (also runs on CPU)."""
    x, y = positions
    e = [interp.reference_interpolate(f, x, y, cell_size) for f in fields[:3]]
    b = [interp.reference_interpolate(f, x, y, cell_size) for f in fields[3:]]
    h = 0.5 * charge_mass * dt
    v = torch.stack(list(velocities)) + h * torch.stack(e)
    t = h * torch.stack(b)
    s = 2.0 * t / (1.0 + (t * t).sum(0))
    v_prime = v + torch.cross(v, t, dim=0)
    v = v + torch.cross(v_prime, s, dim=0) + h * torch.stack(e)
    return [x + v[0] * dt, y + v[1] * dt], [v[0], v[1], v[2]]
