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
"""Bilinear grid-to-particle interpolation for particle-in-cell (Triton 2.0).

Gathering grid fields at particle positions is bandwidth bound and irregular.
When particles are sorted by cell (the output order of the fused radix sort),
the particles of one program fall into a compact patch of the grid, so the
four corner loads of neighboring particles hit the same L1/L2 lines. The
autotuner picks the tile size per GPU (the same source runs on A100 and
H100).

`bilinear_gather` is a `@triton.jit` helper that other kernels (the fused
Boris push in `src.physics_engine.pic_plasma_dynamics`) call directly.

Typical usage example:

  values = interpolate(field, x, y, cell_size=(dx, dy))
"""

from collections.abc import Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def bilinear_weights(x, y, inv_dx, inv_dy, nx, ny):
    """Base corner index and bilinear weights of positions (x, y).

    Grid nodes sit at integer multiples of the cell size; positions are
    clamped to the grid.

    Returns:
      (base, w00, w10, w01, w11), where base = iy * nx + ix indexes the
      lower-left node of a row-major (ny, nx) grid.
    """
    fx = x * inv_dx
    fy = y * inv_dy
    ix = tl.minimum(tl.maximum(fx.to(tl.int32), 0), nx - 2)
    iy = tl.minimum(tl.maximum(fy.to(tl.int32), 0), ny - 2)
    wx = tl.minimum(tl.maximum(fx - ix, 0.0), 1.0)
    wy = tl.minimum(tl.maximum(fy - iy, 0.0), 1.0)
    base = iy * nx + ix
    return (base, (1.0 - wx) * (1.0 - wy), wx * (1.0 - wy),
            (1.0 - wx) * wy, wx * wy)


@triton.jit
def bilinear_gather(field_ptr, base, w00, w10, w01, w11, nx, mask):
    """Interpolates a row-major (ny, nx) field from precomputed weights."""
    f00 = tl.load(field_ptr + base, mask=mask, other=0.0)
    f10 = tl.load(field_ptr + base + 1, mask=mask, other=0.0)
    f01 = tl.load(field_ptr + base + nx, mask=mask, other=0.0)
    f11 = tl.load(field_ptr + base + nx + 1, mask=mask, other=0.0)
    return f00 * w00 + f10 * w10 + f01 * w01 + f11 * w11


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256}, num_warps=2),
        triton.Config({'BLOCK': 512}, num_warps=4),
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=8),
    ],
    key=['n'],
)
@triton.jit
def _interpolate_kernel(field_ptr, x_ptr, y_ptr, out_ptr, n, inv_dx, inv_dy,
                        nx, ny, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    base, w00, w10, w01, w11 = bilinear_weights(x, y, inv_dx, inv_dy, nx, ny)
    value = bilinear_gather(field_ptr, base, w00, w10, w01, w11, nx, mask)
    tl.store(out_ptr + offs, value, mask=mask)


def interpolate(field: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                cell_size: Sequence[float]) -> torch.Tensor:
    """Bilinear interpolation of a node-centered 2D field at particles.

    Args:
      field: Contiguous float32 CUDA tensor of shape (ny, nx), ny, nx >= 2.
      x: Particle x positions, float32 CUDA tensor (N,).
      y: Particle y positions, float32 CUDA tensor (N,).
      cell_size: Grid spacing (dx, dy).

    Returns:
      Interpolated values, float32 (N,).
    """
    ny, nx = field.shape
    n = x.numel()
    out = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(n, meta['BLOCK']),)

    _interpolate_kernel[grid](field.contiguous(), x, y, out, n,
                              1.0 / cell_size[0], 1.0 / cell_size[1], nx, ny)
    return out


def reference_interpolate(field: torch.Tensor, x: torch.Tensor,
                          y: torch.Tensor,
                          cell_size: Sequence[float]) -> torch.Tensor:
    """PyTorch reference of `interpolate` (also runs on CPU)."""
    ny, nx = field.shape
    fx, fy = x / cell_size[0], y / cell_size[1]
    ix = fx.to(torch.int32).clamp(0, nx - 2)
    iy = fy.to(torch.int32).clamp(0, ny - 2)
    wx = (fx - ix).clamp(0.0, 1.0)
    wy = (fy - iy).clamp(0.0, 1.0)
    f = field.reshape(-1)
    base = (iy * nx + ix).long()
    return (f[base] * (1 - wx) * (1 - wy) + f[base + 1] * wx * (1 - wy) +
            f[base + nx] * (1 - wx) * wy + f[base + nx + 1] * wx * wy)
