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
"""Fused Triton radix sort for DSMC particle binning (report Part II).

The DSMC binning stage (cell id -> histogram -> scan -> scatter) is fused into
an Onesweep-style least-significant-digit radix sort (A. Adinets and D.
Merrill, arXiv:2206.01784):

  1. `_bin_keys_kernel` reads the Structure-of-Arrays positions once, computes
     every particle's collision cell in registers, appends counter-based
     Philox random bits (J. Salmon et al., SC '11) to the key, writes the
     (key, particle index) pairs, and accumulates the digit histograms of
     *all* sort passes from that single read. The random low bits shuffle
     particles inside each cell, so adjacent particles after the sort are
     random NTC collision partners: collision-partner search is fused into
     the sort and no RNG state is stored in memory.
  2. `_onesweep_kernel` performs one digit pass per launch. Each program
     takes a tile id from an atomic counter, builds its tile's digit
     histogram, publishes it, and resolves its global offsets with
     single-pass decoupled look-back over its predecessors' status words.
     Each status word packs a 2-bit flag and a 29-bit count into one int32,
     so publication needs no memory-ordering assumptions. Keys are then
     ranked stably inside the tile and scattered directly to their final
     positions. Keys are read from DRAM once per pass; the second sweep over a
     tile's sub-tiles hits L1. The last pass also accumulates per-cell
     particle counts (the cell-density statistics the collision stage needs).

The kernels use only Triton 2.0.0 language features (the newest Triton
release before August 25, 2023). In particular, Triton 2.0 has no
`tl.sort`, `tl.cumsum`, or `tl.histogram`; in-tile ranking therefore uses a
tensor-core `tl.dot` of a strictly lower-triangular matrix with the one-hot
digit matrix, which yields every key's count of earlier equal digits.

`radix_sort_reference.onesweep_reference` replays the identical tiling and
look-back protocol in NumPy, so the algorithm is testable without a GPU.

Typical usage example:

  result = sort_particles_by_cell(x, y, z, grid, seed=step)
  partner_order = result.particle_index  # Cell-sorted, shuffled in cells.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from src.custom_kernels.triton import radix_sort_reference as ref


@triton.jit(do_not_specialize=['seed'])
def _bin_keys_kernel(
    x_ptr, y_ptr, z_ptr,  # Structure-of-Arrays positions.
    keys_ptr, values_ptr,  # Outputs: sort keys and particle indices.
    hist_ptr,  # Output: digit histograms, (MAX_PASSES, RADIX), zeroed.
    n,
    x0, y0, z0, inv_dx, inv_dy, inv_dz,
    nx, ny, nz,
    seed,
    NUM_PASSES: tl.constexpr,
    SHUFFLE: tl.constexpr,
    PASS_PAD: tl.constexpr,
    SUB: tl.constexpr,
    SUBTILES: tl.constexpr,
    RADIX_C: tl.constexpr,
    RADIX_BITS_C: tl.constexpr,
):
    # Triton 2.0 treats every annotated argument as a compile-time constant,
    # so only true meta-parameters carry `tl.constexpr` annotations.
    pid = tl.program_id(0)
    rows = tl.arange(0, SUB)
    bins = tl.arange(0, RADIX_C)
    pass_ids = tl.arange(0, PASS_PAD)
    hist = tl.zeros([PASS_PAD, RADIX_C], dtype=tl.int32)
    for s in range(SUBTILES):
        offs = (pid * SUBTILES + s) * SUB + rows
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        z = tl.load(z_ptr + offs, mask=mask, other=0.0)
        ix = tl.minimum(tl.maximum(((x - x0) * inv_dx).to(tl.int32), 0),
                        nx - 1)
        iy = tl.minimum(tl.maximum(((y - y0) * inv_dy).to(tl.int32), 0),
                        ny - 1)
        iz = tl.minimum(tl.maximum(((z - z0) * inv_dz).to(tl.int32), 0),
                        nz - 1)
        cell = (iz * ny + iy) * nx + ix
        # Philox4x32-10 keyed by `seed`, counter = particle index.
        bits = tl.randint(seed, offs)
        shuffle = (bits >> (32 - SHUFFLE)).to(tl.int32)
        key = (cell << SHUFFLE) | shuffle
        tl.store(keys_ptr + offs, key, mask=mask)
        tl.store(values_ptr + offs, offs, mask=mask)
        for p in tl.static_range(NUM_PASSES):
            digit = tl.where(mask, (key >> (p * RADIX_BITS_C)) & (RADIX_C - 1),
                             RADIX_C)
            counts = tl.sum((digit[:, None] == bins[None, :]).to(tl.int32),
                            axis=0)
            hist += tl.where(pass_ids[:, None] == p, counts[None, :], 0)
    tl.atomic_add(hist_ptr + pass_ids[:, None] * RADIX_C + bins[None, :], hist,
                  mask=pass_ids[:, None] < NUM_PASSES)


@triton.jit(do_not_specialize=['pass_index', 'cell_shift'])
def _onesweep_kernel(
    keys_in, values_in, keys_out, values_out,
    hist_ptr,  # Digit histograms of all passes, (MAX_PASSES, RADIX).
    status_ptr,  # Per-tile status words, (num_tiles, RADIX), zeroed.
    counter_ptr,  # Tile-id counter, zeroed.
    cell_count_ptr,  # Per-cell particle counts (written on the last pass).
    n,
    pass_index,
    cell_shift,
    IS_LAST: tl.constexpr,
    SUB: tl.constexpr,
    SUBTILES: tl.constexpr,
    RADIX_C: tl.constexpr,
    RADIX_BITS_C: tl.constexpr,
    FLAG_AGG: tl.constexpr,
    FLAG_INC: tl.constexpr,
    FLAG_BITS: tl.constexpr,
    VALUE_BITS: tl.constexpr,
):
    # Tiles are numbered in the order programs start, so every predecessor a
    # tile waits for is already running (forward-progress guarantee).
    tile = tl.atomic_add(counter_ptr, 1)
    rows = tl.arange(0, SUB)
    bins = tl.arange(0, RADIX_C)
    shift = pass_index * RADIX_BITS_C
    base = tile * (SUB * SUBTILES)

    # 1. Digit histogram of the tile.
    tile_hist = tl.zeros([RADIX_C], dtype=tl.int32)
    for s in range(SUBTILES):
        offs = base + s * SUB + rows
        mask = offs < n
        key = tl.load(keys_in + offs, mask=mask, other=0)
        digit = tl.where(mask, (key >> shift) & (RADIX_C - 1), RADIX_C)
        tile_hist += tl.sum((digit[:, None] == bins[None, :]).to(tl.int32),
                            axis=0)

    # 2. Publish the aggregate; tile 0 is its own inclusive prefix.
    status_row = status_ptr + tile * RADIX_C + bins
    flag = FLAG_AGG + (tile == 0).to(tl.int32) * FLAG_AGG
    tl.store(status_row, tile_hist | flag)

    # 3. Decoupled look-back, independently per digit.
    exclusive = tl.zeros([RADIX_C], dtype=tl.int32)
    pred = tl.zeros([RADIX_C], dtype=tl.int32) + (tile - 1)
    done = (pred < 0).to(tl.int32)
    while tl.sum(1 - done, axis=0) > 0:
        word = tl.load(status_ptr + tl.maximum(pred, 0) * RADIX_C + bins,
                       mask=done == 0, other=FLAG_INC, volatile=True)
        state = word & FLAG_BITS
        ready = (state != 0) & (done == 0)
        exclusive += tl.where(ready, word & VALUE_BITS, 0)
        advance = ready & (state == FLAG_AGG)
        pred = tl.where(advance, pred - 1, pred)
        done = tl.where((ready & (state == FLAG_INC)) | (pred < 0), 1, done)

    # 4. Publish the inclusive prefix for the successors.
    tl.store(status_row, (exclusive + tile_hist) | FLAG_INC,
             mask=(bins < RADIX_C) & (tile > 0))

    # 5. Start of every digit bucket in the output.
    total = tl.load(hist_ptr + pass_index * RADIX_C + bins)
    digit_base = tl.sum(tl.where(bins[None, :] < bins[:, None], total[None, :],
                                 0), axis=1)
    offset = digit_base + exclusive

    # 6. Stable in-tile rank and scatter.
    strictly_lower = (rows[:, None] > rows[None, :]).to(tl.float16)
    running = tl.zeros([RADIX_C], dtype=tl.int32)
    for s in range(SUBTILES):
        offs = base + s * SUB + rows
        mask = offs < n
        key = tl.load(keys_in + offs, mask=mask, other=0)
        value = tl.load(values_in + offs, mask=mask, other=0)
        digit = tl.where(mask, (key >> shift) & (RADIX_C - 1), RADIX_C)
        match = digit[:, None] == bins[None, :]
        # earlier[i, r] = number of keys j < i in this sub-tile with digit r.
        earlier = tl.dot(strictly_lower, match.to(tl.float16))
        local_rank = tl.sum(tl.where(match, earlier, 0.0), axis=1).to(tl.int32)
        start = tl.sum(tl.where(match, (offset + running)[None, :], 0), axis=1)
        destination = start + local_rank
        tl.store(keys_out + destination, key, mask=mask)
        tl.store(values_out + destination, value, mask=mask)
        running += tl.sum(match.to(tl.int32), axis=0)
        if IS_LAST:
            # Fused epilogue: per-cell density statistics.
            tl.atomic_add(cell_count_ptr + (key >> cell_shift), 1, mask=mask)


class SortResult(NamedTuple):
    """Output of `sort_particles_by_cell`.

    Attributes:
      keys: Sorted keys (cell << SHUFFLE_BITS | shuffle bits), int32 (N,).
      particle_index: Original index of every sorted particle, int32 (N,).
      cell_counts: Particles per cell, int32 (C,).
      cell_start: Exclusive prefix sum of cell_counts, int32 (C + 1,).
    """
    keys: torch.Tensor
    particle_index: torch.Tensor
    cell_counts: torch.Tensor
    cell_start: torch.Tensor


def sort_particles_by_cell(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                           grid: ref.CellGrid, seed: int) -> SortResult:
    """Bins particles by collision cell (shuffled in-cell) on the GPU.

    Args:
      x: Particle x positions, float32 CUDA tensor (N,).
      y: Particle y positions, float32 CUDA tensor (N,).
      z: Particle z positions, float32 CUDA tensor (N,).
      grid: Collision-cell grid.
      seed: Philox key; use a different value every timestep.

    Returns:
      The sorted keys, permutation, and per-cell counts and offsets.

    Raises:
      ValueError: If the inputs are not float32 CUDA tensors of equal size.
    """
    for t in (x, y, z):
        if not t.is_cuda or t.dtype != torch.float32 or t.numel() != x.numel():
            raise ValueError('Positions must be float32 CUDA tensors of the '
                             'same length.')
    n = x.numel()
    device = x.device
    passes = ref.num_passes(grid.num_cells)
    keys = torch.empty(n, dtype=torch.int32, device=device)
    values = torch.empty(n, dtype=torch.int32, device=device)
    hist = torch.zeros(ref.MAX_PASSES * ref.RADIX, dtype=torch.int32,
                       device=device)
    num_tiles = triton.cdiv(n, ref.TILE)
    inv = [c / (h - l) for l, h, c in zip(grid.lo, grid.hi, grid.cells)]
    _bin_keys_kernel[(num_tiles,)](
        x, y, z, keys, values, hist, n, *grid.lo, *inv, *grid.cells, seed,
        NUM_PASSES=passes, SHUFFLE=ref.SHUFFLE_BITS, PASS_PAD=ref.MAX_PASSES,
        SUB=ref.SUBTILE, SUBTILES=ref.SUBTILES_PER_TILE, RADIX_C=ref.RADIX,
        RADIX_BITS_C=ref.RADIX_BITS, num_warps=4)

    keys_alt = torch.empty_like(keys)
    values_alt = torch.empty_like(values)
    cell_counts = torch.zeros(grid.num_cells, dtype=torch.int32,
                              device=device)
    status = torch.empty(num_tiles * ref.RADIX, dtype=torch.int32,
                         device=device)
    counter = torch.empty(1, dtype=torch.int32, device=device)
    for p in range(passes):
        status.zero_()
        counter.zero_()
        _onesweep_kernel[(num_tiles,)](
            keys, values, keys_alt, values_alt, hist, status, counter,
            cell_counts, n, p, ref.SHUFFLE_BITS, IS_LAST=p == passes - 1,
            SUB=ref.SUBTILE, SUBTILES=ref.SUBTILES_PER_TILE, RADIX_C=ref.RADIX,
            RADIX_BITS_C=ref.RADIX_BITS, FLAG_AGG=ref.FLAG_AGGREGATE,
            FLAG_INC=ref.FLAG_INCLUSIVE, FLAG_BITS=ref.FLAG_MASK,
            VALUE_BITS=ref.VALUE_MASK, num_warps=8)
        keys, keys_alt = keys_alt, keys
        values, values_alt = values_alt, values

    cell_start = torch.zeros(grid.num_cells + 1, dtype=torch.int32,
                             device=device)
    cell_start[1:] = torch.cumsum(cell_counts, 0)
    return SortResult(keys, values, cell_counts, cell_start)
