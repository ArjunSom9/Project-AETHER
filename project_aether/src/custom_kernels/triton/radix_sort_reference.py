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
"""NumPy replay of the fused Triton radix sort (no GPU required).

`fused_radix_sort` implements an Onesweep-style LSD radix sort with
decoupled look-back in Triton. This module holds the shared constants and
replays the exact algorithm -- tiling, packed status words, per-digit
look-back, digit bases, and stable in-tile ranks -- in NumPy, so the
algorithm can be tested on CPU-only machines where Triton is unavailable.
"""

import math
from typing import NamedTuple

import numpy as np

RADIX_BITS = 7
RADIX = 1 << RADIX_BITS
SUBTILE = 128  # Keys ranked per tensor-core dot.
SUBTILES_PER_TILE = 8
TILE = SUBTILE * SUBTILES_PER_TILE
SHUFFLE_BITS = 7
MAX_PASSES = 4  # Padded (power of two) pass dimension of the histograms.

# Status word: bits 29-30 hold the flag, bits 0-28 the count.
FLAG_AGGREGATE = 1 << 29
FLAG_INCLUSIVE = 1 << 30
FLAG_MASK = FLAG_AGGREGATE | FLAG_INCLUSIVE
VALUE_MASK = FLAG_AGGREGATE - 1


class CellGrid(NamedTuple):
    """Uniform Cartesian grid of DSMC collision cells.

    Attributes:
      lo: Lower corner (x, y, z).
      hi: Upper corner (x, y, z).
      cells: Number of cells (nx, ny, nz).
    """
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    cells: tuple[int, int, int]

    @property
    def num_cells(self) -> int:
        return self.cells[0] * self.cells[1] * self.cells[2]


def num_passes(num_cells: int, shuffle_bits: int = SHUFFLE_BITS) -> int:
    """Digit passes needed for keys of (cell bits + shuffle bits)."""
    key_bits = max(1, (num_cells - 1).bit_length()) + shuffle_bits
    if key_bits > 31:
        raise ValueError(f'Sort keys need {key_bits} bits; at most 31 fit.')
    return math.ceil(key_bits / RADIX_BITS)


# ------------------------------------------------------------------------------
# NumPy replay of the kernels' algorithm (for CPU testing).
# ------------------------------------------------------------------------------


def _lookback(status: np.ndarray, tile: int) -> np.ndarray:
    """Exactly the per-digit look-back loop of `_onesweep_kernel`."""
    exclusive = np.zeros((RADIX,), np.int64)
    pred = np.full((RADIX,), tile - 1, np.int64)
    done = pred < 0
    while not done.all():
        word = np.where(done, FLAG_INCLUSIVE,
                        status[np.maximum(pred, 0), np.arange(RADIX)])
        state = word & FLAG_MASK
        ready = (state != 0) & ~done
        if not ready.any():
            raise RuntimeError('Look-back would spin: a predecessor tile has '
                               'not published.')
        exclusive += np.where(ready, word & VALUE_MASK, 0)
        advance = ready & (state == FLAG_AGGREGATE)
        pred = np.where(advance, pred - 1, pred)
        done = done | (ready & (state == FLAG_INCLUSIVE)) | (pred < 0)
    return exclusive


def onesweep_reference(keys: np.ndarray, values: np.ndarray, passes: int,
                       rng: np.random.Generator,
                       tile: int = TILE) -> tuple[np.ndarray, np.ndarray]:
    """Replays the sort with randomized look-back states.

    Tiles run in tile-id order, but before each look-back a random subset of
    the predecessors is left in the AGGREGATE state, emulating tiles that
    have published their aggregate but not yet their inclusive prefix.

    Args:
      keys: Non-negative int32 keys.
      values: Payload carried with the keys.
      passes: Number of RADIX_BITS digit passes.
      rng: Random generator choosing the emulated publication states.
      tile: Keys per tile.

    Returns:
      Sorted (keys, values); the sort is stable.
    """
    keys, values = keys.astype(np.int64), values.copy()
    n = keys.size
    num_tiles = -(-n // tile)
    for p in range(passes):
        digits = (keys >> (p * RADIX_BITS)) & (RADIX - 1)
        total = np.bincount(digits, minlength=RADIX)
        digit_base = np.concatenate([[0], np.cumsum(total)[:-1]])
        status = np.zeros((num_tiles, RADIX), np.int64)
        tile_hist = np.zeros((num_tiles, RADIX), np.int64)
        inclusive = np.zeros((num_tiles, RADIX), np.int64)
        out_keys, out_values = np.empty_like(keys), np.empty_like(values)
        for t in range(num_tiles):
            d = digits[t * tile:(t + 1) * tile]
            tile_hist[t] = np.bincount(d, minlength=RADIX)
            # Every predecessor (other than tile 0, which is always
            # inclusive) randomly shows only its aggregate or already its
            # inclusive prefix, as concurrent tiles would.
            for q in range(1, t):
                status[q] = (inclusive[q] | FLAG_INCLUSIVE
                             if rng.random() < 0.5 else
                             tile_hist[q] | FLAG_AGGREGATE)
            status[t] = tile_hist[t] | (FLAG_INCLUSIVE if t == 0 else
                                        FLAG_AGGREGATE)
            exclusive = _lookback(status, t)
            inclusive[t] = exclusive + tile_hist[t]
            status[t] = inclusive[t] | FLAG_INCLUSIVE
            # Stable rank inside the tile.
            local = np.zeros_like(d)
            seen = np.zeros((RADIX,), np.int64)
            for i, digit in enumerate(d):
                local[i] = seen[digit]
                seen[digit] += 1
            dest = digit_base[d] + exclusive[d] + local
            out_keys[dest] = keys[t * tile:(t + 1) * tile]
            out_values[dest] = values[t * tile:(t + 1) * tile]
        keys, values = out_keys, out_values
    return keys.astype(np.int32), values
