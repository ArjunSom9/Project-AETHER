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
"""Block-sparse attention for long particle trajectories (report Part IV).

Interactions along a trajectory are spatially local, with a few global summary
tokens, so attention uses a fixed block layout combining

  * sliding-window blocks |i - j| <= window (Longformer, Beltagy et al. 2020),
  * global blocks: the first `global_blocks` block rows and columns attend
    everywhere (BigBird, Zaheer et al. 2020; Longformer's global attention).

`window_for_density` picks the window that makes the layout 75% sparse, the
operating point of report Part IV and of the CUTLASS block-sparse GEMM, which
implements this same (window, global) pattern with a compile-time mask.

`block_sparse_attention` evaluates exact softmax attention over the active
blocks only, with FlashAttention-style tiling (T. Dao et al., NeurIPS 2022):
each query tile streams over its active key tiles while carrying the running
row maximum and normalizer (online softmax), so the full score matrix is never
materialized.

Typical usage example:

  layout = block_sparse_layout(num_blocks=64, window=7, global_blocks=1)
  out = block_sparse_attention(q, k, v, layout, block_size=64)
"""

import math

import jax
import jax.numpy as jnp
import numpy as np


def block_sparse_layout(num_blocks: int, window: int,
                        global_blocks: int = 1) -> np.ndarray:
    """Boolean block mask of shape (num_blocks, num_blocks).

    Args:
      num_blocks: Number of blocks along the sequence.
      window: Half-width of the band of local blocks.
      global_blocks: Number of leading global block rows and columns.

    Returns:
      layout[i, j] is True when query block i attends to key block j.
    """
    i, j = np.meshgrid(np.arange(num_blocks), np.arange(num_blocks),
                       indexing='ij')
    return ((np.abs(i - j) <= window) | (i < global_blocks) |
            (j < global_blocks))


def layout_density(layout: np.ndarray) -> float:
    """Fraction of active blocks (1 - sparsity)."""
    return float(np.mean(layout))


def window_for_density(num_blocks: int, global_blocks: int = 1,
                       target_density: float = 0.25) -> int:
    """Band half-width whose layout density is closest to `target_density`."""
    best_window, best_err = 0, math.inf
    for window in range(num_blocks):
        err = abs(layout_density(
            block_sparse_layout(num_blocks, window, global_blocks)) -
                  target_density)
        if err < best_err:
            best_window, best_err = window, err
    return best_window


def token_mask(layout: np.ndarray, block_size: int) -> np.ndarray:
    """Expands a block layout to a (L, L) token-level mask."""
    return np.kron(layout, np.ones((block_size, block_size), dtype=bool))


def dense_masked_attention(q: jax.Array, k: jax.Array, v: jax.Array,
                           layout: np.ndarray, block_size: int) -> jax.Array:
    """Reference: materializes the masked (L, L) score matrix.

    Args:
      q: Queries of shape (batch, heads, L, d).
      k: Keys of shape (batch, heads, L, d).
      v: Values of shape (batch, heads, L, d).
      layout: Block layout.
      block_size: Tokens per block; L == layout.shape[0] * block_size.

    Returns:
      Attention output of shape (batch, heads, L, d).
    """
    scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(q.shape[-1])
    mask = jnp.asarray(token_mask(layout, block_size))
    scores = jnp.where(mask, scores, -jnp.inf)
    return jnp.einsum('bhqk,bhkd->bhqd', jax.nn.softmax(scores, axis=-1), v)


def _active_key_blocks(layout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per query block, the active key-block indices (valid entries first)."""
    counts = layout.sum(axis=1)
    max_active = int(counts.max())
    index = np.zeros((layout.shape[0], max_active), dtype=np.int32)
    valid = np.zeros((layout.shape[0], max_active), dtype=bool)
    for row in range(layout.shape[0]):
        cols = np.nonzero(layout[row])[0]
        index[row, :len(cols)] = cols
        valid[row, :len(cols)] = True
    return index, valid


def block_sparse_attention(q: jax.Array, k: jax.Array, v: jax.Array,
                           layout: np.ndarray, block_size: int) -> jax.Array:
    """Exact softmax attention restricted to the active blocks of `layout`.

    Work and memory scale with the number of active blocks. Every query block
    must have at least one active key block (true for any layout from
    `block_sparse_layout`, whose diagonal is always active).

    Args:
      q: Queries of shape (batch, heads, L, d).
      k: Keys of shape (batch, heads, L, d).
      v: Values of shape (batch, heads, L, d).
      layout: Static block layout (NumPy bool array).
      block_size: Tokens per block.

    Returns:
      Attention output of shape (batch, heads, L, d), equal (up to rounding) to
      `dense_masked_attention`.

    Raises:
      ValueError: If the sequence length does not match the layout.
    """
    b, h, length, d = q.shape
    num_blocks = layout.shape[0]
    if length != num_blocks * block_size:
        raise ValueError(
            f'Sequence length {length} != {num_blocks} blocks x {block_size}.')
    index, valid = _active_key_blocks(layout)
    scale = 1.0 / math.sqrt(d)
    qb = q.reshape(b, h, num_blocks, block_size, d)
    kb = k.reshape(b, h, num_blocks, block_size, d)
    vb = v.reshape(b, h, num_blocks, block_size, d)

    def query_tile(q_tile, idx_row, valid_row):
        # q_tile: (b, h, block, d). Online softmax over the active key tiles.
        def body(carry, inputs):
            m, l, acc = carry
            key_block, is_valid = inputs
            k_tile = jnp.take(kb, key_block, axis=2)
            v_tile = jnp.take(vb, key_block, axis=2)
            s = jnp.einsum('bhqd,bhkd->bhqk', q_tile, k_tile) * scale
            s = jnp.where(is_valid, s, -jnp.inf)
            m_new = jnp.maximum(m, jnp.max(s, axis=-1))
            p = jnp.exp(s - m_new[..., None])
            correction = jnp.exp(m - m_new)
            l = l * correction + jnp.sum(p, axis=-1)
            acc = (acc * correction[..., None] +
                   jnp.einsum('bhqk,bhkd->bhqd', p, v_tile))
            return (m_new, l, acc), None

        init = (jnp.full((b, h, block_size), -jnp.inf, q.dtype),
                jnp.zeros((b, h, block_size), q.dtype),
                jnp.zeros((b, h, block_size, d), q.dtype))
        (_, l, acc), _ = jax.lax.scan(body, init, (idx_row, valid_row))
        return acc / l[..., None]

    out = jax.vmap(query_tile, in_axes=(2, 0, 0), out_axes=2)(
        qb, jnp.asarray(index), jnp.asarray(valid))
    return out.reshape(b, h, length, d)
