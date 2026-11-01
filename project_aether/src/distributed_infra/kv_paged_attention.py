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
"""PagedAttention state cache for the serving system (report Part VI, 6.2).

Implements the paged key/value cache introduced by vLLM (W. Kwon et al.,
"vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention", blog post of
June 20, 2023):

  * Device memory is a pool of fixed-size blocks of `block_size` token slots.
  * Each sequence owns a block table (logical block -> physical block);
    blocks are allocated on demand, so the only internal fragmentation is the
    unused tail of each sequence's last block.
  * Blocks are reference counted. Forking a sequence (the digital twin's
    what-if branch rollouts) copies the block table, not the state; a shared
    block is copied only when a branch writes into it (copy-on-write).

`paged_attention` computes exact softmax attention directly against the paged
layout, streaming over each sequence's blocks with FlashAttention-style online
softmax. It serves chunked prefill (many query tokens with causal masking) and
decode (one query token) alike. Physical block 0 is reserved as a scratch
block for padding slots.

Typical usage example:

  cache = PagedKVCache(num_layers=4, num_blocks=1024, block_size=16,
                       num_heads=4, head_dim=32)
  cache.add_sequence(7)
  slots = cache.reserve(7, num_new_tokens=100)
"""

import collections
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

NULL_BLOCK = 0  # Scratch block that absorbs writes from padding tokens.


class CacheExhaustedError(RuntimeError):
    """No free physical block is left in the pool."""


class BlockAllocator:
    """Free list and reference counts of the physical block pool."""

    def __init__(self, num_blocks: int):
        if num_blocks < 2:
            raise ValueError('Need at least one block besides NULL_BLOCK.')
        self._free = collections.deque(range(1, num_blocks))
        self._refcount = np.zeros((num_blocks,), np.int64)

    @property
    def num_free(self) -> int:
        return len(self._free)

    def refcount(self, block: int) -> int:
        return int(self._refcount[block])

    def allocate(self) -> int:
        """Returns a free block with reference count 1.

        Raises:
          CacheExhaustedError: If the pool is empty.
        """
        if not self._free:
            raise CacheExhaustedError('Paged cache has no free blocks.')
        block = self._free.popleft()
        self._refcount[block] = 1
        return block

    def share(self, block: int) -> None:
        self._refcount[block] += 1

    def release(self, block: int) -> None:
        self._refcount[block] -= 1
        if self._refcount[block] == 0:
            self._free.append(block)


def _copy_block(k: jax.Array, v: jax.Array, src: jax.Array,
                dst: jax.Array) -> tuple[jax.Array, jax.Array]:
    return k.at[:, dst].set(k[:, src]), v.at[:, dst].set(v[:, src])


def _scatter_tokens(k: jax.Array, v: jax.Array, slots: jax.Array,
                    k_new: jax.Array,
                    v_new: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Writes (L, T, H, D) values to flat token slots (T,) of every layer."""

    def scatter(cache, values):
        num_layers, num_blocks, block_size = cache.shape[:3]
        flat = cache.reshape(num_layers, num_blocks * block_size,
                             *cache.shape[3:])
        return flat.at[:, slots].set(values).reshape(cache.shape)

    return scatter(k, k_new), scatter(v, v_new)


def _import(k: jax.Array, v: jax.Array, index: jax.Array, k_blocks: jax.Array,
            v_blocks: jax.Array) -> tuple[jax.Array, jax.Array]:
    return (k.at[:, index].set(k_blocks.astype(k.dtype)),
            v.at[:, index].set(v_blocks.astype(v.dtype)))


@functools.lru_cache(maxsize=None)
def _in_place(fn: Any, sharding: jax.sharding.Sharding | None) -> Any:
    """Jits `fn(k, v, ...) -> (k, v)` so that both caches update in place.

    The caches are donated, and with a sharded pool their output sharding is
    pinned to the input sharding; otherwise GSPMD may choose an equivalent
    output sharding that disables donation and copies the whole pool.
    """
    kwargs = {} if sharding is None else {'out_shardings': (sharding,
                                                             sharding)}
    return jax.jit(fn, donate_argnums=(0, 1), **kwargs)


@jax.jit
def _gather_blocks(cache: jax.Array, blocks: jax.Array) -> jax.Array:
    return jnp.take(cache, blocks, axis=1)


class PagedKVCache:
    """Device-resident paged K/V storage plus host-side block tables.

    Attributes:
      k: Keys, shape (num_layers, num_blocks, block_size, num_heads, head_dim).
      v: Values, same shape as `k`.
      block_size: Token slots per block.
      allocator: The block allocator.
      tables: Block table of every live sequence.
      lengths: Number of tokens stored for every live sequence.
    """

    def __init__(self, num_layers: int, num_blocks: int, block_size: int,
                 num_heads: int, head_dim: int, dtype: Any = jnp.float32,
                 sharding: jax.sharding.Sharding | None = None):
        shape = (num_layers, num_blocks, block_size, num_heads, head_dim)
        self.k = jnp.zeros(shape, dtype)
        self.v = jnp.zeros(shape, dtype)
        if sharding is not None:
            self.k = jax.device_put(self.k, sharding)
            self.v = jax.device_put(self.v, sharding)
        self.sharding = sharding
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        self.tables: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}

    def add_sequence(self, seq_id: int) -> None:
        if seq_id in self.tables:
            raise ValueError(f'Sequence {seq_id} already exists.')
        self.tables[seq_id] = []
        self.lengths[seq_id] = 0

    def free(self, seq_id: int) -> None:
        for block in self.tables.pop(seq_id):
            self.allocator.release(block)
        del self.lengths[seq_id]

    def fork(self, parent_id: int, child_id: int) -> None:
        """Creates `child_id` sharing every block of `parent_id`."""
        self.add_sequence(child_id)
        for block in self.tables[parent_id]:
            self.allocator.share(block)
        self.tables[child_id] = list(self.tables[parent_id])
        self.lengths[child_id] = self.lengths[parent_id]

    def preallocate(self, seq_id: int, max_tokens: int) -> None:
        """Reserves blocks for `max_tokens` up front (contiguous baseline).

        This emulates the per-request max-length reservation of serving
        systems without paging; the blocks are filled by later `reserve`
        calls, and unused ones are wasted until the sequence is freed.
        """
        table = self.tables[seq_id]
        while len(table) * self.block_size < max_tokens:
            table.append(self.allocator.allocate())

    def blocks_needed(self, seq_id: int, num_new_tokens: int) -> int:
        """New blocks `reserve` would allocate (copy-on-write included)."""
        length = self.lengths[seq_id]
        table = self.tables[seq_id]
        total = -(-(length + num_new_tokens) // self.block_size)
        needed = total - len(table)
        tail_shared = (length % self.block_size != 0 and
                       self.allocator.refcount(table[-1]) > 1)
        return needed + int(tail_shared and num_new_tokens > 0)

    def reserve(self, seq_id: int, num_new_tokens: int) -> np.ndarray:
        """Allocates slots for the next tokens of a sequence.

        Args:
          seq_id: Sequence to extend.
          num_new_tokens: Number of tokens about to be written.

        Returns:
          Flat slot index (block * block_size + offset) of each new token.

        Raises:
          CacheExhaustedError: If the pool runs out of blocks.
        """
        table = self.tables[seq_id]
        length = self.lengths[seq_id]
        if num_new_tokens > 0 and length % self.block_size != 0:
            tail = table[-1]
            if self.allocator.refcount(tail) > 1:  # Copy-on-write.
                fresh = self.allocator.allocate()
                self.k, self.v = _in_place(_copy_block, self.sharding)(
                    self.k, self.v, tail, fresh)
                self.allocator.release(tail)
                table[-1] = fresh
        slots = np.empty((num_new_tokens,), np.int32)
        for i in range(num_new_tokens):
            position = length + i
            if position // self.block_size >= len(table):
                table.append(self.allocator.allocate())
            block = table[position // self.block_size]
            slots[i] = block * self.block_size + position % self.block_size
        self.lengths[seq_id] = length + num_new_tokens
        return slots

    def write(self, slots: jax.Array, k_new: jax.Array,
              v_new: jax.Array) -> None:
        """Stores (num_layers, T, num_heads, head_dim) keys/values at slots."""
        self.k, self.v = _in_place(_scatter_tokens, self.sharding)(
            self.k, self.v, slots, k_new, v_new)

    def block_tables(self, seq_ids: list[int], max_blocks: int) -> np.ndarray:
        """Block tables of `seq_ids` padded with NULL_BLOCK, (B, max_blocks)."""
        out = np.full((len(seq_ids), max_blocks), NULL_BLOCK, np.int32)
        for row, seq_id in enumerate(seq_ids):
            table = self.tables[seq_id]
            out[row, :len(table)] = table
        return out

    def export_blocks(self, seq_id: int) -> tuple[jax.Array, jax.Array]:
        """Gathers a sequence's K and V blocks, (layers, n_blocks, bs, H, D)."""
        blocks = jnp.asarray(self.tables[seq_id], jnp.int32)
        return _gather_blocks(self.k, blocks), _gather_blocks(self.v, blocks)

    def import_blocks(self, seq_id: int, k_blocks: jax.Array,
                      v_blocks: jax.Array, num_tokens: int) -> None:
        """Adds a sequence whose state arrives as whole blocks (handoff)."""
        self.add_sequence(seq_id)
        table = [self.allocator.allocate()
                 for _ in range(k_blocks.shape[1])]
        index = jnp.asarray(table, jnp.int32)
        self.k, self.v = _in_place(_import, self.sharding)(
            self.k, self.v, index, k_blocks, v_blocks)
        self.tables[seq_id] = table
        self.lengths[seq_id] = num_tokens

    def utilization(self) -> float:
        """Filled slots / allocated slots of all live blocks (1 - waste).

        A block shared by several sequences counts once.
        """
        filled: dict[int, int] = {}
        for seq_id, table in self.tables.items():
            for i, block in enumerate(table):
                tokens = min(self.block_size,
                             self.lengths[seq_id] - i * self.block_size)
                filled[block] = max(filled.get(block, 0), tokens)
        if not filled:
            return 1.0
        return sum(filled.values()) / (len(filled) * self.block_size)


def paged_attention(q: jax.Array, k_cache: jax.Array, v_cache: jax.Array,
                    block_tables: jax.Array, context_lens: jax.Array,
                    q_positions: jax.Array) -> jax.Array:
    """Causal softmax attention of new tokens against a paged K/V cache.

    Args:
      q: Queries of shape (B, T, H, D).
      k_cache: One layer's keys, (num_blocks, block_size, H, D).
      v_cache: One layer's values, same shape.
      block_tables: Physical blocks of each sequence, (B, max_blocks).
      context_lens: Tokens stored for each sequence (including the new ones),
        shape (B,).
      q_positions: Absolute position of each query token, (B, T). A query
        attends to keys at positions <= its own and < context_len.

    Returns:
      Attention output of shape (B, T, H, D).
    """
    b, t, h, d = q.shape
    block_size = k_cache.shape[1]
    scale = 1.0 / np.sqrt(d)

    def body(carry, j):
        m, l, acc = carry
        blocks = block_tables[:, j]  # (B,)
        k_blk = k_cache[blocks]  # (B, bs, H, D)
        v_blk = v_cache[blocks]
        s = jnp.einsum('bthd,bshd->bhts', q, k_blk) * scale
        key_pos = j * block_size + jnp.arange(block_size)
        visible = ((key_pos[None, None, :] <= q_positions[:, :, None]) &
                   (key_pos[None, None, :] < context_lens[:, None, None]))
        s = jnp.where(visible[:, None, :, :], s, -jnp.inf)
        m_new = jnp.maximum(m, jnp.max(s, axis=-1))
        m_safe = jnp.where(jnp.isfinite(m_new), m_new, 0.0)
        p = jnp.exp(s - m_safe[..., None])
        correction = jnp.exp(jnp.where(jnp.isfinite(m), m - m_safe, -jnp.inf))
        l = l * correction + jnp.sum(p, axis=-1)
        acc = acc * correction[..., None] + jnp.einsum('bhts,bshd->bhtd', p,
                                                       v_blk)
        return (m_new, l, acc), None

    init = (jnp.full((b, h, t), -jnp.inf, q.dtype),
            jnp.zeros((b, h, t), q.dtype), jnp.zeros((b, h, t, d), q.dtype))
    (_, l, acc), _ = jax.lax.scan(body, init,
                                  jnp.arange(block_tables.shape[1]))
    out = acc / jnp.maximum(l, 1e-30)[..., None]
    return out.transpose(0, 2, 1, 3)


def contiguous_attention(q: jax.Array, k: jax.Array, v: jax.Array,
                         context_lens: jax.Array,
                         q_positions: jax.Array) -> jax.Array:
    """Reference attention over contiguous (B, S, H, D) key/value buffers."""
    s = jnp.einsum('bthd,bshd->bhts', q, k) / np.sqrt(q.shape[-1])
    key_pos = jnp.arange(k.shape[1])
    visible = ((key_pos[None, None, :] <= q_positions[:, :, None]) &
               (key_pos[None, None, :] < context_lens[:, None, None]))
    s = jnp.where(visible[:, None], s, -jnp.inf)
    return jnp.einsum('bhts,bshd->bthd', jax.nn.softmax(s, axis=-1), v)
