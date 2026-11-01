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
"""Batch assembly shared by the prefill and decode pools.

Iteration-level scheduling (Orca, G.-I. Yu et al., OSDI 2022) changes the
batch composition every iteration. `run_batch` packs the selected sequences'
new tokens into padded arrays, reserves their cache slots, and calls the
jitted model. Batch size, chunk length, and block-table width are rounded up
to powers of two so XLA compiles only a handful of shapes.

Typical usage example:

  predictions = model_runner.run_batch(params, cache, cfg, [3, 9],
                                       [prompt_chunk, next_token[None]])
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from src.distributed_infra import kv_paged_attention
from src.surrogate_models import trajectory_transformer


def bucket(n: int) -> int:
    """Smallest power of two >= n (and >= 1)."""
    return 1 << max(0, (n - 1).bit_length())


def tensor_mesh(devices: list[jax.Device]) -> jax.sharding.Mesh:
    """One-axis ('tensor') mesh over a device pool."""
    return jax.sharding.Mesh(np.asarray(devices), ('tensor',))


def shard_params(params: Any, mesh: jax.sharding.Mesh) -> Any:
    """Places parameters with the Megatron tensor-parallel layout on `mesh`.

    The annotations are the only parallelization input: GSPMD propagates them
    through the jitted model and inserts the required all-reduces (Y. Xu et
    al., "GSPMD", arXiv:2105.04663).
    """
    specs = trajectory_transformer.tensor_parallel_specs(params)
    spec_type = jax.sharding.PartitionSpec
    return jax.tree_util.tree_map(
        lambda spec, x: jax.device_put(
            x, jax.sharding.NamedSharding(mesh, spec)),
        specs, params, is_leaf=lambda node: isinstance(node, spec_type))


def kv_sharding(mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
    """Shards (layers, blocks, block_size, heads, head_dim) over heads.

    The spec omits the trailing replicated dimension because that is the
    canonical form XLA returns for outputs; an equivalent but differently
    spelled spec would be a different jit cache key and force recompiles.
    """
    return jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(None, None, None, 'tensor'))


def replicated(mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


@functools.lru_cache(maxsize=None)
def sharded_forward(mesh: jax.sharding.Mesh) -> Any:
    """The model step jitted for one device pool.

    The output shardings are pinned to the cache layout so the donated
    caches are updated in place; left unspecified, GSPMD may pick an
    equivalent but distinct output sharding, which silently disables buffer
    donation and copies the whole cache every step.
    """
    kv = kv_sharding(mesh)
    return jax.jit(trajectory_transformer.apply_chunk,
                   static_argnames='cfg', donate_argnums=(1, 2),
                   out_shardings=(replicated(mesh), kv, kv))


def run_batch(params: Any, cache: kv_paged_attention.PagedKVCache,
              cfg: trajectory_transformer.TrajectoryConfig,
              seq_ids: list[int], chunks: list[np.ndarray],
              mesh: jax.sharding.Mesh | None = None) -> list[np.ndarray]:
    """Runs one model iteration over a batch of sequences.

    Args:
      params: Model parameters.
      cache: Paged cache holding every sequence in `seq_ids`.
      cfg: Model configuration.
      seq_ids: Sequences in the batch.
      chunks: New tokens of each sequence, arrays of shape (T_i, F).
      mesh: Device pool the cache and parameters live on, if sharded.

    Returns:
      Per-sequence predictions of shape (T_i, F); row j predicts the token
      after chunk row j.

    Raises:
      kv_paged_attention.CacheExhaustedError: If slots cannot be reserved.
    """
    lengths_before = [cache.lengths[s] for s in seq_ids]
    slot_lists = [cache.reserve(s, len(c)) for s, c in zip(seq_ids, chunks)]

    b = bucket(len(seq_ids))
    t = bucket(max(len(c) for c in chunks))
    max_blocks = bucket(max(len(cache.tables[s]) for s in seq_ids))
    feature_dim = cfg.feature_dim

    tokens = np.zeros((b, t, feature_dim), np.float32)
    positions = np.zeros((b, t), np.int32)
    # Padding tokens write into the scratch block.
    slots = np.full((b, t), kv_paged_attention.NULL_BLOCK * cache.block_size,
                    np.int32)
    context_lens = np.ones((b,), np.int32)
    for row, (chunk, start, slot) in enumerate(
            zip(chunks, lengths_before, slot_lists)):
        n = len(chunk)
        tokens[row, :n] = chunk
        positions[row, :n] = np.arange(start, start + n)
        slots[row, :n] = slot
        context_lens[row] = start + n
    tables = np.zeros((b, max_blocks), np.int32)
    tables[:len(seq_ids)] = cache.block_tables(seq_ids, max_blocks)

    inputs = (tokens, positions, slots, tables, context_lens)
    if mesh is None:
        inputs = tuple(jnp.asarray(x) for x in inputs)
        step = trajectory_transformer.forward
    else:
        inputs = jax.device_put(inputs, replicated(mesh))  # One transfer.
        step = sharded_forward(mesh)
    out, cache.k, cache.v = step(params, cache.k, cache.v, *inputs, cfg)
    out = np.asarray(out)
    return [out[row, :len(chunk)] for row, chunk in enumerate(chunks)]
