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
"""Decode pool of the disaggregated serving system (report Part VI, 6.1-6.2).

Decode is memory-bandwidth bound: every step reads the whole key/value
history to produce one token per sequence, so throughput comes from large,
long-lived batches. The decode pool therefore:

  * receives prefilled state from the prefill pool as whole cache blocks
    (`admit`), transferring them onto its own devices,
  * keeps that state in a PagedAttention cache, where waste is limited to each
    sequence's partially filled last block, so many sequences fit at once,
  * forks what-if branches by sharing blocks (copy-on-write), so a branch
    costs a block-table copy rather than a state copy,
  * runs continuous (iteration-level) batching: sequences join and leave the
    batch at every step.

Typical usage example:

  worker = DecodeWorker(params, cfg, jax.devices()[4:])
  worker.admit(completed_prefill, request_id=0, max_new_tokens=32)
  finished = worker.step()
"""

import collections
import dataclasses
from typing import Any

import jax
import numpy as np

from src.distributed_infra import disaggregated_prefill
from src.distributed_infra import kv_paged_attention
from src.distributed_infra import model_runner
from src.surrogate_models import trajectory_transformer


@dataclasses.dataclass
class DecodeSequence:
    """A sequence being rolled forward on the decode pool.

    Attributes:
      seq_id: Cache sequence identifier.
      request_id: Request the sequence belongs to.
      next_input: Token fed to the next decode step, (F,).
      remaining: Tokens still to generate.
      outputs: Generated tokens so far.
    """
    seq_id: int
    request_id: int
    next_input: np.ndarray
    remaining: int
    outputs: list[np.ndarray] = dataclasses.field(default_factory=list)


def branch_input(token: np.ndarray, branch_index: int) -> np.ndarray:
    """Deterministically perturbed starting token of a what-if branch."""
    return token * (1.0 + 0.01 * (branch_index + 1))


class DecodeWorker:
    """Continuous-batching decode on its own device pool."""

    def __init__(self, params: Any,
                 cfg: trajectory_transformer.TrajectoryConfig,
                 devices: list[jax.Device], num_blocks: int = 1024,
                 block_size: int = 16, max_batch: int = 64):
        """Initializes the worker.

        Args:
          params: Unsharded model parameters.
          cfg: Model configuration.
          devices: Devices of the decode pool.
          num_blocks: Physical cache blocks on the pool.
          block_size: Token slots per block; must match the prefill pool.
          max_batch: Maximum sequences per decode iteration.
        """
        self.cfg = cfg
        self.mesh = model_runner.tensor_mesh(devices)
        self.params = model_runner.shard_params(params, self.mesh)
        self._kv_sharding = model_runner.kv_sharding(self.mesh)
        self.cache = kv_paged_attention.PagedKVCache(
            cfg.num_layers, num_blocks, block_size, cfg.num_heads,
            cfg.head_dim, sharding=self._kv_sharding)
        self.max_batch = max_batch
        self._active: collections.deque[DecodeSequence] = collections.deque()

    @property
    def busy(self) -> bool:
        return bool(self._active)

    def can_admit(self, done: disaggregated_prefill.CompletedPrefill) -> bool:
        return done.k_blocks.shape[1] <= self.cache.allocator.num_free

    def admit(self, done: disaggregated_prefill.CompletedPrefill,
              request_id: int, max_new_tokens: int,
              branch_ids: tuple[int, ...] = ()) -> list[DecodeSequence]:
        """Takes over a prefilled sequence (and optional what-if branches).

        Args:
          done: Prefill result, still resident on the prefill pool.
          request_id: Request the sequence belongs to.
          max_new_tokens: Tokens to generate, including the first prediction.
          branch_ids: Sequence ids of extra branches forked from the same
            prefilled state.

        Returns:
          The new decode sequences (main sequence first).
        """
        # The handoff: move the blocks onto the decode pool's devices.
        k_blocks = jax.device_put(done.k_blocks, self._kv_sharding)
        v_blocks = jax.device_put(done.v_blocks, self._kv_sharding)
        self.cache.import_blocks(done.seq_id, k_blocks, v_blocks,
                                 done.num_tokens)
        first = done.first_prediction
        sequences = [DecodeSequence(done.seq_id, request_id, first,
                                    max_new_tokens - 1, [first])]
        for i, branch_id in enumerate(branch_ids):
            self.cache.fork(done.seq_id, branch_id)
            start = branch_input(first, i)
            sequences.append(DecodeSequence(branch_id, request_id, start,
                                            max_new_tokens - 1, [start]))
        for seq in sequences:
            if seq.remaining > 0:
                self._active.append(seq)
            else:
                self.cache.free(seq.seq_id)
        return sequences

    def step(self) -> list[DecodeSequence]:
        """Runs one decode iteration over up to `max_batch` sequences.

        Returns:
          Sequences that finished in this iteration.
        """
        if not self._active:
            return []
        batch = []
        while self._active and len(batch) < self.max_batch:
            seq = self._active.popleft()
            if self.cache.blocks_needed(seq.seq_id, 1) > (
                    self.cache.allocator.num_free):
                self._active.appendleft(seq)
                break
            batch.append(seq)
        if not batch:
            raise kv_paged_attention.CacheExhaustedError(
                'Decode pool cannot make progress: no free blocks.')
        predictions = model_runner.run_batch(
            self.params, self.cache, self.cfg, [s.seq_id for s in batch],
            [s.next_input[None] for s in batch], self.mesh)
        finished = []
        for seq, pred in zip(batch, predictions):
            seq.next_input = pred[0]
            seq.outputs.append(pred[0])
            seq.remaining -= 1
            if seq.remaining == 0:
                self.cache.free(seq.seq_id)
                finished.append(seq)
            else:
                self._active.append(seq)
        return finished
