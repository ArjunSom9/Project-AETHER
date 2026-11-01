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
"""Prefill pool of the disaggregated serving system (report Part VI, 6.1).

Prefill is compute bound and parallelizes wide (R. Pope et al., "Efficiently
Scaling Transformer Inference", arXiv:2211.05102), so the prefill pool runs
the model with a wide Megatron tensor-parallel layout over its own devices.

Scheduling is iteration level (Orca): every iteration packs prompt *chunks*
from the queued requests up to a fixed token budget, so a long prompt never
monopolizes the pool and newly arrived requests start within one iteration.
Splitting prefills into chunks follows SARATHI (A. Agrawal et al.,
arXiv:2308.16369). Note that the SARATHI preprint appeared on 2023-08-31,
after the project window closed; the token-budget chunking implemented here is
the same mechanism as Orca's iteration-level batching of prompt tokens.

When a prompt finishes, its key/value blocks are exported for the handoff to
the decode pool and its prefill-pool blocks are released.

Typical usage example:

  worker = PrefillWorker(params, cfg, jax.devices()[:4])
  worker.submit(seq_id=1, prompt=prompt)
  completed = worker.step()
"""

import collections
import dataclasses
from typing import Any, NamedTuple

import jax
import numpy as np

from src.distributed_infra import kv_paged_attention
from src.distributed_infra import model_runner
from src.surrogate_models import trajectory_transformer


@dataclasses.dataclass
class _PrefillJob:
    seq_id: int
    prompt: np.ndarray
    done: int = 0


class CompletedPrefill(NamedTuple):
    """State handed from the prefill pool to the decode pool.

    Attributes:
      seq_id: Sequence identifier.
      k_blocks: Keys of the sequence's blocks,
        (num_layers, n_blocks, block_size, heads, head_dim).
      v_blocks: Values, same shape.
      num_tokens: Prompt tokens stored in the blocks.
      first_prediction: Model prediction after the last prompt token, (F,).
    """
    seq_id: int
    k_blocks: jax.Array
    v_blocks: jax.Array
    num_tokens: int
    first_prediction: np.ndarray


class PrefillWorker:
    """Chunked, iteration-level prefill on a tensor-parallel device pool."""

    def __init__(self, params: Any,
                 cfg: trajectory_transformer.TrajectoryConfig,
                 devices: list[jax.Device], num_blocks: int = 512,
                 block_size: int = 16, token_budget: int = 256):
        """Initializes the worker.

        Args:
          params: Unsharded model parameters.
          cfg: Model configuration; num_heads must be divisible by the number
            of devices.
          devices: Devices of the prefill pool.
          num_blocks: Physical cache blocks on the pool.
          block_size: Token slots per block.
          token_budget: Maximum prompt tokens processed per iteration.
        """
        self.cfg = cfg
        self.mesh = model_runner.tensor_mesh(devices)
        self.params = model_runner.shard_params(params, self.mesh)
        self.cache = kv_paged_attention.PagedKVCache(
            cfg.num_layers, num_blocks, block_size, cfg.num_heads,
            cfg.head_dim, sharding=model_runner.kv_sharding(self.mesh))
        self.token_budget = token_budget
        self._queue: collections.deque[_PrefillJob] = collections.deque()

    @property
    def busy(self) -> bool:
        return bool(self._queue)

    def submit(self, seq_id: int, prompt: np.ndarray) -> None:
        """Queues a prompt of shape (T, feature_dim)."""
        self.cache.add_sequence(seq_id)
        self._queue.append(_PrefillJob(seq_id, np.asarray(prompt, np.float32)))

    def step(self) -> list[CompletedPrefill]:
        """Runs one prefill iteration within the token budget.

        Returns:
          The prefills that finished in this iteration.
        """
        budget = self.token_budget
        free_blocks = self.cache.allocator.num_free
        batch = []
        for job in self._queue:
            if budget == 0:
                break
            n = min(len(job.prompt) - job.done, budget)
            needed = self.cache.blocks_needed(job.seq_id, n)
            if needed > free_blocks:
                break
            batch.append((job, n))
            budget -= n
            free_blocks -= needed
        if not batch:
            return []

        predictions = model_runner.run_batch(
            self.params, self.cache, self.cfg, [job.seq_id for job, _ in batch],
            [job.prompt[job.done:job.done + n] for job, n in batch],
            self.mesh)

        completed = []
        for (job, n), pred in zip(batch, predictions):
            job.done += n
            if job.done == len(job.prompt):
                k_blocks, v_blocks = self.cache.export_blocks(job.seq_id)
                completed.append(CompletedPrefill(
                    job.seq_id, k_blocks, v_blocks, len(job.prompt),
                    pred[-1].copy()))
                self.cache.free(job.seq_id)
        finished = {c.seq_id for c in completed}
        self._queue = collections.deque(
            job for job in self._queue if job.seq_id not in finished)
        return completed
