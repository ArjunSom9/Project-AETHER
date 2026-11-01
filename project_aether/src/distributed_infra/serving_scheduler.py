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
"""Request scheduling: disaggregated serving vs. the monolithic baseline.

Report Part VI compares two serving designs on the digital-twin query mix
(Table 6):

  * `MonolithicServer`: prefill and decode share one device pool and one
    batch; request-level scheduling (a batch runs to completion before the
    next starts, so short requests wait behind long ones); every request
    reserves cache for its maximum length up front.
  * `DisaggregatedServer`: separate prefill and decode pools, iteration-level
    scheduling with chunked prefill, a PagedAttention cache, and block
    handoff between pools.

Both servers execute the real model and measure the wall time of every
iteration. Latencies are then obtained by replaying the arrival process on a
virtual clock that advances by the measured iteration times. The prefill and
decode pools occupy disjoint devices, so in the disaggregated replay one
iteration of each pool advances the clock by the slower of the two (plus the
measured handoff time). On a CPU host all virtual devices share the same
cores, so absolute numbers there only exercise the mechanics.

Typical usage example:

  requests = make_workload(num_requests=32, feature_dim=cfg.feature_dim)
  records = DisaggregatedServer(params, cfg, pre, dec).serve(requests)
  print(summarize(records))
"""

import dataclasses
import time
from typing import Any, NamedTuple

import jax
import numpy as np

from src.distributed_infra import disaggregated_decode
from src.distributed_infra import disaggregated_prefill
from src.distributed_infra import kv_paged_attention
from src.distributed_infra import model_runner
from src.surrogate_models import trajectory_transformer


class Request(NamedTuple):
    """A digital-twin query.

    Attributes:
      request_id: Unique identifier.
      arrival_time: Arrival on the virtual clock [s].
      prompt: Measured history to condition on, (T, feature_dim).
      max_new_tokens: Tokens to roll forward per branch.
      branches: Extra what-if branches forked from the same state.
    """
    request_id: int
    arrival_time: float
    prompt: np.ndarray
    max_new_tokens: int
    branches: int = 0


@dataclasses.dataclass
class RequestRecord:
    """Timeline and outputs of one served request."""
    arrival_time: float
    first_token_time: float = float('nan')
    finish_time: float = float('nan')
    outputs: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)

    @property
    def latency(self) -> float:
        return self.finish_time - self.arrival_time


def make_workload(num_requests: int, feature_dim: int,
                  prompt_lengths: tuple[int, int] = (32, 256),
                  max_new_tokens: tuple[int, int] = (8, 48),
                  mean_interarrival: float = 0.01,
                  branch_fraction: float = 0.25,
                  max_branches: int = 3, seed: int = 0) -> list[Request]:
    """Synthetic telemetry requests with Poisson arrivals.

    Prompts are noisy multi-frequency oscillations (probe readings during a
    trajectory segment); lengths and generation budgets are uniform, and a
    fraction of the requests asks for what-if branch rollouts.
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    requests = []
    freq = rng.uniform(0.01, 0.2, size=(feature_dim,))
    for request_id in range(num_requests):
        t += rng.exponential(mean_interarrival)
        length = int(rng.integers(*prompt_lengths, endpoint=True))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(feature_dim,))
        steps = np.arange(length)[:, None]
        prompt = (np.sin(freq * steps + phase) +
                  0.05 * rng.standard_normal((length, feature_dim)))
        branches = (int(rng.integers(1, max_branches, endpoint=True))
                    if rng.random() < branch_fraction else 0)
        requests.append(Request(
            request_id, t, prompt.astype(np.float32),
            int(rng.integers(*max_new_tokens, endpoint=True)), branches))
    return requests


def summarize(records: dict[int, RequestRecord]) -> dict[str, float]:
    """Latency statistics of a served workload [s]."""
    latency = np.array([r.latency for r in records.values()])
    ttft = np.array([r.first_token_time - r.arrival_time
                     for r in records.values()])
    makespan = max(r.finish_time for r in records.values())
    return {'mean_latency': float(latency.mean()),
            'p50_latency': float(np.percentile(latency, 50)),
            'p90_latency': float(np.percentile(latency, 90)),
            'mean_ttft': float(ttft.mean()),
            'makespan': float(makespan)}


class _SeqIds:
    """Hands out sequence ids for requests and their branches."""

    def __init__(self):
        self._next = 1

    def take(self, n: int) -> list[int]:
        ids = list(range(self._next, self._next + n))
        self._next += n
        return ids


def _timed(fn, *args) -> tuple[Any, float]:
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


class DisaggregatedServer:
    """Separate prefill and decode pools with block handoff (report 6.1)."""

    def __init__(self, params: Any,
                 cfg: trajectory_transformer.TrajectoryConfig,
                 prefill_devices: list[jax.Device],
                 decode_devices: list[jax.Device], block_size: int = 16,
                 prefill_blocks: int = 512, decode_blocks: int = 2048,
                 token_budget: int = 256, max_decode_batch: int = 64):
        self.prefill = disaggregated_prefill.PrefillWorker(
            params, cfg, prefill_devices, prefill_blocks, block_size,
            token_budget)
        self.decode = disaggregated_decode.DecodeWorker(
            params, cfg, decode_devices, decode_blocks, block_size,
            max_decode_batch)
        self.utilization_samples: list[float] = []

    @property
    def mean_utilization(self) -> float:
        """Average filled fraction of the decode pool's allocated slots."""
        return float(np.mean(self.utilization_samples))

    def serve(self, requests: list[Request]) -> dict[int, RequestRecord]:
        """Serves `requests` and returns their timelines and outputs."""
        pending = sorted(requests, key=lambda r: r.arrival_time)
        records = {r.request_id: RequestRecord(r.arrival_time)
                   for r in requests}
        by_seq: dict[int, tuple[Request, tuple[int, ...]]] = {}
        open_sequences: dict[int, int] = {}
        handoffs: list[disaggregated_prefill.CompletedPrefill] = []
        ids = _SeqIds()
        now, cursor, done = 0.0, 0, 0
        while done < len(requests):
            while cursor < len(pending) and pending[cursor].arrival_time <= now:
                req = pending[cursor]
                main, *branches = ids.take(1 + req.branches)
                by_seq[main] = (req, tuple(branches))
                open_sequences[req.request_id] = 1 + req.branches
                self.prefill.submit(main, req.prompt)
                cursor += 1
            if not (self.prefill.busy or self.decode.busy or handoffs):
                now = pending[cursor].arrival_time
                continue

            completed, t_prefill = _timed(self.prefill.step)
            finished, t_decode = _timed(self.decode.step)
            handoffs.extend(completed)
            admitted, t_handoff = [], 0.0
            while handoffs and self.decode.can_admit(handoffs[0]):
                item = handoffs.pop(0)
                req, branch_ids = by_seq[item.seq_id]
                sequences, dt = _timed(self.decode.admit, item,
                                       req.request_id, req.max_new_tokens,
                                       branch_ids)
                admitted.append(req)
                # Sequences that need no decode step finish at admission.
                finished.extend(s for s in sequences if s.remaining == 0)
                t_handoff += dt
            now += max(t_prefill, t_decode) + t_handoff
            if self.decode.busy:
                self.utilization_samples.append(
                    self.decode.cache.utilization())

            for req in admitted:
                records[req.request_id].first_token_time = now
            for seq in finished:
                rec = records[seq.request_id]
                rec.outputs[seq.seq_id] = np.stack(seq.outputs)
                open_sequences[seq.request_id] -= 1
                if open_sequences[seq.request_id] == 0:
                    rec.finish_time = now
                    done += 1
        return records


class MonolithicServer:
    """Co-located prefill + decode with request-level scheduling (baseline)."""

    def __init__(self, params: Any,
                 cfg: trajectory_transformer.TrajectoryConfig,
                 devices: list[jax.Device], block_size: int = 16,
                 num_blocks: int = 2560, max_batch: int = 16,
                 max_seq_len: int = 512):
        """Initializes the baseline server.

        Args:
          params: Unsharded model parameters.
          cfg: Model configuration.
          devices: The single, shared device pool.
          block_size: Token slots per cache block.
          num_blocks: Cache blocks on the pool.
          max_batch: Requests per (request-level) batch.
          max_seq_len: Length every sequence reserves up front, as in serving
            systems with contiguous per-request KV buffers.
        """
        self.cfg = cfg
        self.mesh = model_runner.tensor_mesh(devices)
        self.params = model_runner.shard_params(params, self.mesh)
        self.cache = kv_paged_attention.PagedKVCache(
            cfg.num_layers, num_blocks, block_size, cfg.num_heads,
            cfg.head_dim, sharding=model_runner.kv_sharding(self.mesh))
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self._blocks_per_seq = -(-max_seq_len // block_size)
        self.utilization_samples: list[float] = []

    @property
    def mean_utilization(self) -> float:
        """Average filled fraction of the reserved cache slots."""
        return float(np.mean(self.utilization_samples))

    def _run(self, seq_ids, chunks):
        return model_runner.run_batch(self.params, self.cache, self.cfg,
                                      seq_ids, chunks, self.mesh)

    def serve(self, requests: list[Request]) -> dict[int, RequestRecord]:
        """Serves `requests` and returns their timelines and outputs."""
        pending = sorted(requests, key=lambda r: r.arrival_time)
        records = {r.request_id: RequestRecord(r.arrival_time)
                   for r in requests}
        ids = _SeqIds()
        now, cursor = 0.0, 0
        while cursor < len(pending):
            if pending[cursor].arrival_time > now:
                now = pending[cursor].arrival_time
            batch, free = [], self.cache.allocator.num_free
            while (cursor < len(pending) and len(batch) < self.max_batch and
                   pending[cursor].arrival_time <= now):
                need = (1 + pending[cursor].branches) * self._blocks_per_seq
                if need > free:
                    break
                batch.append(pending[cursor])
                free -= need
                cursor += 1
            if not batch:
                raise kv_paged_attention.CacheExhaustedError(
                    'A single request exceeds the reserved cache capacity.')
            now = self._serve_batch(batch, records, ids, now)
        return records

    def _serve_batch(self, batch: list[Request],
                     records: dict[int, RequestRecord], ids: _SeqIds,
                     now: float) -> float:
        seqs = []  # (request, seq_id, next_input, remaining, outputs)
        main_ids = []
        for req in batch:
            main, *branches = ids.take(1 + req.branches)
            for seq_id in [main] + branches:
                self.cache.add_sequence(seq_id)
                # Contiguous reservation of the maximum sequence length.
                self.cache.preallocate(seq_id, self.max_seq_len)
            main_ids.append((req, main, branches))

        # Prefill: whole prompts in one batch (no chunking); branches share
        # nothing, so each branch re-runs the prompt.
        prefill_ids, prompts = [], []
        for req, main, branches in main_ids:
            for seq_id in [main] + branches:
                prefill_ids.append(seq_id)
                prompts.append(req.prompt)
        preds, dt = _timed(self._run, prefill_ids, prompts)
        now += dt
        self.utilization_samples.append(self.cache.utilization())
        last_prediction = {seq_id: pred[-1].copy()
                           for seq_id, pred in zip(prefill_ids, preds)}
        for req, main, branches in main_ids:
            first = last_prediction[main]
            records[req.request_id].first_token_time = now
            seqs.append([req, main, first, req.max_new_tokens - 1, [first]])
            for i, branch in enumerate(branches):
                start = disaggregated_decode.branch_input(first, i)
                seqs.append([req, branch, start, req.max_new_tokens - 1,
                             [start]])

        # Decode until every sequence of the batch is done.
        while any(s[3] > 0 for s in seqs):
            live = [s for s in seqs if s[3] > 0]
            preds, dt = _timed(self._run, [s[1] for s in live],
                               [s[2][None] for s in live])
            now += dt
            self.utilization_samples.append(self.cache.utilization())
            for s, pred in zip(live, preds):
                s[2] = pred[0]
                s[4].append(pred[0])
                s[3] -= 1
        for req, seq_id, _, _, outputs in seqs:
            records[req.request_id].outputs[seq_id] = np.stack(outputs)
            records[req.request_id].finish_time = now
            self.cache.free(seq_id)
        return now
