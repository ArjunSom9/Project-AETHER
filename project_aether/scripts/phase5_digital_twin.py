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
"""Phase 5 (weeks 9-12): disaggregated serving of the digital twin (Part VI).

1. Serves the same digital-twin query mix (trajectory rollouts, some with
   what-if branches) with the monolithic baseline and with the disaggregated
   system, after a warm-up pass that fills the compilation caches, and
   reports measured latency and cache utilization (report Table 6).
2. Tracks debris in the hypervelocity-impact clip with transformer track
   queries and reports the cost of one incremental track-query step.

On a CPU host the device pools are virtual devices sharing the same cores, so
the latencies exercise the mechanics rather than reproduce H100 numbers.

Usage (from project_aether/):
  XLA_FLAGS=--xla_force_host_platform_device_count=8 \
      python -m scripts.phase5_digital_twin
"""

import argparse
import pathlib
import time

import jax
import numpy as np
import tabulate

from src.distributed_infra import serving_scheduler
from src.surrogate_models import mot_trackformer
from src.surrogate_models import trajectory_transformer as tt

_CLIP = (pathlib.Path(__file__).resolve().parents[1] / 'data' /
         'sample_clip')


def serving_report(args) -> None:
    """Serves one workload with both designs and prints Table 6 metrics."""
    devices = jax.devices()
    if len(devices) < 2:
        raise SystemExit('Needs at least 2 devices (on CPU set XLA_FLAGS='
                         '--xla_force_host_platform_device_count=8).')
    half = len(devices) // 2
    cfg = tt.TrajectoryConfig(num_heads=8)
    params = tt.init_params(jax.random.PRNGKey(0), cfg)
    requests = serving_scheduler.make_workload(
        args.requests, cfg.feature_dim,
        mean_interarrival=args.interarrival, seed=1)

    def make_servers():
        return (serving_scheduler.MonolithicServer(params, cfg, devices,
                                                   max_batch=8),
                serving_scheduler.DisaggregatedServer(
                    params, cfg, devices[:half], devices[half:]))

    # Warm-up. Batch composition depends on the measured iteration times, so
    # a second pass can still meet a new bucket shape; after two passes the
    # replay runs without compilation.
    for _ in range(args.warmup_passes):
        for server in make_servers():
            server.serve(requests)
    monolithic, disaggregated = make_servers()
    stats_m = serving_scheduler.summarize(monolithic.serve(requests))
    stats_d = serving_scheduler.summarize(disaggregated.serve(requests))

    def row(name, stats, util):
        return [name, f'{stats["mean_latency"] * 1e3:.1f}',
                f'{stats["p90_latency"] * 1e3:.1f}',
                f'{stats["mean_ttft"] * 1e3:.1f}', f'{util:.0%}']

    print(f'{args.requests} requests, {sum(r.branches for r in requests)} '
          f'what-if branches, {len(devices)} {jax.default_backend()} devices')
    print(tabulate.tabulate(
        [row('Monolithic (request-level, contiguous KV)', stats_m,
             monolithic.mean_utilization),
         row('Disaggregated (chunked prefill, paged KV)', stats_d,
             disaggregated.mean_utilization)],
        headers=['Design', 'Mean latency ms', 'p90 ms', 'Mean TTFT ms',
                 'KV utilization'], tablefmt='github'))
    print(f'Measured mean-latency ratio (monolithic / disaggregated): '
          f'{stats_m["mean_latency"] / stats_d["mean_latency"]:.2f}x\n')


def tracking_report() -> None:
    """Tracks debris in the sample clip and times one track-query step."""
    frames, _ = mot_trackformer.load_clip(_CLIP)
    cfg = mot_trackformer.TrackerConfig()
    params = mot_trackformer.init_params(jax.random.PRNGKey(0), cfg)
    state = mot_trackformer.init_track_state(cfg)
    background = frames.max(axis=0)
    detections = [mot_trackformer.detect_debris(f, background, cfg)
                  for f in frames]
    mot_trackformer.tracker_step(params, state, *detections[0], cfg)
    step_times = []
    for feats, pos, mask in detections:
        start = time.perf_counter()
        state, out = mot_trackformer.tracker_step(params, state, feats, pos,
                                                  mask, cfg)
        out.scores.block_until_ready()
        step_times.append(time.perf_counter() - start)
    counts = [int(mask.sum()) for _, _, mask in detections]
    print(f'Debris tracking over {len(frames)} frames: detections per frame '
          f'{min(counts)}-{max(counts)}; one incremental track-query step '
          f'takes {np.median(step_times) * 1e3:.2f} ms (median). The '
          'tracker is untrained here, so identities are illustrative only.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--interarrival', type=float, default=0.02)
    parser.add_argument('--warmup-passes', type=int, default=2)
    args = parser.parse_args()
    serving_report(args)
    tracking_report()


if __name__ == '__main__':
    main()
