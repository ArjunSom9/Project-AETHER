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
"""Disaggregated serving, paged cache, collectives, and sharding.

conftest.py exposes the CPU as 8 XLA devices.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import shard_map

from src.distributed_infra import kv_paged_attention as kv
from src.distributed_infra import mesh_sharding_4d as m4
from src.distributed_infra import sccl_synthesizer as sccl
from src.distributed_infra import serving_scheduler
from src.physics_engine import cfd_navier_stokes as cfd
from src.surrogate_models import trajectory_transformer as tt

_P = jax.sharding.PartitionSpec


@pytest.fixture(name='devices')
def fixture_devices():
    devices = jax.devices()
    if len(devices) < 8:
        pytest.skip('needs 8 XLA devices (see conftest.py)')
    return devices


class TestPagedCache:

    def test_paged_attention_equals_contiguous_attention(self):
        h, d, bs = 2, 8, 4
        cache = kv.PagedKVCache(1, 32, bs, h, d)
        lengths = [7, 13, 2]
        k_full = jax.random.normal(jax.random.PRNGKey(0), (3, 16, h, d))
        v_full = jax.random.normal(jax.random.PRNGKey(1), (3, 16, h, d))
        for seq, n in enumerate(lengths):
            cache.add_sequence(seq)
            slots = cache.reserve(seq, n)
            cache.write(jnp.asarray(slots), k_full[None, seq, :n],
                        v_full[None, seq, :n])
        q = jax.random.normal(jax.random.PRNGKey(2), (3, 2, h, d))
        q_pos = jnp.array([[5, 6], [11, 12], [0, 1]])
        lens = jnp.asarray(lengths)
        paged = kv.paged_attention(q, cache.k[0], cache.v[0],
                                   jnp.asarray(cache.block_tables([0, 1, 2],
                                                                  4)),
                                   lens, q_pos)
        np.testing.assert_allclose(
            paged, kv.contiguous_attention(q, k_full, v_full, lens, q_pos),
            rtol=1e-5, atol=1e-5)

    def test_fork_shares_blocks_and_copies_on_write(self):
        cache = kv.PagedKVCache(1, 16, 4, 1, 2)
        cache.add_sequence(0)
        cache.reserve(0, 10)  # 3 blocks, last one half full.
        cache.fork(0, 1)
        assert all(cache.allocator.refcount(b) == 2 for b in cache.tables[0])
        free_before = cache.allocator.num_free
        cache.reserve(1, 1)
        assert cache.tables[1][:2] == cache.tables[0][:2]
        assert cache.tables[1][-1] != cache.tables[0][-1]
        assert cache.allocator.num_free == free_before - 1
        cache.free(0)
        cache.free(1)
        assert cache.allocator.num_free == 15

    def test_waste_is_limited_to_last_block(self):
        cache = kv.PagedKVCache(1, 64, 16, 1, 2)
        for seq, n in enumerate([17, 33, 100]):
            cache.add_sequence(seq)
            cache.reserve(seq, n)
        assert cache.utilization() == pytest.approx(150 / (2 + 3 + 7) / 16)

    def test_exhaustion_raises(self):
        cache = kv.PagedKVCache(1, 3, 4, 1, 2)
        cache.add_sequence(0)
        with pytest.raises(kv.CacheExhaustedError):
            cache.reserve(0, 9)


class TestServing:

    def test_disaggregated_and_monolithic_produce_the_same_rollouts(
            self, devices):
        cfg = tt.TrajectoryConfig(feature_dim=8, embed_dim=32, num_layers=2,
                                  num_heads=8)
        params = tt.init_params(jax.random.PRNGKey(0), cfg)
        requests = serving_scheduler.make_workload(
            8, cfg.feature_dim, prompt_lengths=(10, 40),
            max_new_tokens=(1, 6), mean_interarrival=0.001, seed=3)
        disaggregated = serving_scheduler.DisaggregatedServer(
            params, cfg, devices[:4], devices[4:], token_budget=32)
        monolithic = serving_scheduler.MonolithicServer(
            params, cfg, devices, max_batch=3, max_seq_len=64)
        out_d = disaggregated.serve(requests)
        out_m = monolithic.serve(requests)
        for request in requests:
            a, b = out_d[request.request_id], out_m[request.request_id]
            assert len(a.outputs) == len(b.outputs) == 1 + request.branches
            for sa, sb in zip(sorted(a.outputs), sorted(b.outputs)):
                assert a.outputs[sa].shape == (request.max_new_tokens,
                                               cfg.feature_dim)
                np.testing.assert_allclose(a.outputs[sa], b.outputs[sb],
                                           atol=5e-3)
            assert a.finish_time >= a.first_token_time >= a.arrival_time
        assert disaggregated.mean_utilization > monolithic.mean_utilization
        assert disaggregated.decode.cache.allocator.num_free == 2047


class TestSccl:

    def test_ring_allgather_needs_n_minus_1_steps(self):
        ring = sccl.ring_topology(4)
        assert sccl.synthesize(sccl.allgather_collective(4), ring, 2,
                               2) is None
        algo = sccl.synthesize(sccl.allgather_collective(4), ring, 3, 3)
        assert algo is not None and algo.num_steps == 3

    def test_synthesized_collectives_match_xla(self, devices):
        mesh = jax.sharding.Mesh(np.asarray(devices), ('r',))
        topo = sccl.dgx_cluster(2, 4)
        sketch = sccl.rail_sketch(topo, 4)
        (algo, *_) = sccl.pareto_frontier(sccl.allgather_collective(8), topo,
                                          max_steps=3, sketch=sketch)

        def run(fn, x):
            return jax.jit(shard_map.shard_map(
                fn, mesh, in_specs=_P('r'), out_specs=_P('r')))(x)

        x = jax.random.normal(jax.random.PRNGKey(0), (8, 6))
        gathered = run(lambda v: sccl.allgather(v, algo, 'r'), x)
        np.testing.assert_allclose(gathered.reshape(8, 8, 6),
                                   np.broadcast_to(x, (8, 8, 6)))
        y = jax.random.normal(jax.random.PRNGKey(1), (8, 16))
        reduced = run(lambda v: sccl.allreduce(v[0], algo, 'r')[None], y)
        np.testing.assert_allclose(reduced,
                                   np.broadcast_to(y.sum(0), (8, 16)),
                                   rtol=1e-5, atol=1e-5)

    def test_synthesized_alltoall_matches_xla(self, devices):
        mesh = jax.sharding.Mesh(np.asarray(devices[:4]), ('r',))
        topo = sccl.dgx_cluster(2, 2)
        algo = sccl.synthesize(sccl.alltoall_collective(4), topo, 2, 3,
                               sccl.rail_sketch(topo, 2))
        assert algo is not None
        x = jax.random.normal(jax.random.PRNGKey(0), (16, 3))

        def run(fn):
            return jax.jit(shard_map.shard_map(
                fn, mesh, in_specs=_P('r'), out_specs=_P('r')))(x)

        np.testing.assert_allclose(
            run(lambda v: sccl.alltoall(v, algo, 'r')),
            run(lambda v: jax.lax.all_to_all(v, 'r', 0, 0, tiled=True)))

    def test_selection_prefers_fewer_steps_for_small_messages(self):
        ring = sccl.synthesize(sccl.allgather_collective(4),
                               sccl.ring_topology(4), 3, 3)
        topo = sccl.dgx_cluster(1, 4, nvlink_capacity=1)
        direct = sccl.synthesize(sccl.allgather_collective(4), topo, 1, 3)
        assert direct is not None
        assert sccl.select_algorithm([ring, direct], 1e3, 1e-5,
                                     1e-10) is direct


class TestMeshSharding:

    def test_hilbert_curve_is_bijective_and_local(self):
        order = 4
        n = 1 << order
        ix, iy = np.meshgrid(np.arange(n), np.arange(n))
        d = np.asarray(m4.hilbert_index(jnp.asarray(ix.ravel()),
                                        jnp.asarray(iy.ravel()), order))
        assert len(set(d.tolist())) == n * n
        walk = np.stack([ix.ravel(), iy.ravel()], 1)[np.argsort(d)]
        assert np.all(np.abs(np.diff(walk, axis=0)).sum(1) == 1)

    def test_tensor_parallel_mlp_matches_reference(self, devices):
        mesh = m4.make_mesh({'data': 2, 'tensor': 2, 'domain': 2}, devices)
        params = m4.init_mlp(jax.random.PRNGKey(0), 16, 64)
        x = jax.random.normal(jax.random.PRNGKey(1), (8, 16))
        reference = m4.mlp(params, x)
        np.testing.assert_allclose(m4.megatron_mlp_gspmd(params, x, mesh),
                                   reference, rtol=1e-5, atol=1e-5)
        algo = sccl.synthesize(sccl.allgather_collective(2),
                               sccl.dgx_cluster(1, 2), 1, 1)
        np.testing.assert_allclose(
            m4.megatron_mlp_shard_map(params, x, mesh,
                                      sccl.make_allreduce(algo)),
            reference, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('bc', [
        cfd.BoundaryConditions(),
        cfd.BoundaryConditions.hypersonic_channel((1.0, 2.0, 0.0, 9.0))
    ])
    def test_halo_exchange_matches_single_device(self, devices, bc):
        mesh = m4.make_mesh({'domain': 4}, devices[:4])
        q = jnp.abs(jax.random.normal(jax.random.PRNGKey(2),
                                      (4, 8, 32))) + 1.0
        q = q.at[3].add(5.0)
        reference = cfd.evaluate_rhs(q, 0.1, 0.1, 1e-3, bc)
        np.testing.assert_allclose(
            m4.distributed_cfd_rhs(q, 0.1, 0.1, 1e-3, bc, mesh), reference,
            rtol=1e-4, atol=1e-4 * float(jnp.abs(reference).max()))

    def test_pipeline_matches_sequential(self, devices):
        mesh = m4.make_mesh({'pipe': 4}, devices[:4])
        weights = jax.random.normal(jax.random.PRNGKey(3), (4, 8, 8)) / 3
        micro = jax.random.normal(jax.random.PRNGKey(4), (6, 2, 8))

        def stage(w, h):
            return jnp.tanh(h @ w)

        expected = micro
        for s in range(4):
            expected = stage(weights[s], expected)
        np.testing.assert_allclose(
            m4.pipeline_forward(weights, stage, micro, mesh), expected,
            rtol=1e-6, atol=1e-6)
