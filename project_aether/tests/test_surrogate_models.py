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
# Test modules need no docstring (style guide 3.8.2.1).
# pylint: disable=missing-module-docstring

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.physics_engine import cfd_navier_stokes as cfd
from src.surrogate_models import block_sparse_attention as bsa
from src.surrogate_models import galerkin_transformer as gt
from src.surrogate_models import mot_trackformer
from src.surrogate_models import physics_informed_loss as pil
from src.surrogate_models import quantization_fp8 as q8

_CLIP = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'sample_clip'


def _fields(batch=2, ny=16, nx=32, seed=0):
    rho = 0.5 + jax.random.uniform(jax.random.PRNGKey(seed), (batch, ny, nx))
    u = 100.0 * jax.random.normal(jax.random.PRNGKey(seed + 1), rho.shape)
    q = jax.vmap(lambda r, vel: cfd.conserved_from_primitive(
        r, vel, jnp.zeros_like(r), jnp.full_like(r, 1e5)))(rho, u)
    return q


class TestGalerkinTransformer:

    @pytest.mark.parametrize('attention', [gt.GALERKIN, gt.FOURIER])
    def test_forward_shapes(self, attention):
        cfg = gt.GalerkinConfig(width=32, num_layers=2, modes=4,
                                attention=attention)
        params = gt.init_params(jax.random.PRNGKey(0), cfg)
        q = _fields()
        out = gt.apply(params, q, gt.Normalizer.fit(q), cfg)
        assert out.shape == q.shape
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_galerkin_attention_is_linear_in_values(self):
        k = jax.random.split(jax.random.PRNGKey(0), 3)
        q, kk, v = (jax.random.normal(ki, (1, 2, 64, 8)) for ki in k)
        # LN(V) is scale invariant, so check against an explicit reference.

        def ln(x):
            return (x - x.mean(-1, keepdims=True)) / jnp.sqrt(
                x.var(-1, keepdims=True) + 1e-5)

        expected = (q @ (jnp.swapaxes(ln(kk), -1, -2) @ ln(v))) / 64
        np.testing.assert_allclose(gt.galerkin_attention(q, kk, v), expected,
                                   rtol=1e-4, atol=1e-5)

    def test_spectral_conv_keeps_low_modes_only(self):
        params = {'re': jnp.ones((2, 1, 1, 2, 2)), 'im': jnp.zeros(
            (2, 1, 1, 2, 2))}
        _, x = jnp.meshgrid(jnp.arange(16), jnp.arange(16), indexing='ij')
        high = jnp.cos(2 * jnp.pi * 7 * x / 16)[None, :, :, None]
        out = gt.spectral_conv2d(params, high, modes=2)
        assert float(jnp.abs(out).max()) < 1e-5


class TestBlockSparseAttention:

    def test_window_gives_75_percent_sparsity(self):
        for num_blocks in (32, 64):
            window = bsa.window_for_density(num_blocks, 1, 0.25)
            density = bsa.layout_density(
                bsa.block_sparse_layout(num_blocks, window, 1))
            assert abs(density - 0.25) < 0.03

    def test_matches_dense_masked_attention(self):
        layout = bsa.block_sparse_layout(16, bsa.window_for_density(16), 1)
        k = jax.random.split(jax.random.PRNGKey(0), 3)
        q, kk, v = (jax.random.normal(ki, (2, 3, 16 * 8, 16)) for ki in k)
        np.testing.assert_allclose(
            bsa.block_sparse_attention(q, kk, v, layout, 8),
            bsa.dense_masked_attention(q, kk, v, layout, 8), rtol=1e-5,
            atol=1e-5)


class TestFp8Quantization:

    def test_weight_memory_is_half_of_bf16(self):
        params = gt.init_params(jax.random.PRNGKey(0),
                                gt.GalerkinConfig(width=64, num_layers=2))
        qparams = q8.quantize_params(params)
        assert (q8.dense_weight_bytes(qparams) /
                q8.dense_weight_bytes(params, jnp.bfloat16)) == 0.5
        dtypes = {leaf.dtype for leaf in jax.tree_util.tree_leaves(qparams)}
        assert jnp.dtype(q8.E4M3) in dtypes

    def test_roundtrip_error_is_bounded_by_format(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (4096,))
        scale = q8.scale_for(jnp.max(jnp.abs(x)))
        err = jnp.abs(q8.fake_quantize(x, scale) - x) / jnp.maximum(
            jnp.abs(x), 1e-2)
        # E4M3 has 3 mantissa bits: relative rounding error <= 2^-4.
        assert float(jnp.median(err)) < 2.0**-4

    def test_calibrated_ptq_preserves_surrogate_output(self):
        cfg = gt.GalerkinConfig(width=32, num_layers=2, modes=4)
        params = gt.init_params(jax.random.PRNGKey(1), cfg)
        q = _fields(seed=3)
        norm = gt.Normalizer.fit(q)
        calib = q8.AmaxCalibrator()
        reference = gt.apply(params, q, norm, cfg, calib=calib)
        assert 'layers/0/attn/q' in calib.amax
        quantized = gt.apply(q8.quantize_params(params, calib), q, norm, cfg)
        rel = jnp.linalg.norm(quantized - reference) / jnp.linalg.norm(
            reference)
        assert float(rel) < 0.05

    def test_fp8_matmul_uses_e5m2_gradients(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (16, 32))
        w = jax.random.normal(jax.random.PRNGKey(1), (32, 8))
        g_fp8 = jax.grad(lambda a: jnp.sum(q8.fp8_matmul(a, w)**2))(x)
        g_ref = jax.grad(lambda a: jnp.sum((a @ w)**2))(x)
        rel = jnp.linalg.norm(g_fp8 - g_ref) / jnp.linalg.norm(g_ref)
        assert 0.0 < float(rel) < 0.2


class TestPhysicsInformedLoss:

    def test_exact_solver_step_has_small_residual(self):
        flow = pil.FlowSetup(dt=2e-7, dx=1e-3, dy=1e-3, mu=1.8e-5)
        q0 = _fields(batch=1, ny=16, nx=16)[0]
        q1 = cfd.step(q0, flow.dt, flow.dx, flow.dy, flow.mu)
        exact = pil.navier_stokes_residual(q0[None], q1[None], flow)
        wrong = pil.navier_stokes_residual(q0[None], (q1 * 1.01)[None], flow)
        assert float(jnp.abs(exact).mean()) < 0.1 * float(
            jnp.abs(wrong).mean())

    def test_annealing_is_geometric(self):
        assert float(pil.annealed_physics_weight(0, 100)) == pytest.approx(1e-4)
        assert float(pil.annealed_physics_weight(100, 100)) == pytest.approx(
            1e-1, rel=1e-5)


class TestTrackFormer:

    def test_birth_keep_and_retire_lifecycle(self):
        cfg = mot_trackformer.TrackerConfig(max_tracks=4,
                                            num_detect_queries=3,
                                            max_misses=1)
        params = mot_trackformer.init_params(jax.random.PRNGKey(0), cfg)
        state = mot_trackformer.init_track_state(cfg)
        hidden = jnp.ones((7, cfg.embed_dim))
        pos = jnp.full((7, 2), 0.5)

        def frame(scores):
            return mot_trackformer.FrameOutputs(hidden, jnp.asarray(scores),
                                                pos)

        # Two confident detect queries are born as tracks 0 and 1.
        state = mot_trackformer.update_tracks(
            params, state, frame([0, 0, 0, 0, .9, .1, .8]), cfg)
        assert sorted(np.asarray(state.ids).tolist()) == [-1, -1, 0, 1]
        # Track 0 is seen, track 1 missed once (kept), then missed again.
        slot0 = int(np.argmax(np.asarray(state.ids) == 0))
        scores = np.zeros(7)
        scores[slot0] = 0.9
        state = mot_trackformer.update_tracks(params, state, frame(scores),
                                              cfg)
        assert 1 in np.asarray(state.ids).tolist()
        state = mot_trackformer.update_tracks(params, state, frame(scores),
                                              cfg)
        assert np.asarray(state.ids).tolist().count(0) == 1
        assert 1 not in np.asarray(state.ids).tolist()
        assert int(state.next_id) == 2

    def test_hungarian_matches_nearest_objects(self):
        pred = np.array([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]])
        gt_pos = np.array([[0.88, 0.92], [0.12, 0.08]])
        rows, cols = mot_trackformer.hungarian_match(pred, np.ones(3), gt_pos)
        assert dict(zip(rows.tolist(), cols.tolist())) == {0: 1, 1: 0}

    def test_loss_is_differentiable(self):
        cfg = mot_trackformer.TrackerConfig(max_tracks=8,
                                            num_detect_queries=4,
                                            max_detections=16)
        params = mot_trackformer.init_params(jax.random.PRNGKey(0), cfg)
        state = mot_trackformer.init_track_state(cfg)
        feats = jnp.ones((16, cfg.feature_dim))
        det_pos = jnp.full((16, 2), 0.3)
        mask = jnp.arange(16) < 5
        grad = jax.grad(mot_trackformer.tracking_loss)(
            params, state, feats, det_pos, mask, jnp.zeros((8, 2)),
            jnp.zeros((8,), bool), jnp.full((4, 2), 0.3),
            jnp.array([True, False, False, False]), cfg)
        leaves = jax.tree_util.tree_leaves(grad)
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)

    def test_debris_detections_grow_after_impact(self):
        frames, numbers = mot_trackformer.load_clip(_CLIP)
        assert np.all(np.diff(numbers) > 0)
        cfg = mot_trackformer.TrackerConfig()
        background = frames.max(axis=0)
        early = mot_trackformer.detect_debris(frames[2], background, cfg)[2]
        late = mot_trackformer.detect_debris(frames[-1], background, cfg)[2]
        assert int(late.sum()) > int(early.sum()) > 0
