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
"""Kernel fusion: XLA audits (CPU) and Triton kernel parity (CUDA only).

The Triton tests need torch 2.0.1 + triton 2.0.0 and an NVIDIA GPU; they are
skipped elsewhere.
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from src.physics_engine import cfd_navier_stokes as cfd
from src.physics_engine import xla_fusion_audit


def _cuda_available():
    try:
        import torch  # pylint: disable=import-outside-toplevel
        import triton  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
        return False
    return torch.cuda.is_available()


requires_cuda = pytest.mark.skipif(not _cuda_available(),
                                   reason='needs torch, triton and a GPU')


class TestXlaFusionAudit:

    def test_elementwise_chain_fuses_into_one_kernel(self):
        x = jnp.ones((1 << 16,))
        report = xla_fusion_audit.analyze(
            lambda a: jnp.sin(a) * 2.0 + jnp.cos(a) - 1.0, x)
        assert report.num_fusions >= 1
        eager = xla_fusion_audit.eager_bytes(
            lambda a: jnp.sin(a) * 2.0 + jnp.cos(a) - 1.0, x)
        # Five primitives eagerly, one fused read + write.
        assert eager / report.bytes_accessed >= 3.0

    def test_cfd_step_fusion_reduces_traffic(self):
        q = cfd.conserved_from_primitive(jnp.ones((32, 64)),
                                         jnp.full((32, 64), 10.0),
                                         jnp.zeros((32, 64)),
                                         jnp.full((32, 64), 1e5))
        step = functools.partial(cfd.ssp_rk3_step, dt=1e-7,
                                 rhs_fn=functools.partial(
                                     cfd.evaluate_rhs, dx=1e-3, dy=1e-3,
                                     mu=1.8e-5))
        report = xla_fusion_audit.analyze(step, q)
        eager = xla_fusion_audit.eager_bytes(step, q)
        assert eager / report.bytes_accessed > 2.0

    def test_count_kernels_parses_entry_computation(self):
        hlo = '\n'.join([
            'HloModule m', '', 'ENTRY %main (p: f32[4]) -> f32[4] {',
            '  %p = f32[4]{0} parameter(0)',
            '  %f = f32[4]{0} fusion(f32[4]{0} %p), kind=kLoop',
            '  ROOT %s = f32[4]{0} sort(f32[4]{0} %f), dimensions={0}', '}'
        ])
        assert xla_fusion_audit.count_kernels(hlo) == (1, 2)


@requires_cuda
class TestTritonKernels:

    def test_fused_particle_push_matches_reference(self):
        import torch  # pylint: disable=import-outside-toplevel
        from src.custom_kernels.triton import fused_particle_push  # pylint: disable=import-outside-toplevel
        n = 100_003
        gen = torch.Generator(device='cuda').manual_seed(0)
        pos = [torch.rand(n, device='cuda', generator=gen) for _ in range(3)]
        vel = [torch.randn(n, device='cuda', generator=gen) for _ in range(3)]
        lo, hi, dt = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.05
        ref_pos, ref_vel = fused_particle_push.reference_push(pos, vel, dt,
                                                             lo, hi)
        fused_particle_push.fused_particle_push(pos, vel, dt, lo, hi)
        for a, b in zip(pos + vel, ref_pos + ref_vel):
            torch.testing.assert_close(a, b)

    def test_specular_reflection(self):
        import torch  # pylint: disable=import-outside-toplevel
        from src.custom_kernels.triton import fused_particle_push  # pylint: disable=import-outside-toplevel
        pos = [torch.tensor([9.0], device='cuda'),
               torch.tensor([5.0], device='cuda'),
               torch.tensor([5.0], device='cuda')]
        vel = [torch.tensor([5.0], device='cuda'),
               torch.zeros(1, device='cuda'), torch.zeros(1, device='cuda')]
        fused_particle_push.fused_particle_push(pos, vel, 1.0, (0., 0., 0.),
                                                (10., 10., 10.))
        # 9 + 5 = 14 reflects to 10 - 4 = 6 with reversed velocity.
        assert pos[0].item() == pytest.approx(6.0)
        assert vel[0].item() == pytest.approx(-5.0)

    def test_fused_softmax_matches_torch(self):
        import torch  # pylint: disable=import-outside-toplevel
        from src.custom_kernels.triton import fused_softmax  # pylint: disable=import-outside-toplevel
        x = torch.randn(257, 1000, device='cuda')
        torch.testing.assert_close(fused_softmax.fused_softmax(x),
                                   torch.softmax(x, dim=-1))

    def test_bilinear_interpolation_matches_reference(self):
        import torch  # pylint: disable=import-outside-toplevel
        from src.custom_kernels.triton import pic_tiled_interpolation as interp  # pylint: disable=import-outside-toplevel
        field = torch.randn(64, 96, device='cuda')
        x = torch.rand(5000, device='cuda') * 95 * 0.1
        y = torch.rand(5000, device='cuda') * 63 * 0.1
        torch.testing.assert_close(
            interp.interpolate(field, x, y, (0.1, 0.1)),
            interp.reference_interpolate(field, x, y, (0.1, 0.1)))

    def test_paged_attention_matches_reference(self):
        import torch  # pylint: disable=import-outside-toplevel
        from src.custom_kernels.triton import paged_attention  # pylint: disable=import-outside-toplevel
        num_blocks, bs, h, d = 64, 16, 4, 64
        k = torch.randn(num_blocks, bs, h, d, device='cuda')
        v = torch.randn(num_blocks, bs, h, d, device='cuda')
        lens = torch.tensor([1, 37, 160], dtype=torch.int32, device='cuda')
        tables = torch.randperm(num_blocks, device='cuda')[:30].reshape(
            3, 10).to(torch.int32)
        q = torch.randn(3, h, d, device='cuda')
        torch.testing.assert_close(
            paged_attention.paged_attention_decode(q, k, v, tables, lens),
            paged_attention.reference_paged_attention(q, k, v, tables, lens),
            rtol=1e-4, atol=1e-4)

    def test_boris_push_matches_reference_and_conserves_energy(self):
        import torch  # pylint: disable=import-outside-toplevel
        from src.physics_engine import pic_plasma_dynamics as pic  # pylint: disable=import-outside-toplevel
        n, (ny, nx) = 4096, (32, 32)
        pos = [torch.rand(n, device='cuda') * 3.0 for _ in range(2)]
        vel = [torch.randn(n, device='cuda') * 1e5 for _ in range(3)]
        zero = torch.zeros(ny, nx, device='cuda')
        fields = [zero, zero, zero, zero, zero, torch.full_like(zero, 0.1)]
        ref_pos, ref_vel = pic.reference_boris_push(pos, vel, fields,
                                                    (0.1, 0.1), -1.76e11,
                                                    1e-12)
        speed2 = sum(v * v for v in vel)
        pic.boris_push(pos, vel, fields, (0.1, 0.1), -1.76e11, 1e-12)
        for a, b in zip(pos + vel, ref_pos + ref_vel):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-3)
        # A pure magnetic field rotates velocities without changing speed.
        torch.testing.assert_close(sum(v * v for v in vel), speed2,
                                   rtol=1e-5, atol=0.0)


def test_boris_reference_conserves_energy_on_cpu():
    torch = pytest.importorskip('torch')
    pic = pytest.importorskip('src.physics_engine.pic_plasma_dynamics')
    pos = [torch.rand(256) * 3.0 for _ in range(2)]
    vel = [torch.randn(256) * 1e5 for _ in range(3)]
    zero = torch.zeros(32, 32)
    fields = [zero, zero, zero, zero, zero, torch.full_like(zero, 0.1)]
    _, new_vel = pic.reference_boris_push(pos, vel, fields, (0.1, 0.1),
                                          -1.76e11, 1e-12)
    np.testing.assert_allclose(sum(v * v for v in new_vel).numpy(),
                               sum(v * v for v in vel).numpy(), rtol=1e-5)
