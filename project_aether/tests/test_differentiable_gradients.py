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
"""Gradients flow through every differentiable stage (report Part III, 3.1)."""

import jax
from jax import experimental
import jax.numpy as jnp
import numpy as np

from src.physics_engine import ablation_monolithic_solver
from src.physics_engine import cfd_navier_stokes as cfd
from src.physics_engine import chemistry_kinetics
from src.physics_engine import dsmc_boltzmann as dsmc
from src.surrogate_models import galerkin_transformer
from src.surrogate_models import physics_informed_loss


def _finite_nonzero(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return (all(bool(jnp.all(jnp.isfinite(x))) for x in leaves) and
            any(bool(jnp.any(x != 0)) for x in leaves))


def test_navier_stokes_rhs_matches_finite_differences():
    # The minmod limiter is non-differentiable where neighboring slopes are
    # equal or zero. Strictly monotone exponential profiles keep every cell
    # on one branch (slope ratios differ from 1), so autodiff and central
    # differences must agree; float64 removes rounding noise.
    with experimental.enable_x64():
        y, x = jnp.meshgrid(jnp.arange(8.0), jnp.arange(16.0), indexing='ij')
        rho = jnp.exp(0.20 * x + 0.15 * y)
        u = 20.0 * jnp.exp(0.10 * x + 0.05 * y)
        v = 5.0 * jnp.exp(0.07 * x + 0.12 * y)
        p = 1e5 * jnp.exp(0.05 * x + 0.03 * y)
        q = cfd.conserved_from_primitive(rho, u, v, p)

        def objective(state):
            return jnp.sum(cfd.evaluate_rhs(state, 0.1, 0.1, 1e-3)**2)

        grad = jax.grad(objective)(q)
        direction = q * jax.random.normal(jax.random.PRNGKey(1), q.shape,
                                          dtype=jnp.float64)
        eps = 1e-7
        fd = (objective(q + eps * direction) -
              objective(q - eps * direction)) / (2.0 * eps)
        np.testing.assert_allclose(float(jnp.vdot(grad, direction)),
                                   float(fd), rtol=1e-4)


def test_gradients_through_whole_simulation():
    q = cfd.conserved_from_primitive(jnp.ones((8, 16)),
                                     jnp.full((8, 16), 50.0),
                                     jnp.zeros((8, 16)),
                                     jnp.full((8, 16), 1e5))

    def final_energy(scale):
        q_end, _ = cfd.simulate(q * scale, 5, 1e-3, 1e-3, 1e-7, 1.8e-5)
        return jnp.sum(q_end[3])

    grad = jax.grad(final_energy)(jnp.float32(1.0))
    assert np.isfinite(float(grad)) and float(grad) != 0.0


def test_ablation_gradient_wrt_surface_heat_flux():
    T = jnp.full((12, 8), 300.0)
    phi = jnp.ones((12, 8))
    zeros = jnp.zeros((12, 8))

    def mean_temperature(flux):
        T_new, _ = ablation_monolithic_solver.implicit_monolithic_step(
            T, phi, zeros, zeros, dx=1e-3, dy=1e-3, dt=0.1,
            surface_heat_flux=flux)
        return jnp.mean(T_new)

    grad = jax.grad(mean_temperature)(jnp.full((8,), 1e5))
    # Heating the surface raises the mean temperature everywhere.
    assert bool(jnp.all(jnp.isfinite(grad))) and bool(jnp.all(grad > 0.0))


def test_ablation_gradient_wrt_initial_temperature():
    T = jnp.full((12, 8), 300.0).at[:2].set(3400.0)
    phi = jnp.ones((12, 8)).at[0].set(0.0)
    zeros = jnp.zeros((12, 8))
    grad = jax.grad(lambda t: jnp.mean(
        ablation_monolithic_solver.implicit_monolithic_step(
            t, phi, zeros, zeros, dx=1e-3, dy=1e-3, dt=0.1)[0]))(T)
    assert _finite_nonzero(grad)


def test_relaxed_dsmc_gradient_wrt_cross_section():
    cfg = dsmc.DsmcConfig(lo=(0.0, 0.0, 0.0), hi=(0.1, 0.1, 0.025),
                          cells=(8, 8, 2), dt=2e-7, fnum=2e14)
    state = dsmc.initialize_maxwellian(jax.random.PRNGKey(0), 20_000, cfg,
                                       2000.0, drift=(-2000.0, 0.0, 0.0))

    def x_kinetic_energy(d_ref):
        out = dsmc.relaxed_dsmc_step(state, jax.random.PRNGKey(1), cfg, d_ref)
        return jnp.mean(out.particles.vx**2)

    grad = jax.grad(x_kinetic_energy)(jnp.float32(cfg.gas.d_ref))
    assert np.isfinite(float(grad)) and float(grad) != 0.0


def test_chemistry_jacobian_is_exact():
    # Rates are quadratic in the concentrations, so central differences are
    # exact up to rounding; float64 keeps that rounding far below 1e-6.
    with experimental.enable_x64():
        c = jnp.array([20.0, 5.0, 0.5, 1.0, 2.0], jnp.float64)
        T, Tv = jnp.float64(9000.0), jnp.float64(6000.0)
        jac = jax.jacfwd(chemistry_kinetics.production_rates)(c, T, Tv)
        for i in range(5):
            dc = jnp.zeros(5, jnp.float64).at[i].set(1e-3 * c[i])
            fd = (chemistry_kinetics.production_rates(c + dc, T, Tv) -
                  chemistry_kinetics.production_rates(c - dc, T, Tv)) / (
                      2 * dc[i])
            np.testing.assert_allclose(jac[:, i], fd, rtol=1e-6)


def test_physics_informed_loss_gradients_reach_surrogate():
    cfg = galerkin_transformer.GalerkinConfig(width=16, num_layers=1, modes=2,
                                              num_heads=2)
    params = galerkin_transformer.init_params(jax.random.PRNGKey(0), cfg)
    q = jnp.stack([cfd.conserved_from_primitive(
        1.0 + 0.1 * jax.random.uniform(jax.random.PRNGKey(i), (8, 8)),
        jnp.full((8, 8), 30.0), jnp.zeros((8, 8)), jnp.full((8, 8), 1e5))
                   for i in range(2)])
    norm = galerkin_transformer.Normalizer.fit(q)
    flow = physics_informed_loss.FlowSetup(dt=1e-6, dx=1e-3, dy=1e-3,
                                           mu=1.8e-5)

    def loss(p):
        pred = galerkin_transformer.apply(p, q, norm, cfg)
        total, _ = physics_informed_loss.loss_fn(pred, q, q, norm.std, flow,
                                                 physics_weight=1.0)
        return total

    assert _finite_nonzero(jax.grad(loss)(params))
