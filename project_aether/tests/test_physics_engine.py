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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.physics_engine import ablation_monolithic_solver
from src.physics_engine import cfd_navier_stokes as cfd
from src.physics_engine import chemistry_kinetics
from src.physics_engine import dsmc_boltzmann as dsmc

_CFG = dsmc.DsmcConfig(lo=(0.0, 0.0, 0.0), hi=(0.1, 0.1, 0.025),
                       cells=(8, 8, 4), dt=2e-7, fnum=2e14)


def _momentum(p):
    return np.array([np.sum(np.asarray(f, np.float64))
                     for f in (p.vx, p.vy, p.vz)])


def _energy(p):
    return float(sum(np.sum(np.asarray(f, np.float64)**2)
                     for f in (p.vx, p.vy, p.vz)))


@pytest.fixture(name='gas')
def fixture_gas():
    return dsmc.initialize_maxwellian(jax.random.PRNGKey(0), 50_000, _CFG,
                                      2000.0, drift=(-2500.0, 0.0, 0.0))


class TestDsmc:

    def test_binning_sorts_by_cell_and_offsets_are_exact(self, gas):
        perm, sorted_cell, cell_start = dsmc.bin_particles(
            gas.particles, jax.random.PRNGKey(1), _CFG)
        cells = np.asarray(sorted_cell)
        assert np.all(np.diff(cells) >= 0)
        expected = np.asarray(dsmc.cell_ids(gas.particles, _CFG))[
            np.asarray(perm)]
        np.testing.assert_array_equal(cells, expected)
        np.testing.assert_array_equal(
            np.asarray(cell_start),
            np.searchsorted(cells, np.arange(_CFG.num_cells + 1)))

    def test_collisions_conserve_momentum_and_energy(self, gas):
        k_bin, k_col = jax.random.split(jax.random.PRNGKey(2))
        perm, sorted_cell, cell_start = dsmc.bin_particles(
            gas.particles, k_bin, _CFG)
        p = dsmc.permute(gas.particles, perm)
        after, _, candidates, collisions = dsmc.collide(
            p, sorted_cell, cell_start, gas.sigma_g_max, k_col, _CFG)
        assert int(collisions.sum()) > 0
        assert np.all(np.asarray(collisions) <= np.asarray(candidates))
        scale = np.sqrt(_energy(p))
        np.testing.assert_allclose(_momentum(after), _momentum(p),
                                   atol=1e-6 * scale)
        assert abs(_energy(after) - _energy(p)) / _energy(p) < 1e-6

    def test_fused_and_unfused_steps_agree(self, gas):
        key = jax.random.PRNGKey(3)
        fused, stats = dsmc.dsmc_step(gas, key, _CFG)
        unfused, _ = dsmc.dsmc_step_unfused(gas, key, _CFG)
        for a, b in zip(fused.particles, unfused.particles):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-3)
        assert int(stats.cell_start[-1]) == 50_000

    def test_wall_tally_counts_hits(self, gas):
        _, stats = dsmc.dsmc_step(gas, jax.random.PRNGKey(4), _CFG)
        assert int(stats.wall_hits) > 0  # The gas drifts toward x = 0.
        assert np.isfinite(float(stats.wall_heat))

    def test_fusion_reduces_memory_traffic(self, gas):
        audit = dsmc.audit_dsmc_fusion(gas, jax.random.PRNGKey(5), _CFG)
        assert audit.fused.num_fusions > 0
        assert audit.fused.bytes_accessed < audit.eager_bytes
        assert audit.reduction > 1.5

    def test_cell_moments_recover_temperature(self, gas):
        perm, sorted_cell, _ = dsmc.bin_particles(
            gas.particles, jax.random.PRNGKey(6), _CFG)
        _, temperature = dsmc.cell_moments(
            dsmc.permute(gas.particles, perm), sorted_cell, _CFG)
        assert abs(float(jnp.mean(temperature)) - 2000.0) < 100.0


class TestNavierStokes:

    def test_periodic_domain_conserves_mass_momentum_energy(self):
        key = jax.random.PRNGKey(0)
        rho = 1.0 + 0.1 * jax.random.uniform(key, (16, 32))
        q = cfd.conserved_from_primitive(rho, jnp.full_like(rho, 50.0),
                                         jnp.zeros_like(rho),
                                         jnp.full_like(rho, 1e5))
        q_end, _ = cfd.simulate(q, 20, 1e-3, 1e-3, 1e-7, 1.8e-5)
        totals = np.asarray(q.sum(axis=(1, 2)), np.float64)
        np.testing.assert_allclose(np.asarray(q_end.sum(axis=(1, 2))),
                                   totals, rtol=1e-5, atol=1e-3)

    def test_mach_10_channel_stays_finite_and_positive(self):
        rho0, temp0 = 0.01, 220.0
        p0 = rho0 * cfd.R_GAS * temp0
        u0 = 10.0 * np.sqrt(cfd.GAMMA * p0 / rho0)
        inflow = cfd.conserved_from_primitive(
            jnp.float32(rho0), jnp.float32(u0), jnp.float32(0.0),
            jnp.float32(p0))
        q = jnp.broadcast_to(inflow[:, None, None], (4, 24, 48))
        blunt = cfd.conserved_from_primitive(
            jnp.float32(5 * rho0), jnp.float32(0.0), jnp.float32(0.0),
            jnp.float32(50 * p0))
        q = q.at[:, 8:16, 20:24].set(blunt[:, None, None])
        bc = cfd.BoundaryConditions.hypersonic_channel(tuple(
            float(x) for x in inflow))
        dt = float(cfd.max_stable_dt(q, 1e-3, 1e-3, 1.8e-5))
        q_end, _ = cfd.simulate(q, 150, 1e-3, 1e-3, dt, 1.8e-5, bc)
        rho, _, _, p, _ = cfd.primitives(q_end)
        assert bool(jnp.all(jnp.isfinite(q_end)))
        assert float(rho.min()) > 0.0 and float(p.min()) > 0.0

    def test_uniform_flow_is_steady(self):
        q = cfd.conserved_from_primitive(jnp.ones((8, 8)),
                                         jnp.full((8, 8), 30.0),
                                         jnp.full((8, 8), -10.0),
                                         jnp.full((8, 8), 2e5))
        rhs = cfd.evaluate_rhs(q, 0.1, 0.1, 1e-3)
        assert float(jnp.abs(rhs).max()) < 1e-2


class TestChemistry:

    def test_elements_conserved_and_air_dissociates(self):
        rho = jnp.full((6,), 1e-3)
        y = jnp.broadcast_to(jnp.array([0.767, 0.233, 0.0, 0.0, 0.0]),
                             (6, 5))
        c0 = chemistry_kinetics.concentrations_from_mass_fractions(rho, y)
        T = jnp.linspace(5000.0, 12000.0, 6)
        c = c0
        for _ in range(20):
            c = chemistry_kinetics.implicit_step(c, T, 0.6 * T, dt=1e-6)
        np.testing.assert_allclose(chemistry_kinetics.element_totals(c),
                                   chemistry_kinetics.element_totals(c0),
                                   rtol=1e-5)
        # Atomic oxygen appears, more of it at higher temperature.
        assert float(c[-1, 4]) > float(c[0, 4]) > 0.0

    def test_stoichiometry_balances_atoms(self):
        np.testing.assert_array_equal(
            chemistry_kinetics.ELEMENTS @ chemistry_kinetics.STOICHIOMETRY, 0)


class TestAblation:

    def _setup(self):
        ny, nx = 20, 12
        T = jnp.full((ny, nx), 300.0).at[:3].set(3500.0)
        phi = jnp.ones((ny, nx)).at[:2].set(0.0)
        u = jnp.zeros((ny, nx)).at[:2].set(40.0)
        return T, phi, u, jnp.zeros((ny, nx))

    def test_solution_satisfies_maximum_principle(self):
        T, phi, u, v = self._setup()
        T_new, phi_new = ablation_monolithic_solver.implicit_monolithic_step(
            T, phi, u, v, dx=1e-3, dy=1e-3, dt=0.05)
        assert 300.0 - 1e-3 <= float(T_new.min())
        assert float(T_new.max()) <= 3500.0 + 1e-3
        assert float(phi_new.min()) >= 0.0

    def test_blowing_velocity_from_mass_loss(self):
        phi_old = jnp.ones((4, 3))
        phi_new = phi_old.at[0].set(0.5)
        v = ablation_monolithic_solver.blowing_velocity(
            phi_old, phi_new, dt=1.0, dy=1e-3)
        mat = ablation_monolithic_solver.Materials()
        np.testing.assert_allclose(v, mat.rho_solid * 0.5e-3 / mat.rho_gas,
                                   rtol=1e-6)
