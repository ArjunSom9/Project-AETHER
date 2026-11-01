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
"""Five-species air chemistry with Park's two-temperature model in JAX.

Species are (N2, O2, NO, N, O). Forward rate coefficients follow
C. Park, "Review of chemical-kinetic problems of future NASA missions, I:
Earth entries", J. Thermophysics and Heat Transfer 7(3), 1993:

  k_f = A T_c^n exp(-theta / T_c),

with the rate-controlling temperature T_c = sqrt(T * T_v) for the three
dissociation reactions and T_c = T for the two Zeldovich exchanges. Third
bodies have species-dependent efficiencies (molecules vs. atoms).

Only forward rates are modeled. That is appropriate for the
dissociation-dominated relaxation zone just behind a strong bow shock;
recombination (backward rates from equilibrium constants) is out of scope.

Each reaction conserves N and O atoms, and `implicit_step` integrates the
stiff system with backward Euler, using `jax.jacfwd` for the exact Jacobian.
Newton's method on that system preserves elemental composition to round-off.

Typical usage example:

  c_next = chemistry_kinetics.implicit_step(c, T, Tv, dt=1e-6)
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

SPECIES = ('N2', 'O2', 'NO', 'N', 'O')
MOLAR_MASS = np.array([28.0134e-3, 31.9988e-3, 30.0061e-3, 14.0067e-3,
                       15.9994e-3])  # [kg/mol]

# cm^3/(mol s) -> m^3/(mol s).
_CM3_TO_M3 = 1e-6

# Pre-exponential factors of the third-body reactions, indexed by the
# collision partner M in SPECIES order (Park 1993, Table 2).
_A_N2_DISSOCIATION = np.array([7.0e21, 7.0e21, 7.0e21, 3.0e22, 3.0e22])
_A_O2_DISSOCIATION = np.array([2.0e21, 2.0e21, 2.0e21, 1.0e22, 1.0e22])
_A_NO_DISSOCIATION = np.array([5.0e15, 5.0e15, 1.1e17, 1.1e17, 1.1e17])

# (n, theta [K]) of the five reactions.
_N2_DISSOCIATION = (-1.6, 113200.0)
_O2_DISSOCIATION = (-1.5, 59500.0)
_NO_DISSOCIATION = (0.0, 75500.0)
_ZELDOVICH_1 = (6.4e17, -1.0, 38400.0)  # N2 + O -> NO + N
_ZELDOVICH_2 = (8.4e12, 0.0, 19450.0)  # NO + O -> N + O2

# Stoichiometric matrix: rows are species, columns are reactions
# (N2 diss., O2 diss., NO diss., Zeldovich 1, Zeldovich 2).
STOICHIOMETRY = np.array([
    [-1, 0, 0, -1, 0],  # N2
    [0, -1, 0, 0, 1],  # O2
    [0, 0, -1, 1, -1],  # NO
    [2, 0, 1, 1, 1],  # N
    [0, 2, 1, -1, -1],  # O
], dtype=np.float32)

# Atoms of (N, O) in each species; ELEMENTS @ STOICHIOMETRY == 0.
ELEMENTS = np.array([[2, 0, 1, 1, 0], [0, 2, 1, 0, 1]], dtype=np.float32)

_MIN_TEMPERATURE = 50.0


def arrhenius(T_c: jax.Array, A: float | jax.Array, n: float,
              theta: float) -> jax.Array:
    """Modified Arrhenius rate A T_c^n exp(-theta / T_c) in m^3/(mol s)."""
    T_c = jnp.maximum(T_c, _MIN_TEMPERATURE)
    return _CM3_TO_M3 * A * jnp.exp(n * jnp.log(T_c) - theta / T_c)


def production_rates(c: jax.Array, T: jax.Array, Tv: jax.Array) -> jax.Array:
    """Net molar production rates omega_dot [mol/(m^3 s)].

    Args:
      c: Molar concentrations [mol/m^3], shape (..., 5).
      T: Translational-rotational temperature [K], shape (...).
      Tv: Vibrational-electronic temperature [K], shape (...).

    Returns:
      omega_dot with the shape of `c`.
    """
    T_a = jnp.sqrt(T * Tv)[..., None]  # Park's dissociation temperature.
    n2, o2, no, _, o = (c[..., i] for i in range(5))

    def third_body(A_m, n, theta):
        # sum over M of k_M [M]; A_m broadcasts over the species axis.
        return jnp.sum(arrhenius(T_a, jnp.asarray(A_m, c.dtype), n, theta) * c,
                       axis=-1)

    r = jnp.stack([
        third_body(_A_N2_DISSOCIATION, *_N2_DISSOCIATION) * n2,
        third_body(_A_O2_DISSOCIATION, *_O2_DISSOCIATION) * o2,
        third_body(_A_NO_DISSOCIATION, *_NO_DISSOCIATION) * no,
        arrhenius(T, *_ZELDOVICH_1) * n2 * o,
        arrhenius(T, *_ZELDOVICH_2) * no * o,
    ], axis=-1)
    return r @ jnp.asarray(STOICHIOMETRY.T, c.dtype)


def _implicit_cell(c_old: jax.Array, T: jax.Array, Tv: jax.Array, dt: float,
                   newton_iterations: int) -> jax.Array:
    """Backward Euler with Newton iterations for one cell (c of shape (5,))."""
    eye = jnp.eye(c_old.shape[-1], dtype=c_old.dtype)
    rates = functools.partial(production_rates, T=T, Tv=Tv)
    jac_fn = jax.jacfwd(rates)

    def newton(_, c):
        residual = c - c_old - dt * rates(c)
        jacobian = eye - dt * jac_fn(c)
        return c - jnp.linalg.solve(jacobian, residual)

    return jax.lax.fori_loop(0, newton_iterations, newton, c_old)


@functools.partial(jax.jit, static_argnames=('dt', 'newton_iterations'))
def implicit_step(c: jax.Array, T: jax.Array, Tv: jax.Array, dt: float,
                  newton_iterations: int = 4) -> jax.Array:
    """Advances the concentrations by one backward-Euler step.

    Args:
      c: Molar concentrations [mol/m^3], shape (..., 5).
      T: Translational-rotational temperature [K], shape (...).
      Tv: Vibrational-electronic temperature [K], shape (...).
      dt: Time step [s].
      newton_iterations: Newton iterations per step.

    Returns:
      Concentrations after the step, same shape as `c`.
    """
    batch_shape = c.shape[:-1]
    flat = functools.partial(_implicit_cell, dt=dt,
                             newton_iterations=newton_iterations)
    out = jax.vmap(flat)(c.reshape(-1, 5), T.reshape(-1), Tv.reshape(-1))
    return out.reshape(batch_shape + (5,))


def concentrations_from_mass_fractions(rho: jax.Array,
                                       y: jax.Array) -> jax.Array:
    """Molar concentrations [mol/m^3] from density and mass fractions."""
    return rho[..., None] * y / jnp.asarray(MOLAR_MASS, y.dtype)


def element_totals(c: jax.Array) -> jax.Array:
    """Moles of (N, O) atoms per m^3, shape (..., 2)."""
    return c @ jnp.asarray(ELEMENTS.T, c.dtype)
