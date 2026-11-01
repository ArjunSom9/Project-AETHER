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
"""Monolithic, differentiable heat-transfer and ablation solver in JAX.

The thermal-protection-system (TPS) solid and the adjacent gas are treated as
one domain whose properties vary smoothly with the solid fraction `phi`
(1 = virgin TPS, 0 = gas). Each step solves the implicit (backward-Euler)
advection-diffusion equation

    rho_cp T_new - dt [div(k grad T_new) - rho_cp_gas u . grad T_new]
        = rho_cp T_old + dt q_ablation

on the whole domain at once instead of exchanging boundary conditions
between separate solid and fluid solvers. The upwinded advection term makes
the operator non-symmetric (so conjugate gradients does not apply), and the
linear system is solved matrix-free with GMRES. Because the operator is an
M-matrix, the solution obeys a discrete maximum principle apart from the
ablation sink. `jax.scipy.sparse.linalg.gmres` is built on
`lax.custom_linear_solve`, so gradients propagate through the implicit solve
(for example to TPS material properties).

The DSMC wall-heat tally (`dsmc_boltzmann.StepStats.wall_heat`) supplies the
surface heat flux; the blowing velocity returned here couples mass loss back
to the gas.

Typical usage example:

  T, phi = ablation_monolithic_solver.implicit_monolithic_step(
      T, phi, u, v, dx=1e-3, dy=1e-3, dt=1e-2)
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Materials(NamedTuple):
    """Thermophysical properties of the TPS solid and the boundary-layer gas.

    Attributes:
      k_solid: Solid thermal conductivity [W/(m K)].
      k_gas: Gas thermal conductivity [W/(m K)].
      rho_cp_solid: Solid volumetric heat capacity [J/(m^3 K)].
      rho_cp_gas: Gas volumetric heat capacity [J/(m^3 K)].
      rho_solid: Solid density [kg/m^3].
      rho_gas: Gas density at the wall [kg/m^3].
      latent_heat: Effective heat of pyrolysis [J/kg].
      ablation_temperature: Onset temperature of ablation [K].
      ablation_width: Width of the smooth onset [K].
      ablation_rate: Maximum volumetric recession rate of phi [1/s].

    The defaults describe a carbon-phenolic-like ablator. The ablation sink is
    treated explicitly, so dt must keep dt * |q| well below rho_cp * T.
    """
    k_solid: float = 0.5
    k_gas: float = 0.05
    rho_cp_solid: float = 2.0e6
    rho_cp_gas: float = 1.0e3
    rho_solid: float = 1.4e3
    rho_gas: float = 1.0e-2
    latent_heat: float = 2.0e6
    ablation_temperature: float = 3000.0
    ablation_width: float = 100.0
    ablation_rate: float = 0.1


def mixture_properties(phi: jax.Array,
                       mat: Materials) -> tuple[jax.Array, jax.Array]:
    """Conductivity and volumetric heat capacity, linear in solid fraction."""
    k_eff = phi * mat.k_solid + (1.0 - phi) * mat.k_gas
    rho_cp = phi * mat.rho_cp_solid + (1.0 - phi) * mat.rho_cp_gas
    return k_eff, rho_cp


def ablation_source(T: jax.Array, phi: jax.Array, dt: float,
                    mat: Materials) -> tuple[jax.Array, jax.Array]:
    """Endothermic ablation sink and the updated solid fraction.

    The recession rate d(phi)/dt = -rate * phi * sigmoid((T - T_abl) / width)
    is a smooth (differentiable) surrogate for an Arrhenius onset.

    Args:
      T: Temperature [K].
      phi: Solid fraction in [0, 1].
      dt: Time step [s].
      mat: Material properties.

    Returns:
      A tuple (q, phi_new): volumetric heat sink q [W/m^3] (non-positive)
      and the solid fraction after the step.
    """
    activation = jax.nn.sigmoid((T - mat.ablation_temperature) /
                                mat.ablation_width)
    dphi_dt = -mat.ablation_rate * phi * activation
    q = mat.rho_solid * mat.latent_heat * dphi_dt
    phi_new = jnp.clip(phi + dt * dphi_dt, 0.0, 1.0)
    return q, phi_new


def _neumann_pad(f: jax.Array) -> jax.Array:
    """One ghost layer reproducing a zero-normal-gradient boundary."""
    return jnp.pad(f, 1, mode='edge')


def diffusion(T: jax.Array, k_eff: jax.Array, dx: float,
              dy: float) -> jax.Array:
    """div(k grad T) with harmonic-mean face conductivities, zero-flux walls."""
    Tp, kp = _neumann_pad(T), _neumann_pad(k_eff)

    def face_k(a, b):
        return 2.0 * a * b / (a + b)

    kx_e = face_k(kp[1:-1, 1:-1], kp[1:-1, 2:])
    kx_w = face_k(kp[1:-1, 1:-1], kp[1:-1, :-2])
    ky_n = face_k(kp[1:-1, 1:-1], kp[2:, 1:-1])
    ky_s = face_k(kp[1:-1, 1:-1], kp[:-2, 1:-1])
    flux_x = (kx_e * (Tp[1:-1, 2:] - T) - kx_w * (T - Tp[1:-1, :-2])) / dx**2
    flux_y = (ky_n * (Tp[2:, 1:-1] - T) - ky_s * (T - Tp[:-2, 1:-1])) / dy**2
    return flux_x + flux_y


def upwind_advection(T: jax.Array, u: jax.Array, v: jax.Array, dx: float,
                     dy: float) -> jax.Array:
    """First-order upwind discretization of u . grad T."""
    Tp = _neumann_pad(T)
    dTdx_back = (T - Tp[1:-1, :-2]) / dx
    dTdx_fwd = (Tp[1:-1, 2:] - T) / dx
    dTdy_back = (T - Tp[:-2, 1:-1]) / dy
    dTdy_fwd = (Tp[2:, 1:-1] - T) / dy
    return (jnp.where(u > 0, u * dTdx_back, u * dTdx_fwd) +
            jnp.where(v > 0, v * dTdy_back, v * dTdy_fwd))


@functools.partial(jax.jit,
                   static_argnames=('dx', 'dy', 'dt', 'mat', 'maxiter'))
def implicit_monolithic_step(
    T_old: jax.Array,
    phi_old: jax.Array,
    u: jax.Array,
    v: jax.Array,
    dx: float,
    dy: float,
    dt: float,
    mat: Materials = Materials(),
    maxiter: int = 50,
    surface_heat_flux: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """One backward-Euler step of the coupled solid/gas energy equation.

    Args:
      T_old: Temperature [K], shape (ny, nx); row 0 is the heated surface.
      phi_old: Solid fraction, shape (ny, nx).
      u: Gas x-velocity [m/s] (zero inside the solid).
      v: Gas y-velocity [m/s] (zero inside the solid).
      dx: Cell width [m].
      dy: Cell height [m].
      dt: Time step [s].
      mat: Material properties.
      maxiter: Maximum GMRES restarts.
      surface_heat_flux: Optional heat flux into row 0 [W/m^2], shape (nx,),
        e.g. the DSMC wall-heat tally divided by the wall area and dt.

    Returns:
      A tuple (T_new, phi_new).
    """
    k_eff, rho_cp = mixture_properties(phi_old, mat)
    q, phi_new = ablation_source(T_old, phi_old, dt, mat)
    if surface_heat_flux is not None:
        q = q.at[0].add(surface_heat_flux / dy)

    def operator(T):
        return rho_cp * T - dt * (diffusion(T, k_eff, dx, dy) -
                                  mat.rho_cp_gas * upwind_advection(
                                      T, u, v, dx, dy))

    # The backward-Euler operator is an M-matrix with condition number
    # O(1e4) for typical TPS meshes, so unpreconditioned GMRES converges. (JAX
    # tests convergence on the preconditioned residual, which lets a Jacobi
    # preconditioner stop early when rho_cp spans three orders of magnitude.)
    rhs = rho_cp * T_old + dt * q
    T_new, _ = jax.scipy.sparse.linalg.gmres(
        operator, rhs, x0=T_old, tol=1e-6, atol=0.0, restart=20,
        maxiter=maxiter, solve_method='batched')
    return T_new, phi_new


def blowing_velocity(phi_old: jax.Array, phi_new: jax.Array, dt: float,
                     dy: float, mat: Materials = Materials()) -> jax.Array:
    """Pyrolysis-gas blowing velocity at the surface of every column.

    Mass conservation per column: the solid mass lost through the column,
    sum_y rho_solid * (phi_old - phi_new) * dy, leaves the surface as gas,
    rho_gas * v_blow * dt.

    Args:
      phi_old: Solid fraction before the step, shape (ny, nx).
      phi_new: Solid fraction after the step, shape (ny, nx).
      dt: Time step [s].
      dy: Cell height [m].
      mat: Material properties.

    Returns:
      Blowing velocity [m/s] for each column, shape (nx,).
    """
    mass_flux = mat.rho_solid * jnp.sum(phi_old - phi_new, axis=0) * dy / dt
    return mass_flux / mat.rho_gas
