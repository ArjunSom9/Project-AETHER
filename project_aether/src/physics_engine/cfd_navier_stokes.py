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
"""Differentiable 2D compressible Navier-Stokes finite-volume solver in JAX.

This is the continuum region of the AETHER timestep (report Part III, 3.1).
The scheme is a cell-centered finite-volume method:

  * MUSCL reconstruction of primitive variables with a minmod limiter,
  * Rusanov (local Lax-Friedrichs) inviscid fluxes, which stay stable through
    the strong shocks of hypersonic flow,
  * compact central-difference viscous and heat-conduction fluxes,
  * three-stage strong-stability-preserving Runge-Kutta (SSP-RK3) in time.

Every stage is written functionally, so XLA can fuse reconstruction, flux
evaluation, and the Runge-Kutta update of a jitted step into a few kernels,
and `jax.grad` differentiates through the whole simulation.

The conserved state has shape (4, ny, nx) with components
[rho, rho*u, rho*v, E]. Short names (`Q`, `F`, `rho`, `u`, `p`, `T`) follow
standard CFD notation.

Typical usage example:

  bc = cfd_navier_stokes.BoundaryConditions.hypersonic_channel(inflow)
  q_final, _ = cfd_navier_stokes.simulate(q0, 100, dx, dy, dt, mu, bc)
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

GAMMA = 1.4  # Ratio of specific heats for air.
PRANDTL = 0.72  # Prandtl number for air.
R_GAS = 287.05  # Specific gas constant for air [J/(kg K)].
NUM_GHOST = 2  # Ghost layers required by the MUSCL stencil.

_DENSITY_FLOOR = 1e-12
_PRESSURE_FLOOR = 1e-12

PERIODIC = 'periodic'
OUTFLOW = 'outflow'
INFLOW = 'inflow'
SLIP_WALL = 'slip_wall'
NO_SLIP_WALL = 'no_slip_wall'
_VALID_KINDS = (PERIODIC, OUTFLOW, INFLOW, SLIP_WALL, NO_SLIP_WALL)


class BoundaryConditions(NamedTuple):
    """Boundary condition kinds for the four sides of the rectangular domain.

    Attributes:
      x_lo: Kind of the left boundary.
      x_hi: Kind of the right boundary.
      y_lo: Kind of the bottom boundary.
      y_hi: Kind of the top boundary.
      inflow_state: Conserved state (rho, rho*u, rho*v, E) used by any side
        whose kind is `INFLOW`.
    """
    x_lo: str = PERIODIC
    x_hi: str = PERIODIC
    y_lo: str = PERIODIC
    y_hi: str = PERIODIC
    inflow_state: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 2.5)

    @classmethod
    def hypersonic_channel(
        cls, inflow_state: tuple[float, float, float, float]
    ) -> 'BoundaryConditions':
        """Supersonic inflow left, outflow right, slip walls top and bottom."""
        return cls(x_lo=INFLOW, x_hi=OUTFLOW, y_lo=SLIP_WALL, y_hi=SLIP_WALL,
                   inflow_state=tuple(float(s) for s in inflow_state))


def conserved_from_primitive(
    rho: jax.Array, u: jax.Array, v: jax.Array, p: jax.Array
) -> jax.Array:
    """Builds the conserved state [rho, rho*u, rho*v, E] from primitives."""
    energy = p / (GAMMA - 1.0) + 0.5 * rho * (u**2 + v**2)
    return jnp.stack([rho, rho * u, rho * v, energy])


def primitives(Q: jax.Array) -> tuple[jax.Array, ...]:
    """Returns (rho, u, v, p, T) for a conserved state of shape (4, ...)."""
    rho = jnp.maximum(Q[0], _DENSITY_FLOOR)
    u = Q[1] / rho
    v = Q[2] / rho
    p = (GAMMA - 1.0) * (Q[3] - 0.5 * rho * (u**2 + v**2))
    p = jnp.maximum(p, _PRESSURE_FLOOR)
    T = p / (rho * R_GAS)
    return rho, u, v, p, T


def sound_speed(rho: jax.Array, p: jax.Array) -> jax.Array:
    """Speed of sound of a calorically perfect gas."""
    return jnp.sqrt(GAMMA * p / rho)


def max_stable_dt(Q: jax.Array, dx: float, dy: float, mu: float,
                  cfl: float = 0.4) -> jax.Array:
    """Largest time step satisfying the convective and viscous CFL limits."""
    rho, u, v, p, _ = primitives(Q)
    c = sound_speed(rho, p)
    convective = jnp.max((jnp.abs(u) + c) / dx + (jnp.abs(v) + c) / dy)
    nu_max = jnp.max(mu / rho) * max(GAMMA / PRANDTL, 4.0 / 3.0)
    viscous = 2.0 * nu_max * (1.0 / dx**2 + 1.0 / dy**2)
    return cfl / (convective + viscous)


# ------------------------------------------------------------------------------
# Ghost cells.
# ------------------------------------------------------------------------------


def ghost_layers(Q: jax.Array, kind: str, axis: int, low: bool,
                inflow_state: tuple[float, ...]) -> jax.Array:
    """Returns the NUM_GHOST ghost layers for one side along `axis`."""
    n = Q.shape[axis]
    if low:
        interior = jax.lax.slice_in_dim(Q, 0, NUM_GHOST, axis=axis)
        wrapped = jax.lax.slice_in_dim(Q, n - NUM_GHOST, n, axis=axis)
        edge = jax.lax.slice_in_dim(Q, 0, 1, axis=axis)
    else:
        interior = jax.lax.slice_in_dim(Q, n - NUM_GHOST, n, axis=axis)
        wrapped = jax.lax.slice_in_dim(Q, 0, NUM_GHOST, axis=axis)
        edge = jax.lax.slice_in_dim(Q, n - 1, n, axis=axis)
    mirrored = jnp.flip(interior, axis=axis)

    if kind == PERIODIC:
        return wrapped
    if kind == OUTFLOW:
        reps = [1] * Q.ndim
        reps[axis] = NUM_GHOST
        return jnp.tile(edge, reps)
    if kind == INFLOW:
        state = jnp.asarray(inflow_state, Q.dtype).reshape(
            (4,) + (1,) * (Q.ndim - 1))
        return jnp.broadcast_to(state, mirrored.shape)
    if kind in (SLIP_WALL, NO_SLIP_WALL):
        # Axis 2 (x) has normal momentum component 1; axis 1 (y) has 2.
        normal = 1 if axis == 2 else 2
        sign = jnp.ones((4,), Q.dtype).at[normal].set(-1.0)
        if kind == NO_SLIP_WALL:
            sign = sign.at[3 - normal].set(-1.0)
        return mirrored * sign.reshape((4,) + (1,) * (Q.ndim - 1))
    raise ValueError(f'Unknown boundary kind: {kind!r}')


def pad_with_ghosts(Q: jax.Array, bc: BoundaryConditions) -> jax.Array:
    """Adds NUM_GHOST ghost layers on every side according to `bc`.

    Args:
      Q: Conserved state of shape (4, ny, nx).
      bc: Boundary conditions.

    Returns:
      Array of shape (4, ny + 2 * NUM_GHOST, nx + 2 * NUM_GHOST).

    Raises:
      ValueError: If a boundary kind is not one of the supported kinds.
    """
    for kind in bc[:4]:
        if kind not in _VALID_KINDS:
            raise ValueError(f'Unknown boundary kind: {kind!r}')
    x_lo = ghost_layers(Q, bc.x_lo, 2, True, bc.inflow_state)
    x_hi = ghost_layers(Q, bc.x_hi, 2, False, bc.inflow_state)
    Q = jnp.concatenate([x_lo, Q, x_hi], axis=2)
    y_lo = ghost_layers(Q, bc.y_lo, 1, True, bc.inflow_state)
    y_hi = ghost_layers(Q, bc.y_hi, 1, False, bc.inflow_state)
    return jnp.concatenate([y_lo, Q, y_hi], axis=1)


# ------------------------------------------------------------------------------
# Fluxes.
# ------------------------------------------------------------------------------


def _minmod(a: jax.Array, b: jax.Array) -> jax.Array:
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.minimum(jnp.abs(a),
                                                           jnp.abs(b))


def _physical_flux(W: jax.Array, normal: int) -> tuple[jax.Array, jax.Array]:
    """Euler flux of primitive state W=[rho,u,v,p] and its max wave speed."""
    rho, u, v, p = W[0], W[1], W[2], W[3]
    un = u if normal == 0 else v
    energy = p / (GAMMA - 1.0) + 0.5 * rho * (u**2 + v**2)
    momentum_x = rho * u * un + (p if normal == 0 else 0.0)
    momentum_y = rho * v * un + (p if normal == 1 else 0.0)
    flux = jnp.stack([rho * un, momentum_x, momentum_y, (energy + p) * un])
    speed = jnp.abs(un) + sound_speed(rho, p)
    return flux, speed


def _rusanov(WL: jax.Array, WR: jax.Array, normal: int) -> jax.Array:
    """Rusanov flux between left and right primitive states."""
    FL, sL = _physical_flux(WL, normal)
    FR, sR = _physical_flux(WR, normal)
    QL = conserved_from_primitive(WL[0], WL[1], WL[2], WL[3])
    QR = conserved_from_primitive(WR[0], WR[1], WR[2], WR[3])
    return 0.5 * (FL + FR) - 0.5 * jnp.maximum(sL, sR) * (QR - QL)


def _muscl_faces(W: jax.Array, axis: int) -> tuple[jax.Array, jax.Array]:
    """Left/right face states along `axis` for the interior faces.

    `W` must carry NUM_GHOST ghost layers along `axis`; the returned states
    sit on the n + 1 faces bounding the n interior cells.
    """
    n = W.shape[axis]
    diff = (jax.lax.slice_in_dim(W, 1, n, axis=axis) -
            jax.lax.slice_in_dim(W, 0, n - 1, axis=axis))
    slope = _minmod(jax.lax.slice_in_dim(diff, 0, n - 2, axis=axis),
                    jax.lax.slice_in_dim(diff, 1, n - 1, axis=axis))
    m = n - 2 * NUM_GHOST + 1  # Number of interior faces.
    WL = (jax.lax.slice_in_dim(W, 1, 1 + m, axis=axis) +
          0.5 * jax.lax.slice_in_dim(slope, 0, m, axis=axis))
    WR = (jax.lax.slice_in_dim(W, 2, 2 + m, axis=axis) -
          0.5 * jax.lax.slice_in_dim(slope, 1, 1 + m, axis=axis))
    # Keep the reconstructed thermodynamic state physical.
    floor = jnp.asarray([_DENSITY_FLOOR, -jnp.inf, -jnp.inf, _PRESSURE_FLOOR],
                        W.dtype).reshape((4,) + (1,) * (W.ndim - 1))
    return jnp.maximum(WL, floor), jnp.maximum(WR, floor)


def _viscous_flux(u: jax.Array, v: jax.Array, T: jax.Array, axis: int,
                  dx: float, dy: float, mu: float) -> jax.Array:
    """Viscous flux on the interior faces normal to `axis` (2=x, 1=y).

    Inputs carry NUM_GHOST ghost layers on both axes.
    """
    g = NUM_GHOST
    conductivity = mu * GAMMA * R_GAS / ((GAMMA - 1.0) * PRANDTL)

    def normal_diff(f, h):
        if axis == 2:
            return (f[g:-g, g:-g + 1] - f[g:-g, g - 1:-g]) / h
        return (f[g:-g + 1, g:-g] - f[g - 1:-g, g:-g]) / h

    def tangential_diff(f, h):
        # Central difference across the face, averaged over the two cells.
        if axis == 2:
            central = (f[g + 1:-g + 1, :] - f[g - 1:-g - 1, :]) / (2.0 * h)
            return 0.5 * (central[:, g - 1:-g] + central[:, g:-g + 1])
        central = (f[:, g + 1:-g + 1] - f[:, g - 1:-g - 1]) / (2.0 * h)
        return 0.5 * (central[g - 1:-g, :] + central[g:-g + 1, :])

    def face_avg(f):
        if axis == 2:
            return 0.5 * (f[g:-g, g - 1:-g] + f[g:-g, g:-g + 1])
        return 0.5 * (f[g - 1:-g, g:-g] + f[g:-g + 1, g:-g])

    h_n, h_t = (dx, dy) if axis == 2 else (dy, dx)
    un_n = normal_diff(u if axis == 2 else v, h_n)  # d(u_n)/dn
    ut_n = normal_diff(v if axis == 2 else u, h_n)  # d(u_t)/dn
    un_t = tangential_diff(u if axis == 2 else v, h_t)  # d(u_n)/dt
    ut_t = tangential_diff(v if axis == 2 else u, h_t)  # d(u_t)/dt
    T_n = normal_diff(T, h_n)

    tau_nn = mu * (4.0 / 3.0 * un_n - 2.0 / 3.0 * ut_t)
    tau_nt = mu * (ut_n + un_t)
    u_f, v_f = face_avg(u), face_avg(v)
    if axis == 2:
        tau_xx_or_yy, tau_xy = tau_nn, tau_nt
        work = u_f * tau_xx_or_yy + v_f * tau_xy
        momentum = (tau_xx_or_yy, tau_xy)
    else:
        tau_xx_or_yy, tau_xy = tau_nn, tau_nt
        work = u_f * tau_xy + v_f * tau_xx_or_yy
        momentum = (tau_xy, tau_xx_or_yy)
    zero = jnp.zeros_like(work)
    return jnp.stack([zero, momentum[0], momentum[1],
                      work + conductivity * T_n])


def rhs_from_padded(Qp: jax.Array, dx: float, dy: float,
                    mu: float) -> jax.Array:
    """Time derivative dQ/dt of the interior cells of a ghost-padded state.

    Args:
      Qp: Conserved state with NUM_GHOST ghost layers, shape
        (4, ny + 2 * NUM_GHOST, nx + 2 * NUM_GHOST).
      dx: Cell width.
      dy: Cell height.
      mu: Dynamic viscosity; zero gives the Euler equations.

    Returns:
      dQ/dt of shape (4, ny, nx).
    """
    g = NUM_GHOST
    rho, u, v, p, T = primitives(Qp)
    W = jnp.stack([rho, u, v, p])

    WLx, WRx = _muscl_faces(W[:, g:-g, :], axis=2)
    Fx = _rusanov(WLx, WRx, normal=0)
    WLy, WRy = _muscl_faces(W[:, :, g:-g], axis=1)
    Fy = _rusanov(WLy, WRy, normal=1)

    if mu > 0.0:
        Fx = Fx - _viscous_flux(u, v, T, 2, dx, dy, mu)
        Fy = Fy - _viscous_flux(u, v, T, 1, dx, dy, mu)

    div_x = (Fx[:, :, 1:] - Fx[:, :, :-1]) / dx
    div_y = (Fy[:, 1:, :] - Fy[:, :-1, :]) / dy
    return -(div_x + div_y)


def evaluate_rhs(Q: jax.Array, dx: float, dy: float, mu: float,
                 bc: BoundaryConditions = BoundaryConditions()) -> jax.Array:
    """Time derivative dQ/dt of the conserved state with boundary conditions."""
    return rhs_from_padded(pad_with_ghosts(Q, bc), dx, dy, mu)


# ------------------------------------------------------------------------------
# Time integration.
# ------------------------------------------------------------------------------


def ssp_rk3_step(Q: jax.Array, dt: float, rhs_fn) -> jax.Array:
    """One Shu-Osher SSP-RK3 step of dQ/dt = rhs_fn(Q)."""
    q1 = Q + dt * rhs_fn(Q)
    q2 = 0.75 * Q + 0.25 * (q1 + dt * rhs_fn(q1))
    return Q / 3.0 + 2.0 / 3.0 * (q2 + dt * rhs_fn(q2))


@functools.partial(jax.jit, static_argnames=('dx', 'dy', 'mu', 'bc'))
def step(Q: jax.Array, dt: float, dx: float, dy: float, mu: float,
         bc: BoundaryConditions = BoundaryConditions()) -> jax.Array:
    """Advances the state by one SSP-RK3 step.

    Args:
      Q: Conserved state of shape (4, ny, nx).
      dt: Time step; may be a traced value.
      dx: Cell width.
      dy: Cell height.
      mu: Dynamic viscosity.
      bc: Boundary conditions.

    Returns:
      The conserved state after one step.
    """
    return ssp_rk3_step(Q, dt, lambda q: evaluate_rhs(q, dx, dy, mu, bc))


@functools.partial(jax.jit,
                   static_argnames=('num_steps', 'dx', 'dy', 'mu', 'bc'))
def simulate(Q_init: jax.Array, num_steps: int, dx: float, dy: float,
             dt: float, mu: float,
             bc: BoundaryConditions = BoundaryConditions()
             ) -> tuple[jax.Array, jax.Array]:
    """Runs `num_steps` SSP-RK3 steps entirely on device with `lax.scan`.

    Args:
      Q_init: Initial conserved state of shape (4, ny, nx).
      num_steps: Number of time steps.
      dx: Cell width.
      dy: Cell height.
      dt: Time step.
      mu: Dynamic viscosity.
      bc: Boundary conditions.

    Returns:
      A tuple (Q_final, trajectory), where trajectory has shape
      (num_steps, 4, ny, nx) and holds the state after every step.
    """

    def body(q, _):
        q_next = ssp_rk3_step(q, dt, lambda s: evaluate_rhs(s, dx, dy, mu, bc))
        return q_next, q_next

    return jax.lax.scan(body, Q_init, None, length=num_steps)
