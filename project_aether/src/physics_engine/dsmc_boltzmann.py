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
"""Direct Simulation Monte Carlo (DSMC) timestep written as one JAX function.

Implements Bird's DSMC method (G. A. Bird, Molecular Gas Dynamics and the
Direct Simulation of Gas Flows, 1994) for a rarefied gas in a box whose x_lo
face is the ablating thermal-protection-system (TPS) surface:

  1. Move: ballistic transport, specular reflection on five faces, diffuse
     re-emission at the TPS wall with a heat-flux tally (the "boundary and
     ablation tally" of report Table 3).
  2. Bin: every particle gets the sort key ``cell_id << shuffle_bits | r``,
     where ``r`` are counter-based (Threefry) random bits. One sort therefore
     groups particles by collision cell *and* shuffles them inside the cell,
     so adjacent sorted particles are random collision partners. This fuses
     collision-partner search into the sort, exactly like the Triton GPU
     pipeline in `src/custom_kernels/triton/fused_radix_sort.py`.
  3. Collide: No-Time-Counter (NTC) selection with variable-hard-sphere (VHS)
     cross sections and isotropic scattering, which conserves momentum and
     energy pair by pair.

`dsmc_step` traces all three stages into one jitted function, so XLA sees the
whole timestep and fuses the elementwise stages (report Part III, 3.2).
`dsmc_step_unfused` runs the same stages as separately compiled executables;
`audit_dsmc_fusion` compares the two with XLA's own cost model.

`relaxed_dsmc_step` replaces the discrete accept/reject test with a
Gumbel-softmax relaxation so gradients reach the cross-section parameters.

Typical usage example:

  cfg = dsmc_boltzmann.DsmcConfig(lo=(0., 0., 0.), hi=(1., 1., 1.),
                                  cells=(16, 16, 4), dt=1e-6, fnum=1e12)
  state = dsmc_boltzmann.initialize_maxwellian(key, 100_000, cfg, 300.)
  state, stats = dsmc_boltzmann.dsmc_step(state, key, cfg)
"""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.physics_engine import xla_fusion_audit

BOLTZMANN = 1.380649e-23  # [J/K]


class GasModel(NamedTuple):
    """Variable-hard-sphere (VHS) gas model.

    The defaults are Bird's (1994, Table A1) single-species model of air.

    Attributes:
      mass: Molecular mass [kg].
      d_ref: VHS reference diameter [m].
      omega: Viscosity-temperature exponent.
      t_ref: Reference temperature of d_ref [K].
    """
    mass: float = 48.1e-27
    d_ref: float = 4.19e-10
    omega: float = 0.77
    t_ref: float = 273.0


class DsmcConfig(NamedTuple):
    """Static configuration of a DSMC box (hashable, used as a jit static arg).

    Attributes:
      lo: Lower corner (x, y, z) of the box [m]; x_lo is the TPS wall.
      hi: Upper corner (x, y, z) of the box [m].
      cells: Number of collision cells along (x, y, z).
      dt: Time step [s].
      fnum: Real molecules represented by one simulated particle.
      wall_temperature: TPS wall temperature for diffuse re-emission [K].
      gas: VHS gas model.
      shuffle_bits: Random bits appended to the cell id in the sort key.
    """
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    cells: tuple[int, int, int]
    dt: float
    fnum: float
    wall_temperature: float = 1500.0
    gas: GasModel = GasModel()
    shuffle_bits: int = 7

    @property
    def num_cells(self) -> int:
        return self.cells[0] * self.cells[1] * self.cells[2]

    @property
    def cell_volume(self) -> float:
        return math.prod((h - l) / n
                         for l, h, n in zip(self.lo, self.hi, self.cells))


class Particles(NamedTuple):
    """Structure-of-Arrays particle storage; every field has shape (N,)."""
    x: jax.Array
    y: jax.Array
    z: jax.Array
    vx: jax.Array
    vy: jax.Array
    vz: jax.Array


class DsmcState(NamedTuple):
    """Particles plus the per-cell NTC majorant (sigma_T * g)_max."""
    particles: Particles
    sigma_g_max: jax.Array


class StepStats(NamedTuple):
    """Diagnostics of one timestep.

    Attributes:
      candidates: NTC candidate pairs per cell, shape (C,).
      collisions: Accepted collisions per cell, shape (C,).
      wall_hits: Simulated particles that struck the TPS wall.
      wall_heat: Energy deposited on the TPS wall by real molecules [J].
      cell_start: Exclusive prefix sum of particles per cell, shape (C + 1,).
    """
    candidates: jax.Array
    collisions: jax.Array
    wall_hits: jax.Array
    wall_heat: jax.Array
    cell_start: jax.Array


def _speeds(p: Particles) -> jax.Array:
    return jnp.stack([p.vx, p.vy, p.vz], axis=-1)


def vhs_cross_section(g: jax.Array, gas: GasModel,
                      d_ref: jax.Array | float | None = None) -> jax.Array:
    """Total VHS cross section sigma_T(g) for equal-mass collision partners.

    Bird (1994) eq. (4.63) with reduced mass m / 2.

    Args:
      g: Relative speed [m/s].
      gas: Gas model.
      d_ref: Optional (possibly traced) override of gas.d_ref, used when
        differentiating with respect to the molecular diameter.

    Returns:
      Cross section [m^2], same shape as g.
    """
    d = gas.d_ref if d_ref is None else d_ref
    reduced_mass = 0.5 * gas.mass
    g2 = jnp.maximum(g * g, 1e-12)
    ratio = 2.0 * BOLTZMANN * gas.t_ref / (reduced_mass * g2)
    return (math.pi * d**2 * ratio**(gas.omega - 0.5) /
            math.gamma(2.5 - gas.omega))


def initialize_maxwellian(key: jax.Array, num_particles: int, cfg: DsmcConfig,
                          temperature: float,
                          drift: tuple[float, float, float] = (0., 0., 0.)
                          ) -> DsmcState:
    """Uniformly fills the box with a drifting Maxwellian gas.

    Args:
      key: PRNG key.
      num_particles: Number of simulated particles.
      cfg: Box configuration.
      temperature: Gas temperature [K].
      drift: Bulk velocity [m/s].

    Returns:
      The initial state with (sigma_T g)_max set from the thermal speed.
    """
    k_pos, k_vel = jax.random.split(key)
    lo = jnp.asarray(cfg.lo, jnp.float32)
    hi = jnp.asarray(cfg.hi, jnp.float32)
    pos = lo + (hi - lo) * jax.random.uniform(k_pos, (num_particles, 3))
    thermal = math.sqrt(BOLTZMANN * temperature / cfg.gas.mass)
    vel = (jnp.asarray(drift, jnp.float32) +
           thermal * jax.random.normal(k_vel, (num_particles, 3)))
    particles = Particles(pos[:, 0], pos[:, 1], pos[:, 2],
                          vel[:, 0], vel[:, 1], vel[:, 2])
    # Bird's recommendation: start the majorant a few times above the mean.
    g_ref = 4.0 * thermal
    sigma_g = float(vhs_cross_section(jnp.float32(g_ref), cfg.gas)) * g_ref
    return DsmcState(particles,
                     jnp.full((cfg.num_cells,), sigma_g, jnp.float32))


# ------------------------------------------------------------------------------
# Stage 1: move and wall interaction.
# ------------------------------------------------------------------------------


def move(p: Particles, key: jax.Array,
         cfg: DsmcConfig) -> tuple[Particles, jax.Array, jax.Array]:
    """Ballistic move with specular walls and a diffuse, tallied TPS wall.

    Args:
      p: Particles.
      key: PRNG key for diffuse re-emission.
      cfg: Box configuration.

    Returns:
      A tuple (particles, wall_hits, wall_heat).
    """
    dt = cfg.dt
    x, y, z = p.x + p.vx * dt, p.y + p.vy * dt, p.z + p.vz * dt
    vx, vy, vz = p.vx, p.vy, p.vz

    # Diffuse re-emission at the TPS wall (x = lo_x): a flux-weighted
    # half-Maxwellian at the wall temperature, with full accommodation. All
    # three uniforms come from one counter-based draw; the tangential normals
    # use the Box-Muller transform.
    hit = x < cfg.lo[0]
    wall_speed = math.sqrt(BOLTZMANN * cfg.wall_temperature / cfg.gas.mass)
    u = jax.random.uniform(key, (3,) + x.shape, minval=1e-7, maxval=1.0)
    vx_out = wall_speed * jnp.sqrt(-2.0 * jnp.log(u[0]))
    radius = wall_speed * jnp.sqrt(-2.0 * jnp.log(u[1]))
    vy_out = radius * jnp.cos(2.0 * math.pi * u[2])
    vz_out = radius * jnp.sin(2.0 * math.pi * u[2])
    e_in = vx * vx + vy * vy + vz * vz
    e_out = vx_out * vx_out + vy_out * vy_out + vz_out * vz_out
    wall_heat = (0.5 * cfg.gas.mass * cfg.fnum *
                 jnp.sum(jnp.where(hit, e_in - e_out, 0.0)))
    wall_hits = jnp.sum(hit.astype(jnp.int32))
    x = jnp.where(hit, 2.0 * cfg.lo[0] - x, x)
    vx = jnp.where(hit, vx_out, vx)
    vy = jnp.where(hit, vy_out, vy)
    vz = jnp.where(hit, vz_out, vz)

    # Specular reflection on the remaining five faces.
    x_hi = x > cfg.hi[0]
    x = jnp.where(x_hi, 2.0 * cfg.hi[0] - x, x)
    # astroid cannot infer jax.Array here; the operand is always an array.
    vx = jnp.where(x_hi, -vx, vx)  # pylint: disable=invalid-unary-operand-type
    coords, vels = [], []
    for axis, (c, v) in enumerate(((y, vy), (z, vz)), start=1):
        lo_hit, hi_hit = c < cfg.lo[axis], c > cfg.hi[axis]
        c = jnp.where(lo_hit, 2.0 * cfg.lo[axis] - c, c)
        c = jnp.where(hi_hit, 2.0 * cfg.hi[axis] - c, c)
        v = jnp.where(lo_hit | hi_hit, -v, v)
        coords.append(c)
        vels.append(v)

    # Guard against particles that crossed more than one cell length.
    def inside(c, axis):
        width = cfg.hi[axis] - cfg.lo[axis]
        return jnp.clip(c, cfg.lo[axis], cfg.hi[axis] - 1e-6 * width)

    x, y, z = inside(x, 0), inside(coords[0], 1), inside(coords[1], 2)
    return (Particles(x, y, z, vx, vels[0], vels[1]), wall_hits, wall_heat)


# ------------------------------------------------------------------------------
# Stage 2: binning (sort by cell with in-cell shuffle).
# ------------------------------------------------------------------------------


def cell_ids(p: Particles, cfg: DsmcConfig) -> jax.Array:
    """Linear collision-cell index of every particle, shape (N,), int32."""
    idx = []
    for axis, c in enumerate((p.x, p.y, p.z)):
        inv = cfg.cells[axis] / (cfg.hi[axis] - cfg.lo[axis])
        i = jnp.floor((c - cfg.lo[axis]) * inv).astype(jnp.int32)
        idx.append(jnp.clip(i, 0, cfg.cells[axis] - 1))
    nx, ny, _ = cfg.cells
    return (idx[2] * ny + idx[1]) * nx + idx[0]


def bin_particles(p: Particles, key: jax.Array,
                  cfg: DsmcConfig) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sorts particles by cell, shuffling them randomly inside each cell.

    Args:
      p: Particles.
      key: PRNG key for the in-cell shuffle bits.
      cfg: Box configuration.

    Returns:
      A tuple (perm, sorted_cell, cell_start): the sorting permutation, the
      cell id of every sorted particle, and the exclusive prefix sum of the
      per-cell particle counts (shape (C + 1,)).
    """
    cid = cell_ids(p, cfg)
    shuffle = jax.random.randint(key, cid.shape, 0, 1 << cfg.shuffle_bits,
                                 dtype=jnp.int32)
    sort_key = (cid << cfg.shuffle_bits) | shuffle
    sorted_key, perm = jax.lax.sort_key_val(
        sort_key, jnp.arange(cid.shape[0], dtype=jnp.int32))
    sorted_cell = sorted_key >> cfg.shuffle_bits
    cell_start = jnp.searchsorted(
        sorted_cell, jnp.arange(cfg.num_cells + 1, dtype=jnp.int32),
        side='left').astype(jnp.int32)
    return perm, sorted_cell, cell_start


def permute(p: Particles, perm: jax.Array) -> Particles:
    """Gathers every SoA field into sorted order."""
    return Particles(*(field[perm] for field in p))


# ------------------------------------------------------------------------------
# Stage 3: NTC collisions on the sorted particles.
# ------------------------------------------------------------------------------


class _PairGeometry(NamedTuple):
    """Candidate-pair bookkeeping shared by the hard and relaxed collisions."""
    cell: jax.Array  # Cell of each sorted particle.
    candidate: jax.Array  # True where particle p is the first of a candidate.
    v: jax.Array  # Velocities (N, 3).
    v_partner: jax.Array  # Velocity of particle p + 1, (N, 3).
    g: jax.Array  # Relative speed |v - v_partner|, (N,).


def _pair_geometry(p: Particles, sorted_cell: jax.Array, cell_start: jax.Array,
                   sigma_g_max: jax.Array, key: jax.Array,
                   cfg: DsmcConfig) -> _PairGeometry:
    n = sorted_cell.shape[0]
    count = cell_start[1:] - cell_start[:-1]
    local = jnp.arange(n, dtype=jnp.int32) - cell_start[sorted_cell]
    n_cell = count[sorted_cell]

    # NTC: expected selections 1/2 N (N - 1) F_N (sigma g)_max dt / V_c,
    # stochastically rounded, capped by the disjoint pairs available.
    expected = (0.5 * count * (count - 1) * cfg.fnum * sigma_g_max * cfg.dt /
                cfg.cell_volume)
    selections = jnp.floor(
        expected + jax.random.uniform(key, count.shape)).astype(jnp.int32)
    selections = jnp.minimum(selections, count // 2)

    candidate = ((local % 2 == 0) & (local + 1 < n_cell) &
                 (local // 2 < selections[sorted_cell]))
    v = _speeds(p)
    v_partner = jnp.roll(v, -1, axis=0)
    g = jnp.linalg.norm(v - v_partner, axis=-1)
    return _PairGeometry(sorted_cell, candidate, v, v_partner, g)


def _scatter_isotropic(geo: _PairGeometry, u_chi: jax.Array,
                       u_eps: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Post-collision velocities of both partners for isotropic VHS scattering.

    Args:
      geo: Pair geometry.
      u_chi: Uniform [0, 1) draws for the deflection angle, shape (N,).
      u_eps: Uniform [0, 1) draws for the azimuth, shape (N,).

    Returns:
      A tuple (v1_post, v2_post) of (N, 3) velocities.
    """
    cos_chi = 2.0 * u_chi - 1.0
    sin_chi = jnp.sqrt(jnp.maximum(1.0 - cos_chi**2, 0.0))
    eps = 2.0 * math.pi * u_eps
    g_post = geo.g[:, None] * jnp.stack(
        [sin_chi * jnp.cos(eps), sin_chi * jnp.sin(eps), cos_chi], axis=-1)
    v_cm = 0.5 * (geo.v + geo.v_partner)
    return v_cm + 0.5 * g_post, v_cm - 0.5 * g_post


def _blend(geo: _PairGeometry, weight: jax.Array, v1_post: jax.Array,
           v2_post: jax.Array) -> jax.Array:
    """Returns (N, 3) velocities blended toward the post-collision values."""
    w_first = weight[:, None]
    w_second = jnp.roll(weight, 1)[:, None]
    v = geo.v + w_first * (v1_post - geo.v)
    v = v + w_second * (jnp.roll(v2_post, 1, axis=0) - geo.v)
    return v


def collide(p: Particles, sorted_cell: jax.Array, cell_start: jax.Array,
            sigma_g_max: jax.Array, key: jax.Array, cfg: DsmcConfig
            ) -> tuple[Particles, jax.Array, jax.Array, jax.Array]:
    """NTC collisions between adjacent particles of each (shuffled) cell.

    Args:
      p: Particles in sorted order.
      sorted_cell: Cell of every sorted particle.
      cell_start: Exclusive prefix sum of per-cell counts, shape (C + 1,).
      sigma_g_max: Per-cell majorant (sigma_T g)_max.
      key: PRNG key.
      cfg: Box configuration.

    Returns:
      A tuple (particles, sigma_g_max, candidates, collisions); the last two
      are per-cell counts.
    """
    k_sel, k_pair = jax.random.split(key)
    geo = _pair_geometry(p, sorted_cell, cell_start, sigma_g_max, k_sel, cfg)
    sigma_g = vhs_cross_section(geo.g, cfg.gas) * geo.g
    # One counter-based draw supplies acceptance, deflection, and azimuth.
    u = jax.random.uniform(k_pair, (3,) + geo.g.shape)
    accept = geo.candidate & (u[0] * sigma_g_max[geo.cell] < sigma_g)
    v1_post, v2_post = _scatter_isotropic(geo, u[1], u[2])
    v = _blend(geo, accept.astype(geo.v.dtype), v1_post, v2_post)

    num_cells = cfg.num_cells
    observed = jax.ops.segment_max(
        jnp.where(geo.candidate, sigma_g, 0.0), geo.cell,
        num_segments=num_cells, indices_are_sorted=True)
    sigma_g_max = jnp.maximum(sigma_g_max, observed)
    candidates = jax.ops.segment_sum(geo.candidate.astype(jnp.int32), geo.cell,
                                     num_segments=num_cells,
                                     indices_are_sorted=True)
    collisions = jax.ops.segment_sum(accept.astype(jnp.int32), geo.cell,
                                     num_segments=num_cells,
                                     indices_are_sorted=True)
    particles = p._replace(vx=v[:, 0], vy=v[:, 1], vz=v[:, 2])
    return particles, sigma_g_max, candidates, collisions


# ------------------------------------------------------------------------------
# Fused and unfused timesteps.
# ------------------------------------------------------------------------------


def _step(state: DsmcState, key: jax.Array,
          cfg: DsmcConfig) -> tuple[DsmcState, StepStats]:
    k_move, k_bin, k_col = jax.random.split(key, 3)
    p, hits, heat = move(state.particles, k_move, cfg)
    perm, sorted_cell, cell_start = bin_particles(p, k_bin, cfg)
    p = permute(p, perm)
    p, sgm, cand, coll = collide(p, sorted_cell, cell_start,
                                 state.sigma_g_max, k_col, cfg)
    return DsmcState(p, sgm), StepStats(cand, coll, hits, heat, cell_start)


_step_jit = jax.jit(_step, static_argnames='cfg')


def dsmc_step(state: DsmcState, key: jax.Array,
              cfg: DsmcConfig) -> tuple[DsmcState, StepStats]:
    """Fused DSMC timestep: move, bin, and collide in one jitted function.

    Args:
      state: Current particles and majorants.
      key: PRNG key.
      cfg: Box configuration (static).

    Returns:
      A tuple (new_state, stats). Particles come back in cell-sorted order.
    """
    return _step_jit(state, key, cfg)


_move_jit = jax.jit(move, static_argnames='cfg')
_bin_jit = jax.jit(bin_particles, static_argnames='cfg')
_permute_jit = jax.jit(permute)
_collide_jit = jax.jit(collide, static_argnames='cfg')


def dsmc_step_unfused(state: DsmcState, key: jax.Array,
                      cfg: DsmcConfig) -> tuple[DsmcState, StepStats]:
    """Same timestep as `dsmc_step`, one XLA executable per stage.

    Every stage boundary materializes the full particle state in device
    memory, which is the unfused baseline of report Table 3.
    """
    k_move, k_bin, k_col = jax.random.split(key, 3)
    p, hits, heat = _move_jit(state.particles, k_move, cfg)
    perm, sorted_cell, cell_start = _bin_jit(p, k_bin, cfg)
    p = _permute_jit(p, perm)
    p, sgm, cand, coll = _collide_jit(p, sorted_cell, cell_start,
                                      state.sigma_g_max, k_col, cfg)
    return DsmcState(p, sgm), StepStats(cand, coll, hits, heat, cell_start)


class FusionAudit(NamedTuple):
    """Memory traffic of one DSMC timestep at three levels of fusion.

    Attributes:
      eager_bytes: Bytes moved by the unfused eager kernel sequence (every
        primitive its own kernel), the baseline of report Table 1.
      stages: (name, report) for every stage compiled separately.
      fused: Cost report of the single jitted timestep.
      state_bytes: Bytes of one pass over the particle state (6 fields).
    """
    eager_bytes: float
    stages: tuple[tuple[str, xla_fusion_audit.CostReport], ...]
    fused: xla_fusion_audit.CostReport
    state_bytes: float

    @property
    def stagewise_bytes(self) -> float:
        return sum(report.bytes_accessed for _, report in self.stages)

    @property
    def reduction(self) -> float:
        """Eager-sequence bytes divided by fused-timestep bytes."""
        return self.eager_bytes / self.fused.bytes_accessed

    def passes(self, num_bytes: float) -> float:
        """Expresses `num_bytes` as passes over the particle state."""
        return num_bytes / self.state_bytes


def audit_dsmc_fusion(state: DsmcState, key: jax.Array,
                      cfg: DsmcConfig) -> FusionAudit:
    """Measures the memory traffic of the eager, stagewise, and fused steps.

    The fused and stagewise numbers come from XLA's cost analysis of the
    optimized executables; the eager number from `xla_fusion_audit.eager_bytes`.

    Args:
      state: Example state (only shapes and dtypes matter).
      key: Example PRNG key.
      cfg: Box configuration.

    Returns:
      The fusion audit.
    """
    analyze = xla_fusion_audit.analyze
    step_fn = functools.partial(_step, cfg=cfg)
    fused = analyze(step_fn, state, key)
    eager = xla_fusion_audit.eager_bytes(step_fn, state, key)
    state_bytes = float(sum(f.nbytes for f in state.particles))

    k_move, k_bin, k_col = jax.random.split(key, 3)
    p, _, _ = _move_jit(state.particles, k_move, cfg)
    perm, sorted_cell, cell_start = _bin_jit(p, k_bin, cfg)
    p_sorted = _permute_jit(p, perm)
    stages = (
        ('move', analyze(functools.partial(move, cfg=cfg), state.particles,
                         k_move)),
        ('bin', analyze(functools.partial(bin_particles, cfg=cfg), p, k_bin)),
        ('permute', analyze(permute, p, perm)),
        ('collide', analyze(functools.partial(collide, cfg=cfg), p_sorted,
                            sorted_cell, cell_start, state.sigma_g_max,
                            k_col)),
    )
    return FusionAudit(eager, stages, fused, state_bytes)


# ------------------------------------------------------------------------------
# Differentiable relaxation.
# ------------------------------------------------------------------------------


def relaxed_collide(p: Particles, sorted_cell: jax.Array,
                    cell_start: jax.Array, sigma_g_max: jax.Array,
                    key: jax.Array, cfg: DsmcConfig, d_ref: jax.Array,
                    temperature: float = 0.1) -> Particles:
    """NTC collisions with a Gumbel-softmax acceptance relaxation.

    Hard DSMC accepts a candidate pair when U < sigma_T g / (sigma_T g)_max,
    which has zero gradient almost everywhere. Here the binary
    {collide, pass} choice is drawn with the Gumbel-softmax estimator, and the
    pre- and post-collision velocities are blended by the relaxed weight, so
    gradients reach `d_ref` through sigma_T. As temperature -> 0 the weights
    approach the hard decision.

    Args:
      p: Particles in sorted order.
      sorted_cell: Cell of every sorted particle.
      cell_start: Exclusive prefix sum of per-cell counts.
      sigma_g_max: Per-cell majorant.
      key: PRNG key.
      cfg: Box configuration.
      d_ref: VHS reference diameter (may be traced).
      temperature: Gumbel-softmax temperature.

    Returns:
      Particles with blended post-collision velocities.
    """
    k_sel, k_gum, k_scat = jax.random.split(key, 3)
    geo = _pair_geometry(p, sorted_cell, cell_start, sigma_g_max, k_sel, cfg)
    sigma_g = vhs_cross_section(geo.g, cfg.gas, d_ref) * geo.g
    prob = jnp.clip(sigma_g / sigma_g_max[geo.cell], 1e-6, 1.0 - 1e-6)
    logits = jnp.stack([jnp.log(prob), jnp.log1p(-prob)], axis=-1)
    gumbel = jax.random.gumbel(k_gum, logits.shape)
    weight = jax.nn.softmax((logits + gumbel) / temperature, axis=-1)[:, 0]
    weight = jnp.where(geo.candidate, weight, 0.0)
    u = jax.random.uniform(k_scat, (2,) + geo.g.shape)
    v1_post, v2_post = _scatter_isotropic(geo, u[0], u[1])
    v = _blend(geo, weight, v1_post, v2_post)
    return p._replace(vx=v[:, 0], vy=v[:, 1], vz=v[:, 2])


@functools.partial(jax.jit, static_argnames=('cfg', 'temperature'))
def relaxed_dsmc_step(state: DsmcState, key: jax.Array, cfg: DsmcConfig,
                      d_ref: jax.Array,
                      temperature: float = 0.1) -> DsmcState:
    """Differentiable DSMC timestep (w.r.t. `d_ref` and the particle state)."""
    k_move, k_bin, k_col = jax.random.split(key, 3)
    p, _, _ = move(state.particles, k_move, cfg)
    perm, sorted_cell, cell_start = bin_particles(p, k_bin, cfg)
    p = permute(p, perm)
    p = relaxed_collide(p, sorted_cell, cell_start, state.sigma_g_max, k_col,
                        cfg, d_ref, temperature)
    return DsmcState(p, state.sigma_g_max)


def cell_moments(p: Particles, sorted_cell: jax.Array,
                 cfg: DsmcConfig) -> tuple[jax.Array, jax.Array]:
    """Per-cell number density [1/m^3] and translational temperature [K]."""
    c = cfg.num_cells
    ones = jnp.ones_like(p.x)
    count = jax.ops.segment_sum(ones, sorted_cell, num_segments=c)
    v = _speeds(p)
    mean = jax.ops.segment_sum(v, sorted_cell, num_segments=c) / jnp.maximum(
        count, 1.0)[:, None]
    dev = v - mean[sorted_cell]
    sq = jax.ops.segment_sum(jnp.sum(dev * dev, axis=-1), sorted_cell,
                             num_segments=c)
    temperature = cfg.gas.mass * sq / (3.0 * BOLTZMANN * jnp.maximum(count,
                                                                     1.0))
    density = count * cfg.fnum / cfg.cell_volume
    return density, temperature
