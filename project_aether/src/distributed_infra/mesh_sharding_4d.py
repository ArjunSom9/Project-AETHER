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
"""Four-axis device mesh and the parallel layouts used across AETHER.

Mesh axes:
  * 'data': independent ensemble members / request batches,
  * 'pipe': pipeline stages,
  * 'tensor': Megatron-style tensor parallelism inside a node,
  * 'domain': spatial domain decomposition of the flow field.

Parallelism is expressed two ways (report Part VI, 6.3):
  * GSPMD (Y. Xu et al., arXiv:2105.04663): sharding annotations on inputs and
    a few intermediates; XLA propagates them and inserts the collectives
    (`megatron_mlp_gspmd`).
  * shard_map: explicit per-device code where AETHER chooses the collective,
    e.g. an SCCL-synthesized all-reduce for the Megatron MLP
    (`megatron_mlp_shard_map`), the CFD halo exchange
    (`distributed_cfd_rhs`), or the shifting-buffer pipeline of the GSPMD
    paper (`pipeline_forward`).

`hilbert_permutation` orders unstructured points along a Hilbert curve so
contiguous shards along 'domain' are spatially compact.

Typical usage example:

  mesh = make_mesh({'data': 1, 'pipe': 2, 'tensor': 2, 'domain': 2})
  y = megatron_mlp_gspmd(params, x, mesh)
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.experimental import shard_map

from src.physics_engine import cfd_navier_stokes

AXES = ('data', 'pipe', 'tensor', 'domain')
_spec = jax.sharding.PartitionSpec


def make_mesh(axis_sizes: dict[str, int],
              devices: list[jax.Device] | None = None) -> jax.sharding.Mesh:
    """Builds a mesh with the four AETHER axes (missing axes get size 1).

    Args:
      axis_sizes: Size of each named axis; the product must equal the number
        of devices.
      devices: Devices to use (default: all).

    Returns:
      The mesh, with axes in `AXES` order.

    Raises:
      ValueError: If an unknown axis name is given.
    """
    unknown = set(axis_sizes) - set(AXES)
    if unknown:
        raise ValueError(f'Unknown mesh axes: {sorted(unknown)}')
    shape = tuple(axis_sizes.get(a, 1) for a in AXES)
    devices = jax.devices() if devices is None else devices
    grid = mesh_utils.create_device_mesh(shape, devices)
    return jax.sharding.Mesh(grid, AXES)


# ------------------------------------------------------------------------------
# Hilbert-curve domain decomposition.
# ------------------------------------------------------------------------------


def hilbert_index(ix: jax.Array, iy: jax.Array, order: int) -> jax.Array:
    """Distance along the Hilbert curve of integer cells (ix, iy).

    Vectorized form of the classic iterative algorithm (Hilbert curve on a
    2^order x 2^order grid).

    Args:
      ix: Column indices in [0, 2^order).
      iy: Row indices in [0, 2^order).
      order: Curve order.

    Returns:
      Curve distance in [0, 4^order), same shape as ix.
    """
    n = 1 << order
    x, y = ix.astype(jnp.int32), iy.astype(jnp.int32)
    d = jnp.zeros_like(x)
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(jnp.int32)
        ry = ((y & s) > 0).astype(jnp.int32)
        d = d + s * s * ((3 * rx) ^ ry)
        flip = (ry == 0) & (rx == 1)
        x = jnp.where(flip, n - 1 - x, x)
        y = jnp.where(flip, n - 1 - y, y)
        swap = ry == 0
        x, y = jnp.where(swap, y, x), jnp.where(swap, x, y)
        s >>= 1
    return d


def hilbert_permutation(coords: jax.Array, order: int = 10) -> jax.Array:
    """Permutation sorting points in [0, 1)^2 along the Hilbert curve."""
    cells = jnp.clip((coords * (1 << order)).astype(jnp.int32), 0,
                     (1 << order) - 1)
    return jnp.argsort(hilbert_index(cells[:, 0], cells[:, 1], order))


def shard_by_hilbert(state: jax.Array, coords: jax.Array,
                     mesh: jax.sharding.Mesh) -> tuple[jax.Array, jax.Array]:
    """Reorders points along the Hilbert curve and shards them over 'domain'.

    Args:
      state: Per-point state, (N, F); N must divide by the 'domain' size.
      coords: Point coordinates in [0, 1)^2, (N, 2).
      mesh: Device mesh.

    Returns:
      A tuple (sharded_state, permutation).
    """
    perm = hilbert_permutation(coords)
    sharding = jax.sharding.NamedSharding(mesh, _spec('domain', None))
    return jax.device_put(state[perm], sharding), perm


# ------------------------------------------------------------------------------
# Megatron-style tensor parallelism.
# ------------------------------------------------------------------------------


def init_mlp(key: jax.Array, d_model: int, d_hidden: int) -> dict[str, Any]:
    k1, k2 = jax.random.split(key)
    return {
        'up': {'w': jax.random.normal(k1, (d_model, d_hidden)) /
                    np.sqrt(d_model),
               'b': jnp.zeros((d_hidden,))},
        'down': {'w': jax.random.normal(k2, (d_hidden, d_model)) /
                      np.sqrt(d_hidden),
                 'b': jnp.zeros((d_model,))},
    }


def mlp(params: dict[str, Any], x: jax.Array) -> jax.Array:
    """Reference (unsharded) two-layer GELU MLP."""
    h = jax.nn.gelu(x @ params['up']['w'] + params['up']['b'])
    return h @ params['down']['w'] + params['down']['b']


def megatron_mlp_specs() -> dict[str, Any]:
    """Column-parallel up-projection, row-parallel down-projection."""
    return {'up': {'w': _spec(None, 'tensor'), 'b': _spec('tensor')},
            'down': {'w': _spec('tensor', None), 'b': _spec()}}


def _place(tree: Any, specs: Any, mesh: jax.sharding.Mesh) -> Any:
    return jax.tree_util.tree_map(
        lambda spec, x: jax.device_put(
            x, jax.sharding.NamedSharding(mesh, spec)),
        specs, tree, is_leaf=lambda n: isinstance(n, _spec))


def megatron_mlp_gspmd(params: dict[str, Any], x: jax.Array,
                       mesh: jax.sharding.Mesh) -> jax.Array:
    """Megatron MLP from sharding annotations alone (GSPMD inserts psum).

    Args:
      params: Unsharded MLP parameters.
      x: Activations (batch, d_model), sharded over 'data'.
      mesh: Device mesh with a 'tensor' axis.

    Returns:
      MLP output, sharded like `x`.
    """
    params = _place(params, megatron_mlp_specs(), mesh)
    x = jax.device_put(x, jax.sharding.NamedSharding(mesh, _spec('data')))

    @jax.jit
    def run(p, a):
        h = jax.nn.gelu(a @ p['up']['w'] + p['up']['b'])
        h = jax.lax.with_sharding_constraint(
            h, jax.sharding.NamedSharding(mesh, _spec('data', 'tensor')))
        return h @ p['down']['w'] + p['down']['b']

    return run(params, x)


def megatron_mlp_shard_map(
    params: dict[str, Any], x: jax.Array, mesh: jax.sharding.Mesh,
    allreduce: Callable[[jax.Array, str], jax.Array] = jax.lax.psum
) -> jax.Array:
    """Megatron MLP with an explicit, replaceable all-reduce.

    Args:
      params: Unsharded MLP parameters.
      x: Activations (batch, d_model), replicated.
      mesh: Device mesh with a 'tensor' axis.
      allreduce: `fn(partial, axis_name)` summing over the axis, e.g.
        `sccl_synthesizer.make_allreduce(algorithm)`.

    Returns:
      MLP output (batch, d_model), replicated.
    """
    specs = megatron_mlp_specs()

    def body(p, a):
        h = jax.nn.gelu(a @ p['up']['w'] + p['up']['b'])
        return allreduce(h @ p['down']['w'], 'tensor') + p['down']['b']

    fn = shard_map.shard_map(body, mesh, in_specs=(specs, _spec()),
                             out_specs=_spec(), check_rep=False)
    return jax.jit(fn)(_place(params, specs, mesh), x)


# ------------------------------------------------------------------------------
# Domain decomposition of the CFD solver with halo exchange.
# ------------------------------------------------------------------------------


def distributed_cfd_rhs(Q: jax.Array, dx: float, dy: float, mu: float,
                        bc: cfd_navier_stokes.BoundaryConditions,
                        mesh: jax.sharding.Mesh) -> jax.Array:
    """dQ/dt with the x-axis of the grid split over the 'domain' mesh axis.

    Each shard receives NUM_GHOST ghost columns from its neighbors with
    `ppermute` (wrapping around for periodic x boundaries); shards at the
    physical boundary fill them from the boundary condition instead.

    Args:
      Q: Conserved state (4, ny, nx); nx / domain_size >= NUM_GHOST.
      dx: Cell width.
      dy: Cell height.
      mu: Dynamic viscosity.
      bc: Boundary conditions.
      mesh: Device mesh with a 'domain' axis.

    Returns:
      dQ/dt of shape (4, ny, nx), sharded along x.
    """
    n = mesh.shape['domain']
    g = cfd_navier_stokes.NUM_GHOST
    periodic = bc.x_lo == cfd_navier_stokes.PERIODIC
    to_right = [(i, (i + 1) % n) for i in range(n if periodic else n - 1)]
    to_left = [((i + 1) % n, i) for i in range(n if periodic else n - 1)]

    def body(q):
        rank = jax.lax.axis_index('domain')
        from_left = jax.lax.ppermute(q[:, :, -g:], 'domain', to_right)
        from_right = jax.lax.ppermute(q[:, :, :g], 'domain', to_left)
        if not periodic:
            wall_lo = cfd_navier_stokes.ghost_layers(q, bc.x_lo, 2, True,
                                                     bc.inflow_state)
            wall_hi = cfd_navier_stokes.ghost_layers(q, bc.x_hi, 2, False,
                                                     bc.inflow_state)
            from_left = jnp.where(rank == 0, wall_lo, from_left)
            from_right = jnp.where(rank == n - 1, wall_hi, from_right)
        q = jnp.concatenate([from_left, q, from_right], axis=2)
        y_lo = cfd_navier_stokes.ghost_layers(q, bc.y_lo, 1, True,
                                              bc.inflow_state)
        y_hi = cfd_navier_stokes.ghost_layers(q, bc.y_hi, 1, False,
                                              bc.inflow_state)
        q = jnp.concatenate([y_lo, q, y_hi], axis=1)
        return cfd_navier_stokes.rhs_from_padded(q, dx, dy, mu)

    spec = _spec(None, None, 'domain')
    fn = shard_map.shard_map(body, mesh, in_specs=spec, out_specs=spec)
    return jax.jit(fn)(jax.device_put(Q, jax.sharding.NamedSharding(mesh,
                                                                     spec)))


# ------------------------------------------------------------------------------
# GSPMD-style pipeline parallelism with a shifting buffer.
# ------------------------------------------------------------------------------


def pipeline_forward(stage_params: Any,
                     stage_fn: Callable[[Any, jax.Array], jax.Array],
                     microbatches: jax.Array,
                     mesh: jax.sharding.Mesh) -> jax.Array:
    """Runs a stack of identical stages as a pipeline over 'pipe'.

    The stage parameters carry a leading stage dimension sharded over 'pipe'.
    Every iteration each stage processes its buffer and shifts the result to
    the next stage with `ppermute` (GSPMD paper, Section 3.3); stage 0 feeds
    a new microbatch, the last stage emits a finished one.

    Args:
      stage_params: Pytree whose leaves have leading dimension num_stages.
      stage_fn: `fn(params_of_one_stage, x) -> x` (shape preserving).
      microbatches: Inputs of shape (num_microbatches, ...), replicated.
      mesh: Device mesh with a 'pipe' axis of size num_stages.

    Returns:
      Outputs for every microbatch, same shape as `microbatches`.
    """
    num_stages = mesh.shape['pipe']
    num_micro = microbatches.shape[0]
    shift = [(i, i + 1) for i in range(num_stages - 1)]

    def body(params, mbs):
        params = jax.tree_util.tree_map(lambda p: p[0], params)
        stage = jax.lax.axis_index('pipe')
        buffer = jnp.zeros_like(mbs[0])
        outputs = jnp.zeros_like(mbs)
        for t in range(num_micro + num_stages - 1):
            feed = mbs[t] if t < num_micro else jnp.zeros_like(mbs[0])
            out = stage_fn(params, jnp.where(stage == 0, feed, buffer))
            k = t - (num_stages - 1)
            if 0 <= k < num_micro:
                outputs = jnp.where(stage == num_stages - 1,
                                    outputs.at[k].set(out), outputs)
            buffer = jax.lax.ppermute(out, 'pipe', shift)
        return jax.lax.psum(
            jnp.where(stage == num_stages - 1, outputs, 0.0), 'pipe')

    param_specs = jax.tree_util.tree_map(lambda _: _spec('pipe'),
                                         stage_params)
    fn = shard_map.shard_map(body, mesh, in_specs=(param_specs, _spec()),
                             out_specs=_spec(), check_rep=False)
    return jax.jit(fn)(stage_params, microbatches)
