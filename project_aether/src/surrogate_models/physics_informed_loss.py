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
"""Physics-informed training loss for the flow-field surrogate.

Report Part IV, 4.1: training minimizes data error plus a PDE-residual penalty
in the style of physics-informed neural networks (M. Raissi, P. Perdikaris,
G. E. Karniadakis, J. Comput. Phys. 378, 2019):

    L = L_data + lambda * L_pde.

The surrogate predicts whole fields on the solver grid, so the residual is
evaluated with the same differentiable finite-volume operator that generated
the training data (`cfd_navier_stokes.evaluate_rhs`). For a predicted step
Q^n -> Q^{n+1} the compressible Navier-Stokes residual is the trapezoidal
(Crank-Nicolson) balance

    R = (Q^{n+1} - Q^n) / dt - 1/2 [RHS(Q^n) + RHS(Q^{n+1})],

which vanishes for any state the conservation laws allow. Gradients of
L_pde flow through the solver operator into the surrogate parameters.

Both terms are measured in normalized units (per-channel standard deviation),
because density and total energy differ by orders of magnitude.

Typical usage example:

  loss, aux = physics_informed_loss.loss_fn(q_pred, q_true, q_prev, scale,
                                            flow, physics_weight=1e-2)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.physics_engine import cfd_navier_stokes


class FlowSetup(NamedTuple):
    """Discretization the surrogate was trained on (hashable, static).

    Attributes:
      dt: Time step between the input and the predicted field [s].
      dx: Cell width [m].
      dy: Cell height [m].
      mu: Dynamic viscosity [Pa s].
      bc: Boundary conditions.
    """
    dt: float
    dx: float
    dy: float
    mu: float
    bc: cfd_navier_stokes.BoundaryConditions = (
        cfd_navier_stokes.BoundaryConditions())


def navier_stokes_residual(q_prev: jax.Array, q_next: jax.Array,
                           flow: FlowSetup) -> jax.Array:
    """Crank-Nicolson residual of one step, shape (batch, 4, ny, nx).

    Args:
      q_prev: Conserved fields at step n, shape (batch, 4, ny, nx).
      q_next: Conserved fields at step n + 1, same shape.
      flow: Discretization.

    Returns:
      The residual R = (q_next - q_prev) / dt - (RHS(q_prev) + RHS(q_next))/2.
    """
    rhs = jax.vmap(lambda q: cfd_navier_stokes.evaluate_rhs(
        q, flow.dx, flow.dy, flow.mu, flow.bc))
    return (q_next - q_prev) / flow.dt - 0.5 * (rhs(q_prev) + rhs(q_next))


def loss_fn(q_pred: jax.Array, q_true: jax.Array, q_prev: jax.Array,
            scale: jax.Array, flow: FlowSetup,
            physics_weight: float | jax.Array
            ) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Total physics-informed loss and its components.

    Args:
      q_pred: Predicted fields at step n + 1, shape (batch, 4, ny, nx).
      q_true: Reference fields at step n + 1, same shape.
      q_prev: Input fields at step n, same shape.
      scale: Per-channel normalization scale, shape (4,).
      flow: Discretization used for the PDE residual.
      physics_weight: lambda in L = L_data + lambda * L_pde.

    Returns:
      A tuple (total, {'data': L_data, 'pde': L_pde}).
    """
    s = scale[None, :, None, None]
    data = jnp.mean(((q_pred - q_true) / s)**2)
    residual = navier_stokes_residual(q_prev, q_pred, flow)
    pde = jnp.mean((residual * flow.dt / s)**2)
    total = data + physics_weight * pde
    return total, {'data': data, 'pde': pde}


def annealed_physics_weight(step: int | jax.Array, total_steps: int,
                            start: float = 1e-4,
                            end: float = 1e-1) -> jax.Array:
    """Geometric ramp of lambda from `start` to `end` over training.

    The data term dominates early; physical consistency is then enforced
    progressively, which avoids collapsing to trivial residual minimizers.
    """
    frac = jnp.clip(jnp.asarray(step, jnp.float32) / max(total_steps, 1),
                    0.0, 1.0)
    return start * (end / start)**frac
