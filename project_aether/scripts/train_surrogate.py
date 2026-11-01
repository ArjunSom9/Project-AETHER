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
"""Trains the physics-informed Galerkin Transformer surrogate (report Part IV).

1. Generates training pairs (Q^n, Q^{n+1}) with the differentiable JAX
   Navier-Stokes solver from randomized hypersonic channel flows.
2. Trains the surrogate on data loss + annealed PDE-residual loss, with the
   batch sharded across all local devices (data parallel; GSPMD inserts the
   gradient all-reduce).
3. Saves parameters and the normalizer to an .npz file that
   `phase4_numerics` quantizes to FP8.

Usage (from project_aether/):
  python -m scripts.train_surrogate --steps 300 --out outputs/surrogate.npz
"""

import argparse
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.physics_engine import cfd_navier_stokes as cfd
from src.surrogate_models import galerkin_transformer as gt
from src.surrogate_models import physics_informed_loss as pil

MODEL_CONFIG = gt.GalerkinConfig(width=64, num_layers=4, modes=8)


def make_dataset(key: jax.Array, num_flows: int, steps: int,
                 grid: tuple[int, int], flow: pil.FlowSetup
                 ) -> tuple[jax.Array, jax.Array]:
    """(Q^n, Q^{n+1}) pairs from randomized channel flows."""
    ny, nx = grid
    inputs, targets = [], []
    for k in jax.random.split(key, num_flows):
        k1, k2, k3 = jax.random.split(k, 3)
        mach = jax.random.uniform(k1, minval=2.0, maxval=6.0)
        rho0, temp0 = 0.02, 250.0
        p0 = rho0 * cfd.R_GAS * temp0
        u0 = mach * jnp.sqrt(cfd.GAMMA * p0 / rho0)
        y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')
        cy = jax.random.uniform(k2, minval=0.3, maxval=0.7) * ny
        bump = jnp.exp(-((x - nx / 3)**2 + (y - cy)**2) /
                       (2.0 * (0.08 * nx)**2))
        rho = rho0 * (1.0 + 2.0 * bump)
        p = p0 * (1.0 + 4.0 * bump * jax.random.uniform(k3, minval=0.5,
                                                        maxval=1.5))
        q0 = cfd.conserved_from_primitive(rho, jnp.full_like(rho, u0),
                                          jnp.zeros_like(rho), p)
        _, trajectory = cfd.simulate(q0, steps, flow.dx, flow.dy, flow.dt,
                                     flow.mu, flow.bc)
        trajectory = jnp.concatenate([q0[None], trajectory])
        inputs.append(trajectory[:-1])
        targets.append(trajectory[1:])
    return jnp.concatenate(inputs), jnp.concatenate(targets)


def _leaf_name(path) -> str:
    return '/'.join(str(getattr(p, 'key', getattr(p, 'idx', p))) for p in path)


def flatten_tree(tree, prefix: str = '') -> dict[str, np.ndarray]:
    return {prefix + _leaf_name(path): np.asarray(leaf)
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}


def load_checkpoint(path: pathlib.Path
                    ) -> tuple[dict, gt.Normalizer, np.ndarray, np.ndarray]:
    """Loads (params, normalizer, sample inputs, sample targets)."""
    with np.load(path) as data:
        template = gt.init_params(jax.random.PRNGKey(0), MODEL_CONFIG)
        leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
        params = jax.tree_util.tree_unflatten(
            treedef, [jnp.asarray(data['params/' + _leaf_name(p)])
                      for p, _ in leaves])
        norm = gt.Normalizer(jnp.asarray(data['norm_mean']),
                             jnp.asarray(data['norm_std']))
        return (params, norm, data['sample_inputs'],
                data['sample_targets'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--flows', type=int, default=8)
    parser.add_argument('--trajectory-steps', type=int, default=24)
    parser.add_argument('--grid', type=int, nargs=2, default=(16, 32))
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', type=pathlib.Path,
                        default=pathlib.Path('outputs/surrogate.npz'))
    args = parser.parse_args()

    # Same freestream as the generated flows, used for the inflow boundary.
    rho0, temp0 = 0.02, 250.0
    p0 = rho0 * cfd.R_GAS * temp0
    inflow = cfd.conserved_from_primitive(
        jnp.float32(rho0), jnp.float32(4.0 * np.sqrt(cfd.GAMMA * p0 / rho0)),
        jnp.float32(0.0), jnp.float32(p0))
    flow = pil.FlowSetup(
        dt=2e-7, dx=1e-3, dy=1e-3, mu=1.8e-5,
        bc=cfd.BoundaryConditions.hypersonic_channel(tuple(
            float(v) for v in inflow)))

    start = time.perf_counter()
    inputs, targets = make_dataset(jax.random.PRNGKey(0), args.flows,
                                   args.trajectory_steps, tuple(args.grid),
                                   flow)
    print(f'{inputs.shape[0]} training pairs generated in '
          f'{time.perf_counter() - start:.1f} s')

    cfg = MODEL_CONFIG
    norm = gt.Normalizer.fit(targets)
    params = gt.init_params(jax.random.PRNGKey(1), cfg)
    # Warmup and global-norm clipping keep early updates from pushing the
    # predicted density or internal energy negative, where the PDE residual
    # (evaluated through the solver) blows up.
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, warmup_steps=max(args.steps // 10, 1),
        decay_steps=args.steps)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adamw(schedule, weight_decay=1e-4))
    opt_state = optimizer.init(params)

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('data',))
    batch_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('data'))
    replicated = jax.sharding.NamedSharding(mesh,
                                            jax.sharding.PartitionSpec())
    params, opt_state = jax.device_put((params, opt_state), replicated)

    @jax.jit
    def train_step(params, opt_state, q_in, q_out, weight):
        def loss(p):
            pred = gt.apply(p, q_in, norm, cfg)
            return pil.loss_fn(pred, q_out, q_in, norm.std, flow, weight)

        (total, aux), grads = jax.value_and_grad(loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, total, aux

    batch = max(args.batch // len(jax.devices()), 1) * len(jax.devices())
    rng = np.random.default_rng(0)
    for step in range(args.steps):
        idx = rng.choice(inputs.shape[0], size=batch, replace=False)
        q_in, q_out = jax.device_put((inputs[idx], targets[idx]),
                                     batch_sharding)
        weight = pil.annealed_physics_weight(step, args.steps)
        params, opt_state, total, aux = train_step(params, opt_state, q_in,
                                                   q_out, weight)
        if step % max(args.steps // 10, 1) == 0 or step == args.steps - 1:
            print(f'step {step:4d} | lambda {float(weight):.1e} | '
                  f'total {float(total):.4e} | data {float(aux["data"]):.4e} '
                  f'| pde {float(aux["pde"]):.4e}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **flatten_tree(params, 'params/'),
             norm_mean=np.asarray(norm.mean), norm_std=np.asarray(norm.std),
             sample_inputs=np.asarray(inputs[:8]),
             sample_targets=np.asarray(targets[:8]))
    print(f'Saved {gt.num_parameters(params):,} parameters to {args.out}')


if __name__ == '__main__':
    main()
