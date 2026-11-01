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
"""Phase 2 (weeks 3-5): fused Triton binning sort vs. a 32-core CPU baseline.

Measures particles per second of the DSMC binning stage (cell id, in-cell
shuffle, sort, per-cell counts) on the report's benchmark case (4M particles,
10k collision cells) for

  * the CPU baseline: the same pipeline in PyTorch on N OpenMP threads, with a
    stateful Mersenne-Twister generator, and
  * the fused Triton pipeline (`fused_radix_sort`).

Both results are checked for identical per-cell counts. With
`--collisions`, it also runs the TMA collision kernel on the sorted GPU
particles and compares mean collisions per cell with the JAX implementation of
the same NTC algorithm (statistical agreement).

Usage (Linux + NVIDIA GPU, from project_aether/):
  python -m scripts.phase2_triton_sort_benchmark --cpu-threads 32
"""

import argparse
import time

import numpy as np
import tabulate
import torch

from src.custom_kernels.triton import fused_radix_sort
from src.custom_kernels.triton import radix_sort_reference


def cpu_binning(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                grid: radix_sort_reference.CellGrid,
                generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    """Unfused CPU pipeline: cell ids, MT19937 shuffle bits, sort, counts."""
    nx, ny, _ = grid.cells
    idx = []
    for c, l, h, n in zip((x, y, z), grid.lo, grid.hi, grid.cells):
        idx.append(((c - l) * (n / (h - l))).to(torch.int32).clamp_(0, n - 1))
    cell = (idx[2] * ny + idx[1]) * nx + idx[0]
    shuffle = torch.randint(0, 1 << radix_sort_reference.SHUFFLE_BITS,
                            cell.shape, generator=generator,
                            dtype=torch.int32)
    keys = (cell << radix_sort_reference.SHUFFLE_BITS) | shuffle
    sorted_keys, order = torch.sort(keys, stable=True)
    counts = torch.bincount(cell, minlength=grid.num_cells)
    return sorted_keys, order, counts


def time_cpu(args, positions, grid) -> tuple[float, torch.Tensor]:
    torch.set_num_threads(args.cpu_threads)
    generator = torch.Generator().manual_seed(0)
    x, y, z = (p.cpu() for p in positions)
    cpu_binning(x, y, z, grid, generator)  # Warm up.
    start = time.perf_counter()
    for _ in range(args.repeats):
        _, _, counts = cpu_binning(x, y, z, grid, generator)
    return (time.perf_counter() - start) / args.repeats, counts


def time_gpu(args, positions, grid) -> tuple[float, object]:
    result = fused_radix_sort.sort_particles_by_cell(*positions, grid, 1)
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for step in range(args.repeats):
        result = fused_radix_sort.sort_particles_by_cell(*positions, grid,
                                                         step + 2)
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / 1e3 / args.repeats, result


def check_collision_statistics(args, grid, result) -> None:
    """GPU TMA collisions vs. the JAX NTC reference on the same particles."""
    # Imported lazily: the comparison needs JAX and the built CUDA library.
    import jax  # pylint: disable=import-outside-toplevel
    from src.custom_kernels.cuda import tma_collide  # pylint: disable=import-outside-toplevel
    from src.physics_engine import dsmc_boltzmann  # pylint: disable=import-outside-toplevel

    gas = dsmc_boltzmann.GasModel()
    cfg = dsmc_boltzmann.DsmcConfig(lo=grid.lo, hi=grid.hi, cells=grid.cells,
                                    dt=args.dt, fnum=args.fnum, gas=gas)
    n = args.particles
    rng = np.random.default_rng(1)
    thermal = np.sqrt(dsmc_boltzmann.BOLTZMANN * 3000.0 / gas.mass)
    vel = rng.normal(0.0, thermal, (n, 3)).astype(np.float32)
    index = result.particle_index.long().cpu().numpy()
    vel_sorted = vel[index]
    g_ref = 4.0 * thermal
    sigma_g = float(dsmc_boltzmann.vhs_cross_section(np.float32(g_ref),
                                                     gas)) * g_ref
    majorant = np.full((grid.num_cells,), sigma_g, np.float32)

    library = tma_collide.load_library(args.cuda_library)
    coefficient, exponent = tma_collide.vhs_parameters(gas.mass, gas.d_ref,
                                                       gas.omega, gas.t_ref)
    padded = -(-n // 4) * 4
    velocity = torch.zeros((3, padded), dtype=torch.float32, device='cuda')
    velocity[:, :n] = torch.as_tensor(vel_sorted.T, device='cuda')
    sigma = torch.as_tensor(majorant, device='cuda')
    collisions = torch.zeros(grid.num_cells, dtype=torch.int32, device='cuda')
    params = tma_collide.CollisionParams(
        args.fnum * args.dt / cfg.cell_volume, coefficient, exponent, 7, 0)
    tma_collide.collide(library, velocity, result.cell_start, sigma,
                        collisions, n, params)
    gpu = collisions.cpu().numpy().astype(np.float64)

    sorted_cell = (result.keys >> radix_sort_reference.SHUFFLE_BITS).cpu()
    p = dsmc_boltzmann.Particles(*(jax.numpy.zeros(n) for _ in range(3)),
                                 *(jax.numpy.asarray(vel_sorted[:, i])
                                   for i in range(3)))
    _, _, _, ref = dsmc_boltzmann.collide(
        p, jax.numpy.asarray(sorted_cell.numpy()),
        jax.numpy.asarray(result.cell_start.cpu().numpy()),
        jax.numpy.asarray(majorant), jax.random.PRNGKey(0), cfg)
    ref = np.asarray(ref, np.float64)
    z = abs(gpu.mean() - ref.mean()) / np.sqrt(
        (gpu.var() + ref.var()) / grid.num_cells)
    print(f'Mean collisions per cell: GPU {gpu.mean():.3f}, '
          f'JAX reference {ref.mean():.3f} (z = {z:.2f}; |z| < 3 means '
          'agreement within stochastic noise).')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--particles', type=int, default=4_000_000)
    parser.add_argument('--cells', type=int, nargs=3, default=(25, 25, 16))
    parser.add_argument('--cpu-threads', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--collisions', action='store_true')
    parser.add_argument('--cuda-library', default=None)
    parser.add_argument('--dt', type=float, default=1e-6)
    parser.add_argument('--fnum', type=float, default=1e12)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('This benchmark needs a CUDA GPU.')

    grid = radix_sort_reference.CellGrid((0.0, 0.0, 0.0), (1.0, 1.0, 0.64),
                                         tuple(args.cells))
    gen = torch.Generator(device='cuda').manual_seed(0)
    scale = torch.tensor(grid.hi, device='cuda')
    positions = [torch.rand(args.particles, device='cuda', generator=gen) *
                 scale[i] for i in range(3)]

    cpu_seconds, cpu_counts = time_cpu(args, positions, grid)
    gpu_seconds, result = time_gpu(args, positions, grid)
    counts_match = torch.equal(cpu_counts.to(torch.int32),
                               result.cell_counts.cpu())
    rows = [
        [f'CPU baseline ({args.cpu_threads} threads)',
         f'{cpu_seconds * 1e3:.2f}',
         f'{args.particles / cpu_seconds / 1e6:.1f}', '1.0x'],
        ['Fused Triton sort (GPU)', f'{gpu_seconds * 1e3:.2f}',
         f'{args.particles / gpu_seconds / 1e6:.1f}',
         f'{cpu_seconds / gpu_seconds:.1f}x'],
    ]
    print(f'{args.particles:,} particles, {grid.num_cells:,} cells, '
          f'{radix_sort_reference.num_passes(grid.num_cells)} digit passes')
    print(tabulate.tabulate(rows, headers=['Pipeline', 'ms / step',
                                           'M particles / s', 'Speedup'],
                            tablefmt='github'))
    print(f'Per-cell counts identical: {counts_match}')
    if args.collisions:
        check_collision_statistics(args, grid, result)


if __name__ == '__main__':
    main()
