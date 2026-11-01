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
"""Phase 3 (weeks 5-7): audit XLA fusion of the jitted timestep (Table 3).

Reports memory traffic per DSMC timestep, in passes over the particle state,
for the unfused eager kernel sequence, the stagewise pipeline (one compiled
executable per stage), and the single jitted timestep, plus the same audit for
one SSP-RK3 step of the continuum solver. Numbers come from XLA's cost model
of the compiled executables on the current backend; XLA:CPU fuses less than
XLA:GPU, so run on an H100 to reproduce the report's GPU figures.

Usage (from project_aether/):
  python -m scripts.phase3_fusion_audit --particles 4000000
  python -m scripts.phase3_fusion_audit --dump-hlo outputs/hlo
"""

import argparse
import functools
import pathlib

import jax
import jax.numpy as jnp
import tabulate

from src.physics_engine import cfd_navier_stokes as cfd
from src.physics_engine import dsmc_boltzmann as dsmc
from src.physics_engine import xla_fusion_audit


def audit_dsmc(args) -> dsmc.FusionAudit:
    """Prints and returns the eager / stagewise / fused DSMC audit."""
    cfg = dsmc.DsmcConfig(lo=(0.0, 0.0, 0.0), hi=(0.1, 0.1, 0.064),
                          cells=(25, 25, 16), dt=2e-7, fnum=2e14)
    state = dsmc.initialize_maxwellian(jax.random.PRNGKey(0), args.particles,
                                       cfg, 2000.0, drift=(-2500.0, 0., 0.))
    audit = dsmc.audit_dsmc_fusion(state, jax.random.PRNGKey(1), cfg)
    rows = [['Unfused eager kernel sequence',
             f'{audit.passes(audit.eager_bytes):.1f}', '-']]
    for name, report in audit.stages:
        rows.append([f'  stage: {name}',
                     f'{audit.passes(report.bytes_accessed):.1f}',
                     f'{report.num_fusions}/{report.num_kernels}'])
    rows.append(['Stagewise total',
                 f'{audit.passes(audit.stagewise_bytes):.1f}', '-'])
    rows.append(['Fused jitted timestep',
                 f'{audit.passes(audit.fused.bytes_accessed):.1f}',
                 f'{audit.fused.num_fusions}/{audit.fused.num_kernels}'])
    print(f'DSMC timestep, {args.particles:,} particles, '
          f'{cfg.num_cells:,} cells, backend {jax.default_backend()}')
    print(tabulate.tabulate(rows, headers=['Pipeline', 'State passes',
                                           'Fusions / kernels'],
                            tablefmt='github'))
    print(f'Bytes reduction, eager -> fused: {audit.reduction:.2f}x\n')
    return audit


def audit_cfd(args) -> xla_fusion_audit.CostReport:
    """Prints the eager vs. fused audit of one SSP-RK3 continuum step."""
    ny, nx = args.grid
    q = cfd.conserved_from_primitive(jnp.ones((ny, nx)),
                                     jnp.full((ny, nx), 100.0),
                                     jnp.zeros((ny, nx)),
                                     jnp.full((ny, nx), 1e5))
    rhs = functools.partial(cfd.evaluate_rhs, dx=1e-3, dy=1e-3, mu=1.8e-5)
    step = functools.partial(cfd.ssp_rk3_step, dt=1e-7, rhs_fn=rhs)
    report = xla_fusion_audit.analyze(step, q)
    eager = xla_fusion_audit.eager_bytes(step, q)
    state = q.nbytes
    print(f'Continuum SSP-RK3 step, {ny}x{nx} cells')
    print(tabulate.tabulate(
        [['Unfused eager kernel sequence', f'{eager / state:.1f}', '-'],
         ['Fused jitted step', f'{report.bytes_accessed / state:.1f}',
          f'{report.num_fusions}/{report.num_kernels}']],
        headers=['Pipeline', 'State passes', 'Fusions / kernels'],
        tablefmt='github'))
    print(f'Bytes reduction, eager -> fused: '
          f'{eager / report.bytes_accessed:.2f}x')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--particles', type=int, default=1_000_000)
    parser.add_argument('--grid', type=int, nargs=2, default=(256, 512))
    parser.add_argument('--dump-hlo', type=pathlib.Path, default=None)
    args = parser.parse_args()
    dsmc_audit = audit_dsmc(args)
    cfd_report = audit_cfd(args)
    if args.dump_hlo is not None:
        args.dump_hlo.mkdir(parents=True, exist_ok=True)
        (args.dump_hlo / 'dsmc_step.hlo').write_text(
            dsmc_audit.fused.hlo_text)
        (args.dump_hlo / 'cfd_step.hlo').write_text(cfd_report.hlo_text)
        print(f'Optimized HLO written to {args.dump_hlo}/')


if __name__ == '__main__':
    main()
