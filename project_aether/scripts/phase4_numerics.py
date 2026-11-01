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
"""Phase 4 (weeks 7-9): FP8 quantization and block-sparse numerics (Part V).

1. Post-training FP8 (E4M3) quantization of the trained surrogate with a
   calibration set: weight memory vs. BF16 and the accuracy cost.
2. The 75%-sparse window + global block layouts of the Table 4 GEMM shapes,
   and the exactness of block-sparse attention against dense masked
   attention.
3. If the CUDA build exists, runs the CUTLASS block-sparse GEMM benchmark
   (Table 4: block-sparse vs. dense cuBLAS on H100).

Usage (from project_aether/):
  python -m scripts.train_surrogate --out outputs/surrogate.npz
  python -m scripts.phase4_numerics --checkpoint outputs/surrogate.npz
"""

import argparse
import pathlib
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
import tabulate

from scripts import train_surrogate
from src.surrogate_models import block_sparse_attention as bsa
from src.surrogate_models import galerkin_transformer as gt
from src.surrogate_models import quantization_fp8 as q8

_BENCHMARK = (pathlib.Path(__file__).resolve().parents[1] / 'src' /
              'custom_kernels' / 'build' / 'block_sparse_gemm_benchmark')


def fp8_report(checkpoint: pathlib.Path) -> None:
    """Quantizes the surrogate to FP8 and prints memory and accuracy."""
    cfg = train_surrogate.MODEL_CONFIG
    if checkpoint.exists():
        params, norm, inputs, targets = train_surrogate.load_checkpoint(
            checkpoint)
        source = f'trained checkpoint {checkpoint}'
    else:
        params = gt.init_params(jax.random.PRNGKey(0), cfg)
        inputs = np.asarray(1.0 + jax.random.uniform(jax.random.PRNGKey(1),
                                                     (8, 4, 16, 32)))
        targets = inputs
        norm = gt.Normalizer.fit(jnp.asarray(inputs))
        source = 'randomly initialized model (no checkpoint found)'
    inputs, targets = jnp.asarray(inputs), jnp.asarray(targets)

    calibration = q8.AmaxCalibrator()
    gt.apply(params, inputs[:4], norm, cfg, calib=calibration)  # Eager.
    qparams = q8.quantize_params(params, calibration)
    apply = jax.jit(gt.apply, static_argnames='cfg')
    reference = apply(params, inputs, norm, cfg)
    quantized = apply(qparams, inputs, norm, cfg)

    def rel(a, b):
        return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))

    bf16_bytes = q8.dense_weight_bytes(params, jnp.bfloat16)
    fp8_bytes = q8.dense_weight_bytes(qparams)
    print(f'FP8 post-training quantization of the {source}')
    print(tabulate.tabulate(
        [['BF16 (baseline)', f'{bf16_bytes / 1024:.1f}', '1.00x',
          f'{rel(reference, targets):.3e}', '-'],
         ['FP8 E4M3 + calibrated activations', f'{fp8_bytes / 1024:.1f}',
          f'{fp8_bytes / bf16_bytes:.2f}x', f'{rel(quantized, targets):.3e}',
          f'{rel(quantized, reference):.3e}']],
        headers=['Weights', 'KiB', 'Memory vs BF16', 'Error vs target',
                 'Error vs BF16 model'], tablefmt='github'))
    print()


def sparsity_report() -> None:
    """Prints the Table 4 layouts and checks block-sparse attention."""
    rows = []
    for size, block in ((4096, 64), (4096, 128), (8192, 128)):
        num_blocks = size // block
        window = bsa.window_for_density(num_blocks, 1, 0.25)
        layout = bsa.block_sparse_layout(num_blocks, window, 1)
        rows.append([f'{size}x{size}', f'{block}x{block}', window,
                     f'{1.0 - bsa.layout_density(layout):.1%}'])
    print('Window + global block layouts of the Table 4 GEMM shapes')
    print(tabulate.tabulate(rows, headers=['Matrix', 'Block', 'Window',
                                           'Sparsity'], tablefmt='github'))

    length, block = 2048, 64
    layout = bsa.block_sparse_layout(
        length // block, bsa.window_for_density(length // block), 1)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    q, k, v = (jax.random.normal(key, (1, 4, length, 64)) for key in keys)
    sparse_fn = jax.jit(lambda a, b, c: bsa.block_sparse_attention(
        a, b, c, layout, block))
    dense_fn = jax.jit(lambda a, b, c: bsa.dense_masked_attention(
        a, b, c, layout, block))
    out_sparse = sparse_fn(q, k, v).block_until_ready()
    out_dense = dense_fn(q, k, v).block_until_ready()
    timings = []
    for fn in (sparse_fn, dense_fn):
        start = time.perf_counter()
        for _ in range(5):
            fn(q, k, v).block_until_ready()
        timings.append((time.perf_counter() - start) / 5 * 1e3)
    print(f'\nBlock-sparse attention, L={length}, 4 heads '
          f'({jax.default_backend()}): {timings[0]:.2f} ms vs dense masked '
          f'{timings[1]:.2f} ms; max |difference| '
          f'{float(jnp.abs(out_sparse - out_dense).max()):.1e}')
    print('(This XLA path is the exactness reference; the block-sparse '
          'speedup of report Table 4 is the CUTLASS kernel on H100.)\n')


def cutlass_report() -> None:
    if not _BENCHMARK.exists():
        print('CUTLASS benchmark not built. On an sm_90 machine with CUDA 12.2:'
              '\n  cmake -S src/custom_kernels -B src/custom_kernels/build '
              '-DCMAKE_BUILD_TYPE=Release\n  cmake --build '
              'src/custom_kernels/build -j')
        return
    print(subprocess.run([str(_BENCHMARK)], check=False, capture_output=True,
                         text=True).stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', type=pathlib.Path,
                        default=pathlib.Path('outputs/surrogate.npz'))
    args = parser.parse_args()
    fp8_report(args.checkpoint)
    sparsity_report()
    cutlass_report()


if __name__ == '__main__':
    main()
