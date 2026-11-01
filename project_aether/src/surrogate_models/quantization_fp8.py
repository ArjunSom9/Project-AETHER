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
"""FP8 (E4M3 / E5M2) quantization of the JAX surrogate models.

Report Part V, 5.2: surrogate weights are post-training quantized to FP8 with
per-tensor scales and a light calibration set:

  * E4M3 (4 exponent / 3 mantissa bits, max 448) stores forward-pass weights
    and activations (Micikevicius et al., arXiv:2209.05433).
  * E5M2 (5 / 2 bits, max 57344) carries gradient / error-feedback tensors,
    where range matters more than precision; `fp8_matmul` quantizes the
    backward cotangents to E5M2 for FP8-aware fine-tuning.

A quantized weight is stored as a float8 array plus one float32 scale, so its
memory is exactly half that of BF16 storage (report Table 5).

All model code in `src/surrogate_models` builds dense layers with
`init_dense` and applies them with `dense`, which transparently handles
float and quantized parameters. The quantized path dequantizes into BF16
before the dot; on H100, XLA's GPU GEMM rewriter turns this
dequantize-dot pattern into an FP8 cuBLASLt GEMM, and the CUTLASS FP8 kernels
in `src/custom_kernels/cutlass` serve the same layout.

Typical usage example:

  calib = quantization_fp8.AmaxCalibrator()
  for batch in calibration_batches:
    model_apply(params, batch, calib=calib)   # Eager, records amax.
  qparams = quantization_fp8.quantize_params(params, calib)
"""

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

E4M3 = jnp.float8_e4m3fn
E5M2 = jnp.float8_e5m2
E4M3_MAX = 448.0
E5M2_MAX = 57344.0
_AMAX_FLOOR = 1e-12

# Separator for the hierarchical layer names used by calibration.
PATH_SEP = '/'


def _format_max(dtype: Any) -> float:
    if dtype == E4M3:
        return E4M3_MAX
    if dtype == E5M2:
        return E5M2_MAX
    raise ValueError(f'Not an FP8 format: {dtype}')


def scale_for(amax: jax.Array | float, dtype: Any = E4M3) -> jax.Array:
    """Per-tensor scale mapping [-amax, amax] onto the format's range."""
    return jnp.maximum(jnp.asarray(amax, jnp.float32),
                       _AMAX_FLOOR) / _format_max(dtype)


def quantize(x: jax.Array, scale: jax.Array, dtype: Any = E4M3) -> jax.Array:
    """Scales, saturates, and rounds `x` to an FP8 format."""
    limit = _format_max(dtype)
    return jnp.clip(x.astype(jnp.float32) / scale, -limit, limit).astype(dtype)


def dequantize(q: jax.Array, scale: jax.Array,
               dtype: Any = jnp.bfloat16) -> jax.Array:
    """Inverse of `quantize`, returned in `dtype`."""
    return (q.astype(jnp.float32) * scale).astype(dtype)


def fake_quantize(x: jax.Array, scale: jax.Array,
                  dtype: Any = E4M3) -> jax.Array:
    """Quantize-dequantize round trip, returned in the dtype of `x`."""
    return dequantize(quantize(x, scale, dtype), scale, x.dtype)


# ------------------------------------------------------------------------------
# FP8-aware matmul with E5M2 gradients.
# ------------------------------------------------------------------------------


@jax.custom_vjp
def fp8_matmul(x: jax.Array, w: jax.Array) -> jax.Array:
    """x @ w with E4M3 operands forward and E5M2 cotangents backward.

    Uses current (just-in-time) per-tensor scaling for every operand.

    Args:
      x: Activations, shape (..., k).
      w: Weights, shape (k, n).

    Returns:
      Float32 product of shape (..., n).
    """
    out, _ = _fp8_matmul_fwd(x, w)
    return out


def _fp8_matmul_fwd(x, w):
    xs, ws = scale_for(jnp.max(jnp.abs(x))), scale_for(jnp.max(jnp.abs(w)))
    xq, wq = fake_quantize(x, xs), fake_quantize(w, ws)
    out = jnp.matmul(xq.astype(jnp.float32), wq.astype(jnp.float32))
    return out, (xq, wq)


def _fp8_matmul_bwd(residuals, g):
    xq, wq = residuals
    gs = scale_for(jnp.max(jnp.abs(g)), E5M2)
    gq = fake_quantize(g, gs, E5M2).astype(jnp.float32)
    dx = jnp.matmul(gq, wq.astype(jnp.float32).T).astype(xq.dtype)
    x2 = xq.astype(jnp.float32).reshape(-1, xq.shape[-1])
    dw = jnp.matmul(x2.T, gq.reshape(-1, gq.shape[-1])).astype(wq.dtype)
    return dx, dw


fp8_matmul.defvjp(_fp8_matmul_fwd, _fp8_matmul_bwd)


# ------------------------------------------------------------------------------
# Dense layers.
# ------------------------------------------------------------------------------


def init_dense(key: jax.Array, in_dim: int, out_dim: int,
               bias: bool = True) -> dict[str, jax.Array]:
    """LeCun-normal initialized dense layer parameters {'w', 'b'}."""
    std = 1.0 / math.sqrt(in_dim)
    params = {'w': std * jax.random.normal(key, (in_dim, out_dim))}
    if bias:
        params['b'] = jnp.zeros((out_dim,))
    return params


def is_quantized(params: dict[str, Any]) -> bool:
    return 'w_fp8' in params


class AmaxCalibrator:
    """Records the running absolute maximum of each dense layer's input.

    Calibration must run eagerly (outside `jax.jit`), because the observed
    values are read back to the host.

    Attributes:
      amax: Mapping from layer name to the largest |input| seen so far.
    """

    def __init__(self):
        self.amax: dict[str, float] = {}

    def observe(self, name: str, x: jax.Array) -> None:
        value = float(jnp.max(jnp.abs(x)))
        self.amax[name] = max(self.amax.get(name, 0.0), value)


def dense(params: dict[str, Any], x: jax.Array, name: str = '',
          calib: AmaxCalibrator | None = None,
          fp8_training: bool = False) -> jax.Array:
    """Applies a float or FP8-quantized dense layer.

    Args:
      params: Either {'w', 'b'?} or the quantized form produced by
        `quantize_params`: {'w_fp8', 'w_scale', 'x_scale'?, 'b'?}.
      x: Input of shape (..., in_dim).
      name: Hierarchical layer name, used as the calibration key. It must
        match the parameter path (see `quantize_params`).
      calib: If given, records the input's absolute maximum under `name`.
      fp8_training: Use `fp8_matmul` (E4M3 forward, E5M2 backward) for the
        float path.

    Returns:
      Output of shape (..., out_dim) in float32.
    """
    if calib is not None:
        calib.observe(name, x)
    if is_quantized(params):
        compute = jnp.bfloat16
        if 'x_scale' in params:
            x = fake_quantize(x.astype(jnp.float32), params['x_scale'])
        w = dequantize(params['w_fp8'], params['w_scale'], compute)
        # BF16 operands with FP32 accumulation. (jnp.matmul only gained
        # `preferred_element_type` after jax 0.4.14, so use dot_general.)
        y = jax.lax.dot_general(
            x.astype(compute), w, (((x.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)
    elif fp8_training:
        y = fp8_matmul(x, params['w'])
    else:
        y = jnp.matmul(x, params['w'])
    if 'b' in params:
        y = y + params['b']
    return y


def child(prefix: str, *keys: Any) -> str:
    """Joins a layer-name prefix with sub-keys, e.g. child('a', 0, 'q')."""
    parts = [prefix] if prefix else []
    return PATH_SEP.join(parts + [str(k) for k in keys])


# ------------------------------------------------------------------------------
# Post-training quantization of parameter trees.
# ------------------------------------------------------------------------------


def _quantize_tree(node: Any, path: str, calib: AmaxCalibrator | None,
                   dtype: Any) -> Any:
    if isinstance(node, dict):
        w = node.get('w')
        if w is not None and getattr(w, 'ndim', 0) == 2:
            w_scale = scale_for(jnp.max(jnp.abs(w)), dtype)
            out = {'w_fp8': quantize(w, w_scale, dtype), 'w_scale': w_scale}
            if 'b' in node:
                out['b'] = node['b']
            if calib is not None and path in calib.amax:
                out['x_scale'] = scale_for(calib.amax[path], dtype)
            return out
        return {k: _quantize_tree(v, child(path, k), calib, dtype)
                for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return type(node)(_quantize_tree(v, child(path, i), calib, dtype)
                          for i, v in enumerate(node))
    return node


def quantize_params(params: Any, calib: AmaxCalibrator | None = None,
                    dtype: Any = E4M3) -> Any:
    """Post-training quantization of every 2D dense weight in a tree.

    Dense layers are dicts with a 2D 'w'. Their name is the '/'-joined path
    of dict keys and list indices from the root, which is the `name` the
    model passes to `dense` during calibration. Other leaves (biases, norms,
    complex spectral weights) are kept in full precision.

    Args:
      params: Nested dicts / lists of float parameters.
      calib: Optional calibrator holding activation absolute maxima.
      dtype: FP8 format for weights and activations (E4M3 by default).

    Returns:
      A parameter tree in which every dense layer is quantized.
    """
    return _quantize_tree(params, '', calib, dtype)


def dense_weight_bytes(params: Any, as_dtype: Any | None = None) -> int:
    """Bytes of all dense weights, optionally as if stored in `as_dtype`.

    Args:
      params: Float or quantized parameter tree.
      as_dtype: If given, count float weights at this dtype's width (e.g.
        jnp.bfloat16 for the BF16 baseline of report Table 5).

    Returns:
      Total bytes of 'w' / 'w_fp8' leaves (scales excluded).
    """
    total = 0
    leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    for path, leaf in leaves:
        key = getattr(path[-1], 'key', None)
        if key == 'w_fp8':
            total += leaf.size * leaf.dtype.itemsize
        elif key == 'w' and leaf.ndim == 2:
            width = (np.dtype(as_dtype).itemsize if as_dtype is not None
                     else leaf.dtype.itemsize)
            total += leaf.size * width
    return int(total)
