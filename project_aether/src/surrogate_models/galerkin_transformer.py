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
"""Physics-informed Galerkin / Fourier Transformer surrogate (report Part IV).

The surrogate maps a flow field Q^n on a grid to Q^{n+1}. Every grid point is
a token carrying its conserved variables and coordinates. The architecture
combines two operator-learning ideas:

  * Galerkin attention (S. Cao, "Choose a Transformer: Fourier or Galerkin",
    NeurIPS 2021): softmax-free attention z = Q (K^T V) / n with layer
    normalization applied to K and V. It is linear in the number of tokens,
    which is what lets the block-sparse and FP8 numerics of Part V apply to
    the attention stage. The Fourier variant z = (Q K^T) V / n normalizes
    Q and K instead (quadratic, kept for comparison).
  * Spectral convolution (Z. Li et al., "Fourier Neural Operator for
    Parametric Partial Differential Equations", ICLR 2021): a learned
    multiplication of the lowest Fourier modes of the field.

The model is trained against data plus a PDE-residual penalty
(`physics_informed_loss`) and predicts the increment Q^{n+1} - Q^n in
normalized units.

All dense layers go through `quantization_fp8.dense`, so the parameter tree
can be post-training quantized to FP8. Parameter names follow
`quantization_fp8.child` so calibration keys match parameter paths.

Typical usage example:

  cfg = galerkin_transformer.GalerkinConfig()
  params = galerkin_transformer.init_params(key, cfg)
  q_next = galerkin_transformer.apply(params, q_now, norm, cfg)
"""

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from src.surrogate_models import quantization_fp8

GALERKIN = 'galerkin'
FOURIER = 'fourier'

_child = quantization_fp8.child
_dense = quantization_fp8.dense


class GalerkinConfig(NamedTuple):
    """Hyper-parameters of the surrogate.

    Attributes:
      in_channels: Physical channels of the input field.
      out_channels: Physical channels of the output field.
      width: Token embedding width.
      num_layers: Attention + feed-forward blocks.
      num_heads: Attention heads.
      attention: GALERKIN (linear) or FOURIER (quadratic).
      modes: Fourier modes kept per axis in the spectral convolution.
      ffn_multiplier: Feed-forward hidden width / width.
    """
    in_channels: int = 4
    out_channels: int = 4
    width: int = 64
    num_layers: int = 4
    num_heads: int = 4
    attention: str = GALERKIN
    modes: int = 8
    ffn_multiplier: int = 2


class Normalizer(NamedTuple):
    """Per-channel affine normalization of the physical fields.

    Attributes:
      mean: Channel means, shape (C,).
      std: Channel standard deviations, shape (C,).
    """
    mean: jax.Array
    std: jax.Array

    @classmethod
    def fit(cls, fields: jax.Array) -> 'Normalizer':
        """Fits to fields of shape (batch, C, ny, nx)."""
        mean = jnp.mean(fields, axis=(0, 2, 3))
        std = jnp.std(fields, axis=(0, 2, 3)) + 1e-8
        return cls(mean, std)

    def encode(self, fields: jax.Array) -> jax.Array:
        return (fields - self.mean[:, None, None]) / self.std[:, None, None]

    def decode(self, fields: jax.Array) -> jax.Array:
        return fields * self.std[:, None, None] + self.mean[:, None, None]


def _layer_norm(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps)


def _init_layer_norm(dim: int) -> dict[str, jax.Array]:
    return {'scale': jnp.ones((dim,)), 'bias': jnp.zeros((dim,))}


def _apply_layer_norm(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    return _layer_norm(x) * params['scale'] + params['bias']


def init_params(key: jax.Array, cfg: GalerkinConfig) -> dict[str, Any]:
    """Initializes the surrogate parameters.

    Args:
      key: PRNG key.
      cfg: Model configuration.

    Returns:
      Nested parameter dict.
    """
    keys = iter(jax.random.split(key, 8 + 8 * cfg.num_layers))
    width, hidden = cfg.width, cfg.width * cfg.ffn_multiplier
    layers = []
    for _ in range(cfg.num_layers):
        layers.append({
            'attn': {
                'q': quantization_fp8.init_dense(next(keys), width, width),
                'k': quantization_fp8.init_dense(next(keys), width, width),
                'v': quantization_fp8.init_dense(next(keys), width, width),
                'o': quantization_fp8.init_dense(next(keys), width, width),
            },
            'norm': _init_layer_norm(width),
            'ffn': {
                'up': quantization_fp8.init_dense(next(keys), width, hidden),
                'down': quantization_fp8.init_dense(next(keys), hidden, width),
            },
        })
    m = cfg.modes
    spectral_scale = 1.0 / (width * width)
    k_re, k_im = jax.random.split(next(keys))
    return {
        'lift': quantization_fp8.init_dense(next(keys), cfg.in_channels + 2,
                                            width),
        'layers': layers,
        'spectral': {
            # Weights for the low-ky and high-ky (negative) mode blocks.
            're': spectral_scale * jax.random.uniform(
                k_re, (2, width, width, m, m)),
            'im': spectral_scale * jax.random.uniform(
                k_im, (2, width, width, m, m)),
            'pointwise': quantization_fp8.init_dense(next(keys), width, width),
        },
        'project': {
            'hidden': quantization_fp8.init_dense(next(keys), width, width),
            # Zero-initialized so the untrained surrogate is the identity
            # map Q^{n+1} = Q^n: its predictions are physical states and the
            # PDE residual is finite from the first step.
            'out': {'w': jnp.zeros((width, cfg.out_channels)),
                    'b': jnp.zeros((cfg.out_channels,))},
        },
    }


def _split_heads(x: jax.Array, num_heads: int) -> jax.Array:
    b, n, d = x.shape
    return x.reshape(b, n, num_heads, d // num_heads).transpose(0, 2, 1, 3)


def _merge_heads(x: jax.Array) -> jax.Array:
    b, h, n, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, n, h * d)


def galerkin_attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """Softmax-free Galerkin attention Q (LN(K)^T LN(V)) / n, O(n d^2).

    Args:
      q: Queries of shape (batch, heads, n, d).
      k: Keys of shape (batch, heads, n, d).
      v: Values of shape (batch, heads, n, d).

    Returns:
      Attention output of shape (batch, heads, n, d).
    """
    n = q.shape[-2]
    kv = jnp.einsum('bhnd,bhne->bhde', _layer_norm(k), _layer_norm(v))
    return jnp.einsum('bhnd,bhde->bhne', q, kv) / n


def fourier_attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """Softmax-free Fourier-type attention (LN(Q) LN(K)^T) V / n, O(n^2 d)."""
    n = q.shape[-2]
    scores = jnp.einsum('bhnd,bhmd->bhnm', _layer_norm(q), _layer_norm(k))
    return jnp.einsum('bhnm,bhmd->bhnd', scores, v) / n


def _attention_block(params: dict[str, Any], x: jax.Array, prefix: str,
                     cfg: GalerkinConfig,
                     calib: quantization_fp8.AmaxCalibrator | None
                     ) -> jax.Array:
    p = params['attn']
    q = _dense(p['q'], x, _child(prefix, 'attn', 'q'), calib)
    k = _dense(p['k'], x, _child(prefix, 'attn', 'k'), calib)
    v = _dense(p['v'], x, _child(prefix, 'attn', 'v'), calib)
    q, k, v = (_split_heads(t, cfg.num_heads) for t in (q, k, v))
    if cfg.attention == GALERKIN:
        z = galerkin_attention(q, k, v)
    elif cfg.attention == FOURIER:
        z = fourier_attention(q, k, v)
    else:
        raise ValueError(f'Unknown attention type: {cfg.attention!r}')
    return _dense(p['o'], _merge_heads(z), _child(prefix, 'attn', 'o'), calib)


def spectral_conv2d(params: dict[str, jax.Array], x: jax.Array,
                    modes: int) -> jax.Array:
    """FNO spectral convolution on the lowest `modes` x `modes` frequencies.

    Args:
      params: {'re', 'im'} weights of shape (2, width, width, modes, modes).
      x: Field of shape (batch, ny, nx, width).
      modes: Modes kept along each axis; must satisfy
        modes <= ny // 2 and modes <= nx // 2 + 1.

    Returns:
      Field of shape (batch, ny, nx, width).
    """
    _, ny, nx, _ = x.shape
    x_hat = jnp.fft.rfft2(x, axes=(1, 2))  # (b, ny, nx // 2 + 1, w)
    weights = params['re'] + 1j * params['im']
    low = jnp.einsum('byxi,ioyx->byxo', x_hat[:, :modes, :modes],
                     weights[0])
    high = jnp.einsum('byxi,ioyx->byxo', x_hat[:, -modes:, :modes],
                      weights[1])
    out_hat = jnp.zeros(x_hat.shape[:3] + (weights.shape[2],),
                        dtype=x_hat.dtype)
    out_hat = out_hat.at[:, :modes, :modes].set(low)
    out_hat = out_hat.at[:, -modes:, :modes].set(high)
    return jnp.fft.irfft2(out_hat, s=(ny, nx), axes=(1, 2))


def grid_coordinates(ny: int, nx: int) -> jax.Array:
    """Cell-center coordinates in [0, 1]^2, shape (ny, nx, 2)."""
    ys = (jnp.arange(ny) + 0.5) / ny
    xs = (jnp.arange(nx) + 0.5) / nx
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    return jnp.stack([xx, yy], axis=-1)


def apply(params: dict[str, Any], fields: jax.Array, norm: Normalizer,
          cfg: GalerkinConfig,
          calib: quantization_fp8.AmaxCalibrator | None = None) -> jax.Array:
    """Predicts the next flow field.

    Args:
      params: Float or FP8-quantized parameters.
      fields: Current conserved fields, shape (batch, C, ny, nx).
      norm: Channel normalizer of the training data.
      cfg: Model configuration.
      calib: Optional activation calibrator (eager execution only).

    Returns:
      Predicted conserved fields at the next step, shape (batch, C, ny, nx).
    """
    b, _, ny, nx = fields.shape
    z = norm.encode(fields).transpose(0, 2, 3, 1)  # (b, ny, nx, C)
    coords = jnp.broadcast_to(grid_coordinates(ny, nx), (b, ny, nx, 2))
    tokens = jnp.concatenate([z, coords], axis=-1).reshape(b, ny * nx, -1)

    x = _dense(params['lift'], tokens, 'lift', calib)
    for i, layer in enumerate(params['layers']):
        prefix = _child('layers', i)
        x = x + _attention_block(layer, x, prefix, cfg, calib)
        h = _apply_layer_norm(layer['norm'], x)
        h = jax.nn.gelu(_dense(layer['ffn']['up'], h,
                               _child(prefix, 'ffn', 'up'), calib))
        x = x + _dense(layer['ffn']['down'], h,
                       _child(prefix, 'ffn', 'down'), calib)

    grid = x.reshape(b, ny, nx, cfg.width)
    spec = params['spectral']
    grid = grid + jax.nn.gelu(
        spectral_conv2d(spec, grid, cfg.modes) +
        _dense(spec['pointwise'], grid, 'spectral/pointwise', calib))

    h = jax.nn.gelu(_dense(params['project']['hidden'], grid,
                           'project/hidden', calib))
    delta = _dense(params['project']['out'], h, 'project/out', calib)
    z_next = z + delta  # Residual prediction in normalized units.
    return norm.decode(z_next.transpose(0, 3, 1, 2))


def num_parameters(params: Any) -> int:
    return sum(math.prod(leaf.shape)
               for leaf in jax.tree_util.tree_leaves(params))
