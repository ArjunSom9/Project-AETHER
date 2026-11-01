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
"""Autoregressive telemetry surrogate served by the disaggregated system.

The digital twin rolls the reconstructed flight state forward in time. Each
token is one telemetry / reduced-state vector (e.g. probe readings plus the
leading modes of the reconstructed field). A causal Transformer predicts the
next token from the history, which makes serving the same two-phase workload
as language-model inference (report Part VI):

  * prefill: encode the measured history (compute bound, many tokens);
  * decode: roll the state forward one token per step (memory-bandwidth
    bound, reads the whole key/value history every step).

`forward` processes a chunk of new tokens for a batch of sequences against a
`kv_paged_attention` cache. The same function therefore serves chunked
prefill (report 6.1, SARATHI-style chunks) and single-token decode.

The attention and MLP weights can be laid out Megatron-style across a
'tensor' mesh axis (`tensor_parallel_specs`); the caller attaches the
shardings and GSPMD propagates them through `forward`.

Typical usage example:

  params = trajectory_transformer.init_params(key, cfg)
  out, k, v = trajectory_transformer.forward(params, k, v, tokens,
                                             positions, slots, tables,
                                             context_lens, cfg)
"""

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from src.distributed_infra import kv_paged_attention
from src.surrogate_models import quantization_fp8

_dense = quantization_fp8.dense
_init_dense = quantization_fp8.init_dense


class TrajectoryConfig(NamedTuple):
    """Hyper-parameters of the trajectory surrogate.

    Attributes:
      feature_dim: Width of one telemetry token.
      embed_dim: Model width.
      num_layers: Transformer layers.
      num_heads: Attention heads.
    """
    feature_dim: int = 16
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


def init_params(key: jax.Array, cfg: TrajectoryConfig) -> dict[str, Any]:
    """Initializes the decoder-only trajectory Transformer."""
    d = cfg.embed_dim
    keys = iter(jax.random.split(key, 2 + 4 * cfg.num_layers))
    layers = []
    for _ in range(cfg.num_layers):
        layers.append({
            'ln1': jnp.ones((d,)),
            'qkv': _init_dense(next(keys), d, 3 * d),
            'o': _init_dense(next(keys), d, d),
            'ln2': jnp.ones((d,)),
            'up': _init_dense(next(keys), d, 4 * d),
            'down': _init_dense(next(keys), 4 * d, d),
        })
    return {'embed': _init_dense(next(keys), cfg.feature_dim, d),
            'layers': layers,
            'head': _init_dense(next(keys), d, cfg.feature_dim)}


def tensor_parallel_specs(params: dict[str, Any]) -> Any:
    """Megatron-LM tensor-parallel PartitionSpecs for the 'tensor' axis.

    QKV and MLP-up are column parallel (output features split), attention
    output and MLP-down are row parallel (input features split), so each layer
    needs one all-reduce after attention and one after the MLP (Shoeybi et
    al., arXiv:1909.08053). Embeddings, norms, and the head are replicated.

    Args:
      params: Parameter tree from `init_params`.

    Returns:
      A tree of PartitionSpecs with the structure of `params`.
    """
    spec = jax.sharding.PartitionSpec
    column = {'w': spec(None, 'tensor'), 'b': spec('tensor')}
    row = {'w': spec('tensor', None), 'b': spec()}
    replicated_dense = {'w': spec(), 'b': spec()}
    layer = {'ln1': spec(), 'qkv': column, 'o': row, 'ln2': spec(),
             'up': column, 'down': row}
    return {'embed': replicated_dense,
            'layers': [layer for _ in params['layers']],
            'head': replicated_dense}


def _rms_norm(x: jax.Array, scale: jax.Array) -> jax.Array:
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) +
                             1e-6) * scale


def sinusoidal_embedding(positions: jax.Array, dim: int) -> jax.Array:
    """Sinusoidal position encoding of shape positions.shape + (dim,)."""
    half = dim // 2
    freq = jnp.exp(-math.log(10000.0) * jnp.arange(half) / half)
    angle = positions[..., None].astype(jnp.float32) * freq
    return jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)


def apply_chunk(params: dict[str, Any], k_cache: jax.Array,
                v_cache: jax.Array, tokens: jax.Array, positions: jax.Array,
                slots: jax.Array, block_tables: jax.Array,
                context_lens: jax.Array, cfg: TrajectoryConfig
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Processes a chunk of new tokens and appends their K/V to the cache.

    This is the un-jitted model; `forward` is its single-device jit, and
    `model_runner.sharded_forward` jits it with explicit pool shardings.

    Args:
      params: Model parameters (optionally tensor-parallel sharded).
      k_cache: Paged keys, (num_layers, num_blocks, block_size, H, Dh).
      v_cache: Paged values, same shape.
      tokens: New tokens, (B, T, feature_dim).
      positions: Absolute positions of the new tokens, (B, T).
      slots: Flat cache slot of every new token, (B, T); padding tokens map
        to slots of `kv_paged_attention.NULL_BLOCK`.
      block_tables: Physical blocks per sequence, (B, max_blocks).
      context_lens: Tokens in the cache after this chunk, (B,).
      cfg: Model configuration.

    Returns:
      A tuple (predictions, k_cache, v_cache); predictions has shape
      (B, T, feature_dim) and holds the predicted next token after each
      input token.
    """
    b, t, _ = tokens.shape
    h, dh = cfg.num_heads, cfg.head_dim
    num_blocks, block_size = k_cache.shape[1:3]
    flat_slots = slots.reshape(-1)
    x = _dense(params['embed'], tokens) + sinusoidal_embedding(
        positions, cfg.embed_dim)
    for i, layer in enumerate(params['layers']):
        qkv = _dense(layer['qkv'], _rms_norm(x, layer['ln1']))
        q, k, v = (a.reshape(b, t, h, dh) for a in jnp.split(qkv, 3, axis=-1))
        k_flat = k_cache[i].reshape(num_blocks * block_size, h, dh)
        v_flat = v_cache[i].reshape(num_blocks * block_size, h, dh)
        k_flat = k_flat.at[flat_slots].set(k.reshape(b * t, h, dh))
        v_flat = v_flat.at[flat_slots].set(v.reshape(b * t, h, dh))
        k_layer = k_flat.reshape(num_blocks, block_size, h, dh)
        v_layer = v_flat.reshape(num_blocks, block_size, h, dh)
        k_cache = k_cache.at[i].set(k_layer)
        v_cache = v_cache.at[i].set(v_layer)
        attn = kv_paged_attention.paged_attention(
            q, k_layer, v_layer, block_tables, context_lens, positions)
        x = x + _dense(layer['o'], attn.reshape(b, t, h * dh))
        hidden = jax.nn.gelu(_dense(layer['up'], _rms_norm(x, layer['ln2'])))
        x = x + _dense(layer['down'], hidden)
    return tokens + _dense(params['head'], x), k_cache, v_cache


# Single-device jit; the caches are donated and updated in place.
forward = jax.jit(apply_chunk, static_argnames='cfg', donate_argnums=(1, 2))
