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
"""Decode-phase PagedAttention kernel (Triton 2.0).

One program handles one (sequence, head) pair. It walks the sequence's block
table, loads a whole block of keys and values per iteration, and maintains
the FlashAttention running maximum and normalizer (online softmax), so the
scores never leave registers (T. Dao, FlashAttention-2, arXiv:2307.08691).
The cache layout is the one of `src.distributed_infra.kv_paged_attention`:
(num_blocks, block_size, num_heads, head_dim) per layer.

Typical usage example:

  out = paged_attention_decode(q, k_cache, v_cache, block_tables, lens)
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_kernel(
    q_ptr, k_ptr, v_ptr, tables_ptr, lens_ptr, out_ptr,
    stride_qb, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_tb,
    stride_ob, stride_oh,
    scale,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, HEAD_DIM)
    slots = tl.arange(0, BLOCK_SIZE)
    q = tl.load(q_ptr + seq * stride_qb + head * stride_qh + dims)
    q = q.to(tl.float32)
    length = tl.load(lens_ptr + seq)
    num_blocks = (length + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Loop-carried statistics must be tensors of a fixed type.
    neg_inf = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float('inf')
    m_i = tl.max(neg_inf, axis=0)
    l_i = tl.sum(tl.zeros([BLOCK_SIZE], dtype=tl.float32), axis=0)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for b in range(0, num_blocks):
        block = tl.load(tables_ptr + seq * stride_tb + b)
        valid = b * BLOCK_SIZE + slots < length
        offs = (block * stride_kb + slots[:, None] * stride_ks +
                head * stride_kh + dims[None, :])
        k = tl.load(k_ptr + offs, mask=valid[:, None], other=0.0)
        scores = tl.sum(k.to(tl.float32) * q[None, :], axis=1) * scale
        scores = tl.where(valid, scores, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        v = tl.load(v_ptr + offs, mask=valid[:, None], other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
    tl.store(out_ptr + seq * stride_ob + head * stride_oh + dims, acc / l_i)


def paged_attention_decode(q: torch.Tensor, k_cache: torch.Tensor,
                           v_cache: torch.Tensor, block_tables: torch.Tensor,
                           context_lens: torch.Tensor) -> torch.Tensor:
    """Attention of one new token per sequence over its paged history.

    Args:
      q: Queries, (B, H, D) CUDA tensor; D a power of two.
      k_cache: Keys of one layer, (num_blocks, block_size, H, D).
      v_cache: Values of one layer, same shape as k_cache.
      block_tables: int32 physical block ids, (B, max_blocks).
      context_lens: int32 history length per sequence (>= 1), (B,).

    Returns:
      Attention output, float32 (B, H, D).
    """
    b, h, d = q.shape
    block_size = k_cache.shape[1]
    out = torch.empty((b, h, d), dtype=torch.float32, device=q.device)
    _paged_attention_kernel[(b, h)](
        q, k_cache, v_cache, block_tables, context_lens, out,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        block_tables.stride(0),
        out.stride(0), out.stride(1),
        1.0 / math.sqrt(d),
        BLOCK_SIZE=block_size, HEAD_DIM=d, num_warps=4)
    return out


def reference_paged_attention(q: torch.Tensor, k_cache: torch.Tensor,
                              v_cache: torch.Tensor,
                              block_tables: torch.Tensor,
                              context_lens: torch.Tensor) -> torch.Tensor:
    """PyTorch reference of `paged_attention_decode` (also runs on CPU)."""
    out = []
    block_size = k_cache.shape[1]
    for i in range(q.shape[0]):
        n = int(context_lens[i])
        blocks = block_tables[i, :(n + block_size - 1) // block_size].long()
        k = k_cache[blocks].reshape(-1, *k_cache.shape[2:])[:n].float()
        v = v_cache[blocks].reshape(-1, *v_cache.shape[2:])[:n].float()
        scores = torch.einsum('hd,shd->hs', q[i].float(), k)
        weights = torch.softmax(scores / math.sqrt(q.shape[-1]), dim=-1)
        out.append(torch.einsum('hs,shd->hd', weights, v))
    return torch.stack(out)
