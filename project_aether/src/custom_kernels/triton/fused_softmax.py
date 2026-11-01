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
"""Row-wise fused softmax kept entirely in SRAM (Triton 2.0).

An unfused softmax makes separate passes for the row maximum, the
subtraction and exponential, the sum, and the division, each one a round trip
to HBM. This kernel loads each row once, computes the numerically stable
softmax in registers, and stores it once. The same pattern is the building
block of FlashAttention-style attention kernels, which keep the softmax
statistics on chip (report Part IV, 4.2). The GPU path of the relaxed DSMC
collision (Gumbel-softmax over {collide, pass}) applies it row-wise.

Typical usage example:

  probs = fused_softmax(logits)  # (rows, cols) float32 CUDA tensor.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask,
                other=-float('inf'))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    y = numerator / tl.sum(numerator, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension of a 2D CUDA tensor.

    Args:
      x: Logits of shape (rows, cols), float32 on a CUDA device.

    Returns:
      Probabilities with the same shape and dtype.

    Raises:
      ValueError: If `x` is not a 2D CUDA tensor.
    """
    if not x.is_cuda or x.dim() != 2:
        raise ValueError('fused_softmax expects a 2D CUDA tensor.')
    rows, cols = x.shape
    y = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    num_warps = 4 if block <= 2048 else 8
    _softmax_kernel[(rows,)](y, x, x.stride(0), y.stride(0), cols,
                             BLOCK=block, num_warps=num_warps)
    return y
