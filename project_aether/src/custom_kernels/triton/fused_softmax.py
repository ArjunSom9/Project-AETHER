"""
Project AETHER: Fused Softmax for Collision Sampling
Optimized for NVIDIA Hopper/Ampere using OpenAI Triton.

In standard DSMC algorithms, evaluating the Maxwell-Boltzmann distribution 
or calculating collision pair probabilities requires multiple passes over the data 
(max, subtract, exp, sum, divide). In a legacy code, this generates 5 roundtrips 
to High Bandwidth Memory (HBM), creating a massive "memory wall".

This module implements a Fused Softmax kernel. By keeping the intermediate 
probability distributions entirely within the GPU's SRAM/Registers, we reduce 
memory bandwidth requirements by 4x. This is critical when evaluating 
millions of potential collision pairs simultaneously in the shock layer.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_softmax_kernel(
    output_ptr,        # Pointer to the output probabilities
    input_ptr,         # Pointer to the input logits (e.g., relative velocities/cross-sections)
    input_row_stride,  # Stride to jump to the next row (particle/cell)
    output_row_stride, # Stride for output rows
    n_cols,            # Number of columns (e.g., number of candidate collision partners)
    BLOCK_SIZE: tl.constexpr # Tile size, must be a power of 2 greater than n_cols
):
    """
    Computes the softmax of a row of input logits directly in SRAM.
    Designed to be invoked where each thread block handles a single row 
    (e.g., one particle's probability distribution over local cell candidates).
    """
    # The rows of the softmax are independent, so we map each row to a program ID.
    row_idx = tl.program_id(0)
    
    # Calculate the starting memory address for this specific row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Generate an array of offsets for the elements in the row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent reading out of bounds if n_cols < BLOCK_SIZE
    mask = col_offsets < n_cols
    
    # Load the row into SRAM. Out-of-bounds elements are padded with -infinity
    # so they evaluate to 0 after the exp() operation.
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # 1. Safe Softmax: Subtract the maximum value in the row to prevent numerical 
    # overflow (FP16/FP32 precision limits) when computing exp().
    row_max = tl.max(row, axis=0)
    safe_row = row - row_max
    
    # 2. Exponentiate the safe values
    numerator = tl.exp(safe_row)
    
    # 3. Sum the exponentials along the row to get the normalization denominator
    denominator = tl.sum(numerator, axis=0)
    
    # 4. Divide to get the final probability distribution
    softmax_output = numerator / denominator
    
    # Store the final probabilities back to HBM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)

def apply_fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Host wrapper for the Triton Fused Softmax kernel.
    
    Args:
        x: A 2D torch.Tensor of logits [num_particles, num_candidates].
           For example, collision propensities between a particle and its neighbors.
           
    Returns:
        A 2D torch.Tensor of normalized probabilities.
    """
    assert x.is_cuda, "Input tensor must be on the GPU"
    assert x.dim() == 2, "Input tensor must be 2D [rows, cols]"
    
    n_rows, n_cols = x.shape
    
    # Allocate the output tensor with the same shape and type
    y = torch.empty_like(x)
    
    # Triton requires the BLOCK_SIZE to be a power of 2. 
    # We find the smallest power of 2 that is >= n_cols.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Grid limits: 1 thread block per row
    grid = (n_rows, )
    
    # Launch the kernel
    # We pass the strides of the tensors so the kernel knows how memory is laid out
    _fused_softmax_kernel[grid](
        y, 
        x, 
        x.stride(0), 
        y.stride(0), 
        n_cols, 
        num_warps=8, # 8 warps (256 threads) is usually optimal for larger rows
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y

# --- Diagnostic / Testing Block ---
if __name__ == "__main__":
    # Mock collision logits for 100,000 particles, each considering 128 neighbors
    n_particles = 100_000
    n_neighbors = 128
    
    # Random logits representing e.g., cross_section * relative_velocity
    mock_logits = torch.randn(n_particles, n_neighbors, device='cuda')
    
    # Calculate probabilities using custom Triton kernel
    probabilities = apply_fused_softmax(mock_logits)
    
    # Verify mathematically (sum of probabilities for each particle should be 1.0)
    row_sums = probabilities.sum(dim=1)
    
    print(f"Computed Softmax for {n_particles} particles.")
    print(f"Sample Row 0 Sum: {row_sums[0].item():.5f} (Expected: ~1.00000)")
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), "Softmax output did not normalize to 1"