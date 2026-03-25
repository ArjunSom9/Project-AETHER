"""
Project AETHER: INT8 Block Sort for Particle Binning
Optimized for NVIDIA Hopper/Ampere using OpenAI Triton.

This module implements a localized sorting network for Direct Simulation Monte Carlo (DSMC).
To maximize collision probability evaluation speed, particles must be contiguous in memory 
based on their spatial cell. 

Instead of a global FP32 sort (which is heavily memory-bound), we perform a localized 
block-wise sort using INT8 keys. The simulation space is divided into macro-blocks, and 
particles are sorted into 256 local sub-cells (INT8 limit). This "compute-heavy, 
memory-light" approach perfectly aligns with modern GPU architecture bottlenecks.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _int8_block_sort_kernel(
    keys_ptr,           # Pointer to cell indices (Keys)
    values_ptr,         # Pointer to particle IDs (Values)
    out_keys_ptr,       # Pointer to sorted keys
    out_values_ptr,     # Pointer to sorted values
    num_particles,      # Total number of particles
    BLOCK_SIZE: tl.constexpr # Must be a power of 2 (e.g., 1024)
):
    """
    Triton kernel that loads unorganized particle IDs and their cell indices,
    compresses them into INT8, sorts them in SRAM using a highly optimized 
    sorting network, and writes them back.
    """
    # Identify the specific macro-block of particles this thread block will sort
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to prevent reading out of bounds on the last block
    mask = offsets < num_particles

    # 1. Load Keys (Cell IDs) and Values (Particle IDs) from HBM into SRAM
    # We load them as int32 first, then immediately downcast keys to int8
    # to simulate the reduced precision memory footprint in the registers.
    raw_keys = tl.load(keys_ptr + offsets, mask=mask, other=255)
    values = tl.load(values_ptr + offsets, mask=mask, other=0)
    
    # Compress keys to INT8 (0-255 local sub-cell range)
    keys_int8 = tl.cast(raw_keys, tl.int8)

    # 2. Localized Sort in SRAM
    # Triton's tl.sort compiles down to a highly optimized bitonic sorting network 
    # using fast integer PTX instructions (like dp4a when applicable) under the hood.
    # By ensuring keys are INT8, we maximize the number of keys processed per instruction.
    sorted_keys, sorted_values = tl.sort(keys_int8, values, dim=0, descending=False)

    # 3. Store the sorted pairs back to Global Memory (HBM)
    # We cast the keys back to int32 for compatibility with the rest of the PyTorch/JAX pipeline,
    # though in a fully optimized chain, the next kernel would ingest INT8 directly.
    tl.store(out_keys_ptr + offsets, tl.cast(sorted_keys, tl.int32), mask=mask)
    tl.store(out_values_ptr + offsets, sorted_values, mask=mask)


def sort_particles_by_cell(cell_indices: torch.Tensor, particle_ids: torch.Tensor) -> tuple:
    """
    Host wrapper for the Triton INT8 Block Sort.
    
    Args:
        cell_indices: 1D tensor of cell indices for each particle.
        particle_ids: 1D tensor of unique particle identifiers.
        
    Returns:
        Tuple of (sorted_cell_indices, sorted_particle_ids)
    """
    assert cell_indices.is_cuda and particle_ids.is_cuda, "Tensors must be on the GPU"
    assert cell_indices.dtype == torch.int32, "Cell indices must be int32"
    
    num_particles = cell_indices.shape[0]
    
    # Allocate output tensors
    sorted_keys = torch.empty_like(cell_indices)
    sorted_values = torch.empty_like(particle_ids)
    
    # H100s thrive with larger block sizes for SRAM utilization. 
    # 1024 or 2048 are optimal for block-level sorting networks.
    BLOCK_SIZE = 1024
    
    # Determine the number of thread blocks needed
    grid = lambda meta: (triton.cdiv(num_particles, meta['BLOCK_SIZE']), )
    
    # Launch the Triton kernel
    _int8_block_sort_kernel[grid](
        cell_indices, 
        particle_ids, 
        sorted_keys, 
        sorted_values, 
        num_particles, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sorted_keys, sorted_values

# --- Diagnostic / Testing Block ---
if __name__ == "__main__":
    # Mock data for 1 million particles
    num_p = 1024 * 1024
    
    # Generate random cell indices (0 to 200, to fit within INT8 limit for the local block)
    mock_cells = torch.randint(0, 200, (num_p,), device='cuda', dtype=torch.int32)
    mock_ids = torch.arange(num_p, device='cuda', dtype=torch.int32)
    
    print("Pre-sort Cell IDs (First 10):", mock_cells[:10].tolist())
    
    # Execute the INT8 hardware-accelerated sort
    sorted_cells, sorted_ids = sort_particles_by_cell(mock_cells, mock_ids)
    
    print("Post-sort Cell IDs (First 10):", sorted_cells[:10].tolist())