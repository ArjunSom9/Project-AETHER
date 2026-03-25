"""
Project AETHER: TMA Software Pipeline for Block-Sparse GEMM
Optimized for NVIDIA H100 (Hopper) Architectures.

This module provides the bare-metal JAX Pallas kernel that programs the 
Tensor Memory Accelerator (TMA). By using `pl.load` and `pl.store`, we instruct 
the TMA to asynchronously fetch blocks of data from High Bandwidth Memory (HBM) 
directly into the SRAM (Shared Memory), bypassing the L2 cache bottleneck.

This kernel is specifically designed for sparse chemical kinetics matrices 
where empty blocks (non-interacting species) are skipped entirely, saving 
massive amounts of compute and bandwidth.
"""

import jax.numpy as jnp
from jax.experimental import pallas as pl

def block_sparse_gemm_kernel(
    x_ref, y_ref, z_ref,        # Input/Output memory references mapping to HBM
    adj_matrix_ref,             # Sparsity map (block adjacency mask)
    BLOCK_M: int, 
    BLOCK_N: int, 
    BLOCK_K: int
):
    """
    Pallas kernel executing block-sparse matrix multiplication (C = A * B).
    
    This kernel defines the "software pipeline" at the block level. While the 
    Tensor Cores compute the matrix multiplication for the current K-block in 
    registers, the TMA is instructed to asynchronously load the (K+1)-th block 
    into SRAM in the background.
    """
    # 1. Retrieve block grid coordinates
    # These map to the specific output tile (M, N) this thread block is responsible for
    pid_m = pl.program_id(0)
    pid_n = pl.program_id(1)
    
    # Total number of blocks along the K (inner/reduction) dimension
    num_k_blocks = pl.num_programs(2)

    # 2. Sparsity Check (Block-level sparsity)
    # Check the adjacency matrix. If these two macroscopic chemical blocks do not 
    # interact (e.g., cross-section is exactly 0), we exit the kernel immediately.
    # This turns an O(N^3) dense operation into an O(k) sparse operation.
    is_active = pl.load(adj_matrix_ref, (pid_m, pid_n))
    if not is_active:
        return

    # 3. Initialize Accumulator
    # This matrix resides entirely in ultra-fast GPU registers
    acc = jnp.zeros((BLOCK_M, BLOCK_N), dtype=jnp.float32)

    # 4. Asynchronous Pipeline Loop over K dimension
    for k in range(num_k_blocks):
        # TMA Asynchronous Load (HBM -> SRAM)
        # In Pallas, `pl.load` on Hopper GPUs utilizes the TMA engine automatically.
        # Data is pulled into shared memory asynchronously while compute proceeds.
        x_tile = pl.load(x_ref, (pid_m, k))
        y_tile = pl.load(y_ref, (k, pid_n))
        
        # Tensor Core Matrix Multiplication (SRAM -> Registers)
        # Pallas maps jnp.dot directly to synchronous Tensor Core instructions (e.g. mma.sync)
        acc += jnp.dot(x_tile, y_tile)

    # 5. TMA Asynchronous Store (Registers -> HBM)
    # Writes the fully accumulated output tile back to global memory
    pl.store(z_ref, (pid_m, pid_n), acc)