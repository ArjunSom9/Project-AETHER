"""
Project AETHER: Differentiable Direct Simulation Monte Carlo (DSMC)
Optimized for JAX Autodiff and Triton Hardware Acceleration.

This module provides the physics engine interface for solving the Boltzmann equation
in the rarefied hypersonic regime. It contains two parallel methodologies:
1. High-Performance execution via OpenAI Triton kernels.
2. Differentiable execution via JAX using Gumbel-Softmax continuous relaxation,
   allowing gradients to backpropagate through the stochastic collision integral
   for inverse material/TPS optimization.
"""

import torch
import jax
import jax.numpy as jnp
from typing import Tuple

# Assume the Triton kernel is accessible in the newly structured directory
try:
    from custom_kernels.triton.fused_particle_push import dsmc_fused_collision_kernel
except ImportError:
    # Fallback/mock for standalone execution without the full directory context
    dsmc_fused_collision_kernel = None

# ==============================================================================
# 1. HIGH-PERFORMANCE TRITON WRAPPER (Forward-Pass Optimized)
# ==============================================================================

def run_dsmc_collisions(
    velocities: torch.Tensor, 
    cross_sections: torch.Tensor, 
    cell_offsets: torch.Tensor, 
    num_particles_per_cell: torch.Tensor,
    block_size: int = 1024
) -> torch.Tensor:
    """
    Host wrapper to execute the high-performance Triton fused collision kernel.
    This routes the PyTorch tensors to the bare-metal GPU kernel, avoiding 
    HBM roundtrips.
    
    Args:
        velocities: [N, 3] tensor of particle velocities.
        cross_sections: [num_cells] tensor of localized collision cross-sections.
        cell_offsets: [num_cells] tensor of starting memory indices for each cell.
        num_particles_per_cell: [num_cells] tensor of particle counts.
        block_size: Thread block size for SRAM tiling.
        
    Returns:
        output_velocities: [N, 3] tensor of post-collision velocities.
    """
    if dsmc_fused_collision_kernel is None:
        raise ImportError("Triton kernel not found in custom_kernels/triton/")

    num_cells = cell_offsets.numel()
    output_velocities = torch.empty_like(velocities)
    
    # 1D Grid defined by the number of spatial cells
    grid = lambda META: (num_cells, )
    
    # Launch Triton Kernel
    dsmc_fused_collision_kernel[grid](
        velocities, 
        cross_sections, 
        cell_offsets, 
        num_particles_per_cell, 
        output_velocities,
        num_cells, 
        BLOCK_SIZE=block_size
    )
    
    return output_velocities


# ==============================================================================
# 2. DIFFERENTIABLE JAX IMPLEMENTATION (Gradient-Pass Optimized)
# ==============================================================================

def simulate_isotropic_scattering(v1: jnp.ndarray, v2: jnp.ndarray, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulates perfectly elastic, isotropic scattering between two particle velocity arrays.
    """
    # Center of mass velocity
    v_cm = 0.5 * (v1 + v2)
    
    # Relative velocity
    v_rel = v1 - v2
    v_rel_mag = jnp.linalg.norm(v_rel, axis=-1, keepdims=True)
    
    # Generate random scattering angles (isotropic)
    key_theta, key_phi = jax.random.split(key)
    
    # theta in [0, pi] (cosine distribution), phi in [0, 2pi]
    cos_theta = jax.random.uniform(key_theta, shape=v_rel_mag.shape, minval=-1.0, maxval=1.0)
    sin_theta = jnp.sqrt(1.0 - cos_theta**2)
    phi = jax.random.uniform(key_phi, shape=v_rel_mag.shape, minval=0.0, maxval=2.0 * jnp.pi)
    
    # Spherical to Cartesian for the post-collision relative velocity direction
    n_x = sin_theta * jnp.cos(phi)
    n_y = sin_theta * jnp.sin(phi)
    n_z = cos_theta
    
    n_vec = jnp.concatenate([n_x, n_y, n_z], axis=-1)
    v_rel_post = v_rel_mag * n_vec
    
    # Transform back to laboratory frame
    v1_post = v_cm + 0.5 * v_rel_post
    v2_post = v_cm - 0.5 * v_rel_post
    
    return v1_post, v2_post

@jax.jit
def differentiable_collision_step(
    v1: jnp.ndarray, 
    v2: jnp.ndarray, 
    sigma: jnp.ndarray, 
    dt: float, 
    key: jax.Array, 
    tau: float = 0.1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Executes a differentiable DSMC collision step using Gumbel-Softmax relaxation.
    
    Standard DSMC uses discrete accept/reject logic (`if rand() < P: collide()`)
    which creates a zero-gradient step. By relaxing this choice into a continuous 
    probability blend, we allow gradients to propagate from the final macroscopic 
    state all the way back to the initial conditions or cross-section parameters (sigma).
    
    Args:
        v1: [N, 3] velocities of particle group 1.
        v2: [N, 3] velocities of particle group 2.
        sigma: Collision cross-section.
        dt: Time step.
        key: JAX PRNG splittable key.
        tau: Temperature for the Gumbel-Softmax distribution (lower = more discrete).
        
    Returns:
        v1_updated, v2_updated: The updated velocities (differentiable).
    """
    key_gumbel, key_scatter = jax.random.split(key)
    
    # Calculate relative velocity magnitude
    v_rel_mag = jnp.linalg.norm(v1 - v2, axis=-1, keepdims=True)
    
    # Collision probability P = n * sigma * v_rel * dt (simplified macroscopic model)
    # We clip to [0.001, 0.999] to prevent log(0) in the Gumbel-Softmax
    raw_prob = sigma * v_rel_mag * dt
    prob_collide = jnp.clip(raw_prob, 1e-6, 1.0 - 1e-6)
    
    # Logits for the binary decision: [Collide, Do Not Collide]
    logits = jnp.concatenate([jnp.log(prob_collide), jnp.log(1.0 - prob_collide)], axis=-1)
    
    # Add Gumbel noise for stochastic sampling
    gumbel_noise = jax.random.gumbel(key_gumbel, shape=logits.shape)
    
    # Apply Softmax with temperature tau
    # y[..., 0] becomes the continuous 'weight' of the collision happening
    y = jax.nn.softmax((logits + gumbel_noise) / tau, axis=-1)
    collide_weight = y[..., 0:1] # Keep dims for broadcasting
    
    # Calculate the post-collision state (if they theoretically collided)
    v1_post, v2_post = simulate_isotropic_scattering(v1, v2, key_scatter)
    
    # Blend the pre-collision and post-collision states smoothly using the weight.
    # Because 'collide_weight' is a function of 'sigma', gradients will flow back
    # into the cross-section parameters during backpropagation.
    v1_updated = (collide_weight * v1_post) + ((1.0 - collide_weight) * v1)
    v2_updated = (collide_weight * v2_post) + ((1.0 - collide_weight) * v2)
    
    return v1_updated, v2_updated