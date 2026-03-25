"""
Project AETHER: Fused Particle Push and Boundary Check Kernel
Optimized for NVIDIA H100 (Hopper) architectures.

This module implements a custom OpenAI Triton kernel for the PIC/DSMC "Move" phase.
It utilizes a Structure-of-Arrays (SoA) layout to maximize cache line utilization
and fuses the kinematic update with spatial boundary checks to bypass HBM roundtrips.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_push_boundary_kernel(
    # Pointers to Structure-of-Arrays (SoA) layout for Positions
    pos_x_ptr, pos_y_ptr, pos_z_ptr,
    # Pointers to Structure-of-Arrays (SoA) layout for Velocities
    vel_x_ptr, vel_y_ptr, vel_z_ptr,
    # Pointers to output SoA (can be same as input for in-place updates)
    out_pos_x_ptr, out_pos_y_ptr, out_pos_z_ptr,
    out_vel_x_ptr, out_vel_y_ptr, out_vel_z_ptr,
    # Simulation Parameters
    dt: tl.float32,
    domain_max_x: tl.float32, domain_max_y: tl.float32, domain_max_z: tl.float32,
    domain_min_x: tl.float32, domain_min_y: tl.float32, domain_min_z: tl.float32,
    num_particles: tl.int32,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel that loads SoA particle data, advances positions by v*dt,
    evaluates domain wall collisions (specular reflection), and stores the result.
    Fusing these steps keeps the intermediate out-of-bounds state entirely in SRAM.
    """
    # 1. Map thread block to particle indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to prevent out-of-bounds memory access for the last block
    mask = offsets < num_particles

    # 2. Coalesced Load from Global Memory (HBM) -> SRAM
    # Using SoA guarantees contiguous memory access across the warp
    x = tl.load(pos_x_ptr + offsets, mask=mask)
    y = tl.load(pos_y_ptr + offsets, mask=mask)
    z = tl.load(pos_z_ptr + offsets, mask=mask)
    
    vx = tl.load(vel_x_ptr + offsets, mask=mask)
    vy = tl.load(vel_y_ptr + offsets, mask=mask)
    vz = tl.load(vel_z_ptr + offsets, mask=mask)

    # 3. Kinematic Push (Compute in SRAM)
    next_x = x + vx * dt
    next_y = y + vy * dt
    next_z = z + vz * dt

    # 4. Fused Boundary Checks (Specular Reflection)
    # X-Axis boundaries
    hit_x_min = next_x < domain_min_x
    hit_x_max = next_x > domain_max_x
    next_vx = tl.where(hit_x_min | hit_x_max, -vx, vx) # Reverse velocity
    next_x = tl.where(hit_x_min, domain_min_x + (domain_min_x - next_x), next_x) # Bounce logic
    next_x = tl.where(hit_x_max, domain_max_x - (next_x - domain_max_x), next_x)

    # Y-Axis boundaries
    hit_y_min = next_y < domain_min_y
    hit_y_max = next_y > domain_max_y
    next_vy = tl.where(hit_y_min | hit_y_max, -vy, vy)
    next_y = tl.where(hit_y_min, domain_min_y + (domain_min_y - next_y), next_y)
    next_y = tl.where(hit_y_max, domain_max_y - (next_y - domain_max_y), next_y)

    # Z-Axis boundaries
    hit_z_min = next_z < domain_min_z
    hit_z_max = next_z > domain_max_z
    next_vz = tl.where(hit_z_min | hit_z_max, -vz, vz)
    next_z = tl.where(hit_z_min, domain_min_z + (domain_min_z - next_z), next_z)
    next_z = tl.where(hit_z_max, domain_max_z - (next_z - domain_max_z), next_z)

    # 5. Coalesced Store from SRAM -> Global Memory (HBM)
    tl.store(out_pos_x_ptr + offsets, next_x, mask=mask)
    tl.store(out_pos_y_ptr + offsets, next_y, mask=mask)
    tl.store(out_pos_z_ptr + offsets, next_z, mask=mask)
    
    tl.store(out_vel_x_ptr + offsets, next_vx, mask=mask)
    tl.store(out_vel_y_ptr + offsets, next_vy, mask=mask)
    tl.store(out_vel_z_ptr + offsets, next_vz, mask=mask)


def fused_particle_push(
    positions_soa: dict, 
    velocities_soa: dict, 
    dt: float, 
    domain_bounds: dict,
    in_place: bool = True
) -> tuple:
    """
    Host wrapper for the Triton fused push and boundary check kernel.
    
    Args:
        positions_soa: dict containing 'x', 'y', 'z' 1D torch.Tensors.
        velocities_soa: dict containing 'u', 'v', 'w' 1D torch.Tensors.
        dt: Time step size.
        domain_bounds: dict containing 'min_x', 'max_x', etc.
        in_place: If True, overwrites input tensors. If False, allocates new tensors.
        
    Returns:
        tuple containing updated (positions_soa, velocities_soa).
    """
    num_particles = positions_soa['x'].shape[0]
    
    # Configure output dictionaries based on in-place flag
    out_pos = positions_soa if in_place else {k: torch.empty_like(v) for k, v in positions_soa.items()}
    out_vel = velocities_soa if in_place else {k: torch.empty_like(v) for k, v in velocities_soa.items()}

    # Determine grid size. Block size 1024 is highly optimized for A100/H100
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(num_particles, meta['BLOCK_SIZE']), )

    # Launch Triton Kernel
    _fused_push_boundary_kernel[grid](
        positions_soa['x'], positions_soa['y'], positions_soa['z'],
        velocities_soa['u'], velocities_soa['v'], velocities_soa['w'],
        out_pos['x'], out_pos['y'], out_pos['z'],
        out_vel['u'], out_vel['v'], out_vel['w'],
        dt=dt,
        domain_max_x=domain_bounds['max_x'], domain_max_y=domain_bounds['max_y'], domain_max_z=domain_bounds['max_z'],
        domain_min_x=domain_bounds['min_x'], domain_min_y=domain_bounds['min_y'], domain_min_z=domain_bounds['min_z'],
        num_particles=num_particles,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_pos, out_vel