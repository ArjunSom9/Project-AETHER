"""
Project AETHER: Monolithic Differentiable Ablation Solver
Optimized for JAX XLA Compilation and Automatic Differentiation.

Legacy codes use "Schwarz splitting" (loose coupling) to exchange boundary 
conditions between solid conduction and fluid dynamics solvers, causing latency 
and stability issues. 

This module implements a monolithic approach where the solid Thermal Protection 
System (TPS) and the hypersonic fluid domain are treated as a single unified 
domain with spatially varying properties. This allows for an implicit time 
integration that is unconditionally stable. By writing this purely functionally 
in JAX, gradients can backpropagate through the fluid-solid interface for 
inverse material property optimization (e.g., optimizing TPS porosity).
"""

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsp_linalg
from functools import partial

# --- Physical Material Properties (Mock) ---
# In a full simulation, these are temperature-dependent and learned via the surrogate.
K_SOLID = 0.5       # Thermal conductivity of TPS (W/mK)
K_FLUID = 0.05      # Thermal conductivity of plasma/gas (W/mK)
RHO_CP_SOLID = 2e6  # Volumetric heat capacity of solid (J/m^3K)
RHO_CP_FLUID = 1e3  # Volumetric heat capacity of gas (J/m^3K)
LATENT_HEAT = 1e7   # Latent heat of ablation/pyrolysis (J/kg)
ABLATION_TEMP = 3000.0 # Temperature at which rapid ablation begins (K)

def get_monolithic_properties(phi: jnp.ndarray):
    """
    Interpolates thermal properties across the unified domain based on the solid fraction.
    phi = 1.0 (Pure Solid TPS)
    phi = 0.0 (Pure Fluid/Plasma)
    0 < phi < 1 (Knudsen/Transpiration Ablation Layer)
    """
    # Smooth interpolation for differentiability
    k_eff = phi * K_SOLID + (1.0 - phi) * K_FLUID
    rho_cp_eff = phi * RHO_CP_SOLID + (1.0 - phi) * RHO_CP_FLUID
    return k_eff, rho_cp_eff

def compute_ablation_source(T: jnp.ndarray, phi: jnp.ndarray, dt: float):
    """
    Calculates the energy sink due to pyrolysis/ablation and updates the solid fraction.
    Uses a differentiable Arrhenius-like continuous relaxation instead of a hard step function.
    """
    # Differentiable activation: rapidly approaches 1 as T exceeds ABLATION_TEMP
    activation = jax.nn.sigmoid(0.01 * (T - ABLATION_TEMP))
    
    # Ablation rate is proportional to available solid and activation temperature
    ablation_rate = phi * activation * 10.0 # Base rate constant
    
    # Energy sink (endothermic phase change)
    Q_ablation = -ablation_rate * LATENT_HEAT
    
    # Update solid fraction (mass loss)
    phi_new = phi - ablation_rate * dt
    phi_new = jnp.clip(phi_new, 0.0, 1.0) # Ensure physical bounds
    
    return Q_ablation, phi_new

def apply_laplacian(T: jnp.ndarray, k_eff: jnp.ndarray, dx: float, dy: float):
    """
    Computes divergence of the heat flux: \nabla \cdot (k \nabla T)
    Using central finite differences with spatially varying conductivity.
    """
    # Interpolate conductivities to cell faces
    k_x_face = 0.5 * (k_eff + jnp.roll(k_eff, shift=-1, axis=1))
    k_y_face = 0.5 * (k_eff + jnp.roll(k_eff, shift=-1, axis=0))
    
    # Fluxes across faces
    dT_dx = (jnp.roll(T, shift=-1, axis=1) - T) / dx
    dT_dy = (jnp.roll(T, shift=-1, axis=0) - T) / dy
    
    flux_x = k_x_face * dT_dx
    flux_y = k_y_face * dT_dy
    
    # Divergence of fluxes
    div_flux_x = (flux_x - jnp.roll(flux_x, shift=1, axis=1)) / dx
    div_flux_y = (flux_y - jnp.roll(flux_y, shift=1, axis=0)) / dy
    
    return div_flux_x + div_flux_y

def apply_advection(T: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float):
    """
    Computes advective heat transfer: \nabla \cdot (\vec{u} T)
    Uses a simple upwind or central scheme (central used here for simplicity).
    Note: u and v are zero inside the solid (phi=1).
    """
    dT_dx = (jnp.roll(T, shift=-1, axis=1) - jnp.roll(T, shift=1, axis=1)) / (2.0 * dx)
    dT_dy = (jnp.roll(T, shift=-1, axis=0) - jnp.roll(T, shift=1, axis=0)) / (2.0 * dy)
    
    return -(u * dT_dx + v * dT_dy)

@partial(jax.jit, static_argnames=['dx', 'dy', 'dt', 'cg_maxiter'])
def implicit_monolithic_step(
    T_old: jnp.ndarray, 
    phi_old: jnp.ndarray, 
    u: jnp.ndarray, 
    v: jnp.ndarray, 
    dx: float, 
    dy: float, 
    dt: float,
    cg_maxiter: int = 50
):
    """
    Executes a single unconditionally stable implicit time step across the unified domain.
    Solves: [RhoCp - dt * (Laplacian + Advection)] T_new = RhoCp * T_old + dt * Q_ablation
    
    Using JAX's Conjugate Gradient solver avoids matrix instantiation, saving massive amounts
    of memory (crucial for HBM limits on GPUs).
    """
    # 1. Get continuous properties for the current state
    k_eff, rho_cp_eff = get_monolithic_properties(phi_old)
    
    # 2. Compute explicit ablation source terms
    Q_abl, phi_new = compute_ablation_source(T_old, phi_old, dt)
    
    # 3. Define the implicit Linear Operator A(x) for the CG solver
    def linear_operator(T_guess):
        # Diffusion term
        diffusion = apply_laplacian(T_guess, k_eff, dx, dy)
        # Advection term (Blowing effect from transpiration pushes hot gas away)
        advection = apply_advection(T_guess, u, v, dx, dy)
        
        # LHS: RhoCp * T - dt * (\nabla (k \nabla T) + Advection)
        return rho_cp_eff * T_guess - dt * (diffusion + rho_cp_eff * advection)

    # 4. Construct the Right-Hand Side (b)
    rhs = rho_cp_eff * T_old + dt * Q_abl
    
    # 5. Solve the linear system A * T_new = rhs using Conjugate Gradient
    # jsp_linalg.cg supports automatic differentiation through the iterative solve!
    T_new, info = jsp_linalg.cg(
        A=linear_operator, 
        b=rhs, 
        x0=T_old, # Warm start with previous temperature
        maxiter=cg_maxiter,
        tol=1e-5
    )
    
    return T_new, phi_new

def compute_blowing_velocity(phi_old: jnp.ndarray, phi_new: jnp.ndarray, dt: float, dx: float):
    """
    Couples the solid ablation back to the fluid solver by generating an injection velocity.
    Mass conservation: \rho_{solid} * \Delta \phi = \rho_{gas} * V_{blowing}
    """
    mass_loss_rate = (phi_old - phi_new) / dt * (RHO_CP_SOLID / 1000.0) # Approx mass loss
    
    # Generating a normal velocity (v_blowing) at the interface pushing into the gas
    # This thickens the boundary layer and blocks incoming convective heat flux.
    v_blowing = mass_loss_rate / (RHO_CP_FLUID / 1000.0) 
    
    return v_blowing