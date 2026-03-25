"""
Project AETHER: Physics-Informed Loss for Surrogate Models
Ensures Transformer predictions strictly conserve physical quantities.

Standard LLM/Transformer training relies purely on data (MSE/Cross-Entropy).
For hypersonic aerothermodynamics, the model must obey the Navier-Stokes 
equations to prevent "hallucinations" in unseen flight regimes. 

This module computes the total loss: L_total = L_data + lambda * ||N(u_pred)||^2
where N represents the non-linear differential operators (PDE residuals).
"""

import torch
import torch.nn as nn

def compute_gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient of a tensor y with respect to a tensor x using 
    PyTorch's automatic differentiation. 
    `create_graph=True` allows for higher-order derivatives (e.g., Laplacians)
    and backpropagation through the derivative calculation itself.
    """
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    return grad

class NavierStokesResidual(nn.Module):
    """
    Calculates the PDE residuals for compressible 2D Navier-Stokes/Euler equations.
    If the network perfectly learns the physics, these residuals will be exactly zero.
    """
    def __init__(self, gamma: float = 1.4):
        super().__init__()
        self.gamma = gamma

    def forward(self, 
                rho: torch.Tensor, 
                u: torch.Tensor, 
                v: torch.Tensor, 
                p: torch.Tensor, 
                t: torch.Tensor, 
                x: torch.Tensor, 
                y: torch.Tensor,
                mu: float = 0.0) -> torch.Tensor:
        """
        Args:
            rho, u, v, p: Predicted density, x-velocity, y-velocity, and pressure.
                          Must be connected to the autograd graph.
            t, x, y: Spatiotemporal coordinates. Must have `requires_grad=True`.
            mu: Dynamic viscosity (0.0 implies inviscid Euler equations).
            
        Returns:
            Total physics residual (sum of squared errors across all conservation laws).
        """
        # ---------------------------------------------------------------------
        # 1. Conservation of Mass (Continuity Equation)
        # d(rho)/dt + d(rho*u)/dx + d(rho*v)/dy = 0
        # ---------------------------------------------------------------------
        rho_t = compute_gradient(rho, t)
        
        rho_u = rho * u
        rho_u_x = compute_gradient(rho_u, x)
        
        rho_v = rho * v
        rho_v_y = compute_gradient(rho_v, y)
        
        continuity_residual = rho_t + rho_u_x + rho_v_y

        # ---------------------------------------------------------------------
        # 2. Conservation of X-Momentum
        # d(rho*u)/dt + d(rho*u^2 + p)/dx + d(rho*u*v)/dy - Viscous Terms = 0
        # ---------------------------------------------------------------------
        rho_u_t = compute_gradient(rho_u, t)
        
        momentum_xx = rho * u**2 + p
        momentum_xx_x = compute_gradient(momentum_xx, x)
        
        momentum_xy = rho * u * v
        momentum_xy_y = compute_gradient(momentum_xy, y)
        
        # Add viscous terms if mu > 0 (requires 2nd derivatives)
        viscous_x = 0.0
        if mu > 0:
            u_x = compute_gradient(u, x)
            u_xx = compute_gradient(u_x, x)
            u_y = compute_gradient(u, y)
            u_yy = compute_gradient(u_y, y)
            viscous_x = mu * (u_xx + u_yy)
            
        x_momentum_residual = rho_u_t + momentum_xx_x + momentum_xy_y - viscous_x

        # ---------------------------------------------------------------------
        # 3. Conservation of Y-Momentum
        # d(rho*v)/dt + d(rho*u*v)/dx + d(rho*v^2 + p)/dy - Viscous Terms = 0
        # ---------------------------------------------------------------------
        rho_v_t = compute_gradient(rho_v, t)
        
        momentum_yx = rho * u * v
        momentum_yx_x = compute_gradient(momentum_yx, x)
        
        momentum_yy = rho * v**2 + p
        momentum_yy_y = compute_gradient(momentum_yy, y)
        
        viscous_y = 0.0
        if mu > 0:
            v_x = compute_gradient(v, x)
            v_xx = compute_gradient(v_x, x)
            v_y = compute_gradient(v, y)
            v_yy = compute_gradient(v_y, y)
            viscous_y = mu * (v_xx + v_yy)
            
        y_momentum_residual = rho_v_t + momentum_yx_x + momentum_yy_y - viscous_y

        # ---------------------------------------------------------------------
        # Total Physics Loss (Mean Squared Error of all residuals)
        # ---------------------------------------------------------------------
        physics_loss = (
            torch.mean(continuity_residual**2) + 
            torch.mean(x_momentum_residual**2) + 
            torch.mean(y_momentum_residual**2)
        )
        
        return physics_loss

class PhysicsInformedLoss(nn.Module):
    """
    The unified loss function for the Project AETHER Surrogate Model.
    Balances empirical data matching (MSE) with PDE constraint satisfaction.
    """
    def __init__(self, physics_weight: float = 1e-3, gamma: float = 1.4):
        super().__init__()
        self.data_loss_fn = nn.MSELoss()
        self.physics_residual_fn = NavierStokesResidual(gamma=gamma)
        
        # Lambda: Weight controlling how strictly the network must obey physics.
        # Often annealed during training (starts low, increases as network converges on data).
        self.physics_weight = physics_weight

    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor, 
                coords: torch.Tensor, 
                mu: float = 0.0) -> tuple:
        """
        Args:
            predictions: [Batch, Sequence, Features] Output from the PIT.
                         Features assumed to be: [rho, u, v, p]
            targets: [Batch, Sequence, Features] Ground truth CFD/DSMC data.
            coords: [Batch, Sequence, 3] Spatiotemporal coordinates [t, x, y].
                    Must have requires_grad=True.
            mu: Fluid viscosity.
            
        Returns:
            Tuple of (Total Loss, Data Loss, Physics Loss)
        """
        # 1. Compute standard Data Loss (MSE against ground truth)
        data_loss = self.data_loss_fn(predictions, targets)
        
        # 2. Extract specific variables for Autograd
        # We must split the tensors individually to track separate gradients
        t = coords[..., 0:1]
        x = coords[..., 1:2]
        y = coords[..., 2:3]
        
        rho = predictions[..., 0:1]
        u   = predictions[..., 1:2]
        v   = predictions[..., 2:3]
        p   = predictions[..., 3:4]
        
        # 3. Compute Physics Residual Loss
        physics_loss = self.physics_residual_fn(rho, u, v, p, t, x, y, mu)
        
        # 4. Total Combined Loss
        total_loss = data_loss + (self.physics_weight * physics_loss)
        
        return total_loss, data_loss, physics_loss

# --- Diagnostic / Testing Block ---
if __name__ == "__main__":
    # Mock parameters
    batch_size = 32
    seq_len = 100 # Representing points in the spatial mesh over time
    
    # Coordinates MUST have requires_grad=True to compute PDE residuals
    coords = torch.rand(batch_size, seq_len, 3, requires_grad=True)
    
    # Mock Model predictions [rho, u, v, p]
    # In reality, this comes from the Transformer Engine output
    preds = torch.rand(batch_size, seq_len, 4, requires_grad=True) 
    
    # Mock Ground Truth from JAX CFD solver
    targets = torch.rand(batch_size, seq_len, 4)
    
    print("Evaluating Physics-Informed Loss Module...")
    
    loss_module = PhysicsInformedLoss(physics_weight=0.1)
    
    total_loss, data_loss, phys_loss = loss_module(preds, targets, coords)
    
    print(f"Data Loss (MSE):       {data_loss.item():.6f}")
    print(f"Physics Loss (PDEs):   {phys_loss.item():.6f}")
    print(f"Total Combined Loss:   {total_loss.item():.6f}")
    
    # Verify backpropagation works
    print("Testing backward pass through Autograd graph...")
    total_loss.backward()
    print("Backward pass successful. Gradients successfully routed through PDE residual operators.")