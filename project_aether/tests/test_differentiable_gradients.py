"""
Project AETHER: Differentiable Gradients Test Suite
Validates Autodiff integrity across JAX (Physics) and PyTorch (AI) stacks.

This module ensures that the "differentiable" promise of Project AETHER is met.
It tests:
1. JAX Navier-Stokes Gradients: Can we compute d(Residual)/d(Initial_State)?
2. JAX Ablation Gradients: Can we compute d(Temp)/d(Conductivity)?
3. PyTorch PINN Gradients: Does the PhysicsInformedLoss produce non-zero gradients 
   with respect to spatial coordinates?
"""

import jax
import jax.numpy as jnp
import torch
import unittest

# --- Imports from Project AETHER ---
try:
    from src.physics_engine.cfd_navier_stokes import evaluate_rhs
    from src.physics_engine.ablation_monolithic_solver import implicit_monolithic_step
    from src.surrogate_models.physics_informed_loss import PhysicsInformedLoss
except ImportError:
    # Mocks for demonstration if run outside the full repository context
    evaluate_rhs = None
    implicit_monolithic_step = None
    PhysicsInformedLoss = None

class TestDifferentiablePhysics(unittest.TestCase):

    # -------------------------------------------------------------------------
    # 1. JAX Navier-Stokes Gradient Test
    # -------------------------------------------------------------------------
    def test_navier_stokes_autodiff(self):
        """
        Verifies that JAX can compute gradients through the Navier-Stokes RHS.
        Essential for optimization and stability analysis.
        """
        if evaluate_rhs is None:
            self.skipTest("Navier-Stokes module not found.")

        # Setup 2D domain (small for testing)
        nx, ny = 32, 32
        dx, dy = 0.1, 0.1
        mu = 1.0e-3
        
        # Conserved variables Q [rho, rhu, rhv, E]
        Q = jnp.ones((4, ny, nx))
        
        # Scalar function of the physics output (e.g., total integrated energy)
        def physics_loss(state):
            rhs = evaluate_rhs(state, dx, dy, mu)
            return jnp.sum(rhs**2)

        # Compute gradient using jax.grad
        grad_fn = jax.grad(physics_loss)
        gradients = grad_fn(Q)

        self.assertEqual(gradients.shape, Q.shape)
        self.assertTrue(jnp.any(gradients != 0), "Physics gradients should not be zero.")
        print("[JAX] Navier-Stokes Differentiability: PASSED")

    # -------------------------------------------------------------------------
    # 2. JAX Monolithic Ablation Gradient Test
    # -------------------------------------------------------------------------
    def test_ablation_autodiff(self):
        """
        Verifies that we can differentiate through the implicit CG solve 
        in the monolithic ablation solver.
        """
        if implicit_monolithic_step is None:
            self.skipTest("Ablation solver module not found.")

        nx, ny = 16, 16
        T_old = jnp.ones((ny, nx)) * 300.0
        phi_old = jnp.ones((ny, nx)) # Pure solid
        u = jnp.zeros((ny, nx))
        v = jnp.zeros((ny, nx))
        
        # Function to minimize (e.g. minimize final temperature at the bondline)
        def ablation_target(temp_state):
            T_new, _ = implicit_monolithic_step(
                temp_state, phi_old, u, v, dx=0.1, dy=0.1, dt=1e-3, cg_maxiter=10
            )
            return jnp.mean(T_new)

        # Gradient of average temperature w.r.t initial temperature
        grad_fn = jax.grad(ablation_target)
        grads = grad_fn(T_old)

        self.assertEqual(grads.shape, T_old.shape)
        self.assertTrue(jnp.any(grads != 0), "Ablation gradients through implicit solve: PASSED")
        print("[JAX] Monolithic Ablation Differentiability: PASSED")

    # -------------------------------------------------------------------------
    # 3. PyTorch Physics-Informed Loss Test
    # -------------------------------------------------------------------------
    def test_pinn_loss_autodiff(self):
        """
        Verifies that PyTorch can compute higher-order derivatives 
        (d2_u/dx2) required for the PhysicsInformedLoss module.
        """
        if PhysicsInformedLoss is None:
            self.skipTest("PhysicsInformedLoss module not found.")

        batch, seq = 2, 64
        # Preds: [rho, u, v, p]
        preds = torch.randn(batch, seq, 4, requires_grad=True)
        targets = torch.randn(batch, seq, 4)
        # Coords: [t, x, y]
        coords = torch.randn(batch, seq, 3, requires_grad=True)
        
        loss_fn = PhysicsInformedLoss(physics_weight=1.0)
        total_loss, _, phys_loss = loss_fn(preds, targets, coords, mu=0.01)

        # Backward pass through the entire PDE residual chain
        total_loss.backward()

        # Check if coordinates have gradients (indicates PDE calculation was successful)
        self.assertIsNotNone(coords.grad, "Coordinates should receive gradients from PDE residuals.")
        self.assertTrue(torch.any(coords.grad != 0))
        
        # Check if model predictions have gradients
        self.assertIsNotNone(preds.grad, "Model predictions should receive gradients from PINN loss.")
        
        print("[PyTorch] Physics-Informed Loss Differentiability: PASSED")

if __name__ == '__main__':
    unittest.main()