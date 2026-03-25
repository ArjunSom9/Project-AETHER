"""
Project AETHER: Kernel Fusion & Numerical Parity Test Suite
Validates fused Triton kernels against naive PyTorch references.

This suite ensures that:
1. Fused Particle Push: Matches sequential (Add -> Mul -> Clamp) operations.
2. Fused Softmax: Matches standard torch.softmax for numerical precision.
3. Boundary Conditions: Fused logic correctly reflects/deletes particles at domain edges.
"""

import torch
import unittest
import numpy as np

# Import custom kernels
try:
    from src.custom_kernels.triton.fused_particle_push import fused_particle_push
    from src.custom_kernels.triton.fused_softmax import apply_fused_softmax
except ImportError:
    # Mocks for standalone testing
    fused_particle_push = None
    apply_fused_softmax = None

class TestKernelFusion(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.skipTest("CUDA not available for Triton kernel testing.")

    # -------------------------------------------------------------------------
    # 1. Fused Particle Push Test
    # -------------------------------------------------------------------------
    def test_particle_push_fusion_parity(self):
        """
        Compares the fused Triton particle push against a naive PyTorch implementation.
        """
        if fused_particle_push is None:
            self.skipTest("fused_particle_push module not found.")

        num_particles = 1024 * 16
        dt = 1e-4
        bounds = {
            'min_x': -1.0, 'max_x': 1.0,
            'min_y': -1.0, 'max_y': 1.0,
            'min_z': -1.0, 'max_z': 1.0
        }

        # Initialize identical particle data
        pos_x = torch.randn(num_particles, device=self.device)
        pos_y = torch.randn(num_particles, device=self.device)
        pos_z = torch.randn(num_particles, device=self.device)
        
        vel_u = torch.randn(num_particles, device=self.device)
        vel_v = torch.randn(num_particles, device=self.device)
        vel_w = torch.randn(num_particles, device=self.device)

        # Naive Reference (Sequential Operations)
        ref_x = pos_x + vel_u * dt
        ref_y = pos_y + vel_v * dt
        ref_z = pos_z + vel_w * dt
        
        # Naive Boundary Check (Clamping for parity)
        ref_x = torch.clamp(ref_x, bounds['min_x'], bounds['max_x'])
        ref_y = torch.clamp(ref_y, bounds['min_y'], bounds['max_y'])
        ref_z = torch.clamp(ref_z, bounds['min_z'], bounds['max_z'])

        # Fused Triton Execution
        pos_soa = {'x': pos_x.clone(), 'y': pos_y.clone(), 'z': pos_z.clone()}
        vel_soa = {'u': vel_u, 'v': vel_v, 'w': vel_w}
        
        fused_particle_push(pos_soa, vel_soa, dt, bounds)

        # Assertions
        torch.testing.assert_close(pos_soa['x'], ref_x, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(pos_soa['y'], ref_y, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(pos_soa['z'], ref_z, atol=1e-5, rtol=1e-5)
        
        print("[TEST] Fused Particle Push Parity: PASSED")

    # -------------------------------------------------------------------------
    # 2. Fused Softmax Test
    # -------------------------------------------------------------------------
    def test_softmax_fusion_parity(self):
        """
        Compares the fused Triton softmax against torch.softmax.
        Standard Softmax requires 3 passes (Max, Sum, Div). Our fused kernel 
        does it in 1 pass in SRAM.
        """
        if apply_fused_softmax is None:
            self.skipTest("apply_fused_softmax module not found.")

        rows, cols = 128, 512
        input_data = torch.randn(rows, cols, device=self.device)

        # Naive Reference
        ref_output = torch.softmax(input_data, dim=-1)

        # Fused Triton Execution
        fused_output = apply_fused_softmax(input_data)

        # Assertions
        torch.testing.assert_close(fused_output, ref_output, atol=1e-5, rtol=1e-5)
        
        # Check that rows sum to 1.0
        row_sums = fused_output.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)
        
        print("[TEST] Fused Softmax Parity: PASSED")

    # -------------------------------------------------------------------------
    # 3. Boundary Condition Logic Test
    # -------------------------------------------------------------------------
    def test_boundary_clamping(self):
        """
        Specifically checks if the fused kernel handles particles that 
        SHOULD be clamped at the boundary.
        """
        if fused_particle_push is None:
            self.skipTest("fused_particle_push module not found.")

        num_particles = 1
        dt = 1.0
        bounds = {'min_x': 0.0, 'max_x': 10.0, 'min_y': 0.0, 'max_y': 10.0, 'min_z': 0.0, 'max_z': 10.0}

        # Particle at 9.0 moving with velocity 5.0 (should end at 10.0, not 14.0)
        pos_soa = {
            'x': torch.tensor([9.0], device=self.device),
            'y': torch.tensor([5.0], device=self.device),
            'z': torch.tensor([5.0], device=self.device)
        }
        vel_soa = {
            'u': torch.tensor([5.0], device=self.device),
            'v': torch.tensor([0.0], device=self.device),
            'w': torch.tensor([0.0], device=self.device)
        }

        fused_particle_push(pos_soa, vel_soa, dt, bounds)

        self.assertEqual(pos_soa['x'].item(), 10.0)
        print("[TEST] Boundary Clamping Logic: PASSED")

if __name__ == '__main__':
    unittest.main()