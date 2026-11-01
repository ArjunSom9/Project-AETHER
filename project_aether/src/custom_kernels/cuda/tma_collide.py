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
"""ctypes binding of the Hopper TMA collision kernel (`dsmc_collide_tma.cu`).

The CUDA sources are built into `libaether_cuda.so` by the CMake project in
`src/custom_kernels` (CUDA 12.2, sm_90a). The library takes raw device
pointers, so it works with PyTorch tensors from any CUDA runtime version that
shares the device's primary context.

Typical usage example:

  library = tma_collide.load_library('build/libaether_cuda.so')
  tma_collide.collide(library, velocity, cell_start, sigma_g_max,
                      collisions, params)
"""

import ctypes
import math
import os
from typing import NamedTuple

import torch

BOLTZMANN = 1.380649e-23
_DEFAULT_LIBRARY = os.path.join(os.path.dirname(__file__), '..', 'build',
                                'libaether_cuda.so')


class CollisionParams(NamedTuple):
    """Gas and timestep parameters of the NTC collision kernel.

    Attributes:
      selection_factor: F_N * dt / V_cell [m^3 s / m^3].
      vhs_coefficient: sigma_T g = coefficient * g^vhs_exponent.
      vhs_exponent: 2 - 2 omega.
      seed: Philox key.
      step: Timestep index (Philox counter).
    """
    selection_factor: float
    vhs_coefficient: float
    vhs_exponent: float
    seed: int
    step: int


def vhs_parameters(mass: float, d_ref: float, omega: float,
                   t_ref: float) -> tuple[float, float]:
    """(coefficient, exponent) of sigma_T g for equal-mass VHS partners."""
    reduced_mass = 0.5 * mass
    coefficient = (math.pi * d_ref**2 *
                   (2.0 * BOLTZMANN * t_ref / reduced_mass)**(omega - 0.5) /
                   math.gamma(2.5 - omega))
    return coefficient, 2.0 - 2.0 * omega


def load_library(path: str | None = None) -> ctypes.CDLL:
    """Loads the CUDA library and declares the C ABI.

    Raises:
      OSError: If the library cannot be loaded.
    """
    library = ctypes.CDLL(path or os.path.abspath(_DEFAULT_LIBRARY))
    fn = library.AetherDsmcCollideTma
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [
        ctypes.c_float] * 3 + [ctypes.c_uint64] * 2 + [ctypes.c_void_p]
    return library


def collide(library: ctypes.CDLL, velocity: torch.Tensor,
            cell_start: torch.Tensor, sigma_g_max: torch.Tensor,
            collisions: torch.Tensor, num_particles: int,
            params: CollisionParams) -> None:
    """Runs NTC collisions in place on cell-sorted velocities.

    Args:
      library: Library from `load_library`.
      velocity: float32 CUDA tensor of shape (3, padded) with padded % 4 == 0;
        columns [0, num_particles) hold vx, vy, vz in cell-sorted order.
      cell_start: int32 CUDA tensor (C + 1,), exclusive prefix of counts.
      sigma_g_max: float32 CUDA tensor (C,), updated in place.
      collisions: int32 CUDA tensor (C,), receives accepted pairs per cell.
      num_particles: Number of real particles.
      params: Collision parameters.

    Raises:
      RuntimeError: If the kernel launch fails.
    """
    stream = torch.cuda.current_stream().cuda_stream
    status = library.AetherDsmcCollideTma(
        velocity.data_ptr(), cell_start.data_ptr(), sigma_g_max.data_ptr(),
        collisions.data_ptr(), num_particles, velocity.shape[1],
        sigma_g_max.numel(), params.selection_factor, params.vhs_coefficient,
        params.vhs_exponent, params.seed, params.step, stream)
    if status != 0:
        raise RuntimeError(
            f'AetherDsmcCollideTma failed with cudaError {status}')
