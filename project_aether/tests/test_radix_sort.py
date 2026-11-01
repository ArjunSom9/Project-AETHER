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
"""The fused Onesweep radix sort: NumPy replay (CPU) and Triton (CUDA)."""

import numpy as np
import pytest

from src.custom_kernels.triton import radix_sort_reference as ref


@pytest.mark.parametrize('n,cells', [(5000, 10_000), (12_345, 300), (1, 7),
                                     (2048, 1), (4096, 16_384)])
def test_replay_is_a_stable_sort(n, cells):
    rng = np.random.default_rng(n)
    keys = ((rng.integers(0, cells, n) << ref.SHUFFLE_BITS) |
            rng.integers(0, 1 << ref.SHUFFLE_BITS, n)).astype(np.int32)
    values = np.arange(n, dtype=np.int32)
    sorted_keys, sorted_values = ref.onesweep_reference(
        keys, values, ref.num_passes(cells), rng, tile=256)
    order = np.argsort(keys, kind='stable')
    np.testing.assert_array_equal(sorted_keys, keys[order])
    np.testing.assert_array_equal(sorted_values, order)


def test_lookback_walks_aggregates_to_an_inclusive_prefix():
    status = np.zeros((4, ref.RADIX), np.int64)
    hist = np.arange(4 * ref.RADIX).reshape(4, ref.RADIX) % 5
    status[0] = hist[0] | ref.FLAG_INCLUSIVE
    status[1] = (hist[0] + hist[1]) | ref.FLAG_INCLUSIVE
    status[2] = hist[2] | ref.FLAG_AGGREGATE
    exclusive = ref._lookback(status, 3)  # pylint: disable=protected-access
    np.testing.assert_array_equal(exclusive, hist[:3].sum(axis=0))


def test_pass_count_for_report_benchmark():
    # 10k cells need 14 bits, plus 7 shuffle bits: three 7-bit passes.
    assert ref.num_passes(10_000) == 3
    with pytest.raises(ValueError):
        ref.num_passes(1 << 25)


def test_status_word_fields_do_not_overlap():
    assert ref.FLAG_AGGREGATE & ref.FLAG_INCLUSIVE == 0
    assert ref.VALUE_MASK & ref.FLAG_MASK == 0
    assert ref.FLAG_MASK < 2**31  # Fits a signed int32.
    assert ref.TILE <= ref.VALUE_MASK


def test_triton_sort_matches_numpy_on_gpu():
    torch = pytest.importorskip('torch')
    pytest.importorskip('triton')
    if not torch.cuda.is_available():
        pytest.skip('needs a CUDA GPU')
    from src.custom_kernels.triton import fused_radix_sort  # pylint: disable=import-outside-toplevel
    n = 1_000_003
    grid = ref.CellGrid(lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0),
                        cells=(25, 25, 16))
    gen = torch.Generator(device='cuda').manual_seed(0)
    x, y, z = (torch.rand(n, device='cuda', generator=gen) for _ in range(3))
    out = fused_radix_sort.sort_particles_by_cell(x, y, z, grid, seed=1234)
    keys = out.keys.cpu().numpy()
    assert np.all(np.diff(keys) >= 0)
    cell = keys >> ref.SHUFFLE_BITS
    index = out.particle_index.cpu().numpy().astype(np.int64)
    assert np.array_equal(np.sort(index), np.arange(n))
    expected_cell = (
        (np.clip((z.cpu().numpy() * 16).astype(int), 0, 15) * 25 +
         np.clip((y.cpu().numpy() * 25).astype(int), 0, 24)) * 25 +
        np.clip((x.cpu().numpy() * 25).astype(int), 0, 24))
    np.testing.assert_array_equal(cell, expected_cell[index])
    np.testing.assert_array_equal(out.cell_counts.cpu().numpy(),
                                  np.bincount(cell, minlength=grid.num_cells))
