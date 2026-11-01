# Project AETHER

**GPU systems engineering for thermochemical hypersonic environment reconstruction**
Execution window: June 5 – August 25, 2023. Companion code to
*Project AETHER – GPU Systems Engineering Report (Jun–Aug 2023)*.

AETHER re-engineers a hypersonic reconstruction stack (DSMC particle
transport, finite-volume aerothermodynamics, and learned surrogate inference)
around Hopper-class GPUs. Every layer is treated as a data-movement problem:
fused Triton kernels for particle sorting, JAX/OpenXLA fusion and hand-written
TMA pipelines for timestep traffic, CUTLASS block-sparse GEMMs and FP8
quantization for arithmetic density, and disaggregated serving with a paged
state cache and synthesized collectives for the system layer.

## Report → code

| Report | What | Code |
|---|---|---|
| Part II | Fused Onesweep-style radix sort: cell ids in registers, Philox in-cell shuffle, per-tile histograms, decoupled look-back, scatter, per-cell counts (Triton 2.0) | `src/custom_kernels/triton/fused_radix_sort.py`, NumPy replay `radix_sort_reference.py` |
| Part II | 32-core CPU baseline vs. fused sort, collision-statistics check | `scripts/phase2_triton_sort_benchmark.py` |
| Part III 3.1–3.2 | Differentiable DSMC timestep as one jitted function; eager vs. fused memory-traffic audit from the compiled HLO | `src/physics_engine/dsmc_boltzmann.py`, `xla_fusion_audit.py`, `scripts/phase3_fusion_audit.py` |
| Part III 3.1 | Navier–Stokes finite volume (MUSCL + Rusanov + viscous, SSP-RK3), ablation (GMRES), 5-species Park chemistry | `src/physics_engine/cfd_navier_stokes.py`, `ablation_monolithic_solver.py`, `chemistry_kinetics.py` |
| Part III 3.3 | Hopper TMA bulk-tensor copies + mbarrier producer/consumer pipeline in CUDA C++ (NTC collisions on sorted particles) | `src/custom_kernels/cuda/` (`sm90_pipeline.h`, `dsmc_collide_tma.cu`) |
| Part IV 4.1 | Physics-informed Galerkin/Fourier Transformer with FNO spectral convolution; data + PDE-residual loss | `src/surrogate_models/galerkin_transformer.py`, `physics_informed_loss.py`, `scripts/train_surrogate.py` |
| Part IV 4.2 | 75%-sparse window + global block attention, FlashAttention-style tiling | `src/surrogate_models/block_sparse_attention.py` |
| Part IV 4.3 | MOTR/TrackFormer track queries; debris detection on the impact clip | `src/surrogate_models/mot_trackformer.py`, `data/sample_clip/` |
| Part V 5.1 | CUTLASS 3.x (CuTe collective builder, TMA + WGMMA) block-sparse GEMM vs. cuBLAS (Table 4) | `src/custom_kernels/cutlass/` |
| Part V 5.2 | FP8 E4M3/E5M2 post-training quantization with calibration | `src/surrogate_models/quantization_fp8.py`, `scripts/phase4_numerics.py` |
| Part VI 6.1–6.2 | Prefill/decode pools, iteration-level scheduling, chunked prefill, PagedAttention (refcounts, copy-on-write branches) vs. monolithic baseline | `src/distributed_infra/` (`disaggregated_*.py`, `kv_paged_attention.py`, `serving_scheduler.py`), `scripts/phase5_digital_twin.py` |
| Part VI 6.3 | SCCL-style SMT synthesis of allgather/allreduce/alltoall with TACCL sketches, executed via `shard_map`; Megatron TP + GSPMD | `src/distributed_infra/sccl_synthesizer.py`, `mesh_sharding_4d.py` |

## Software stack (all released on or before 2023-08-25)

| Component | Version |
|---|---|
| Python | 3.11 (3.11.4, 2023-06-06) |
| JAX / jaxlib | 0.4.14 (2023-07-27); jaxlib is served from the JAX release index (removed from PyPI) |
| NumPy / SciPy / ml_dtypes | 1.25.2 / 1.11.2 / 0.2.0 |
| Optax / chex | 0.1.7 / 0.1.82 |
| PyTorch / Triton (GPU) | 2.0.1 / 2.0.0 (Triton 2.1 shipped 2023-09-01) |
| CUDA / cuDNN | 12.2 Update 1 (12.2.128) / 8.9.4 |
| CUTLASS | commit `4575443d` "CUTLASS 3.2 (#1024)", 2023-08-08 (the v3.2.0 tag was published 2023-08-28, after the window) |
| z3-solver | 4.12.2.0 |

Every direct and transitive Python dependency is pinned in
`project_aether/requirements*.txt`.

## Quick start

```bash
cd project_aether
pip install -r requirements-cpu.txt          # Linux GPU nodes: requirements-gpu.txt
python -m pytest                              # 8 virtual XLA devices on CPU
python -m pylint --rcfile=pyproject.toml src scripts tests

python -m scripts.phase3_fusion_audit
python -m scripts.train_surrogate --out outputs/surrogate.npz
python -m scripts.phase4_numerics --checkpoint outputs/surrogate.npz
XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m scripts.phase5_digital_twin

# H100 only:
python -m scripts.phase2_triton_sort_benchmark --cpu-threads 32
cmake -S src/custom_kernels -B src/custom_kernels/build -DCMAKE_BUILD_TYPE=Release
cmake --build src/custom_kernels/build -j
./src/custom_kernels/build/dsmc_collide_tma_test
./src/custom_kernels/build/block_sparse_gemm_benchmark
```

## Verification status

* **JAX code** (physics engine, surrogates, quantization, serving, sharding,
  collective synthesis) is covered by the pytest suite, which passes on CPU
  with the pinned stack. Gradients are checked against central differences,
  the radix-sort algorithm against `np.argsort`, synthesized collectives
  against `psum` / `all_gather` / `all_to_all`, and disaggregated serving
  against the monolithic baseline.
* **Triton kernels** use only Triton 2.0.0 APIs; their GPU parity tests are
  skipped without CUDA.
* **CUDA C++ / CUTLASS** sources pass clang's sm_90a semantic analysis
  against the CUDA 12.2.1 headers and the pinned CUTLASS commit, and their
  CPU-replay / cuBLAS correctness checks run on an sm_90 GPU.
* The headline performance figures of the report (50x, 3–4x HBM bytes,
  2–3x over cuBLAS, 9x latency) are H100 measurements; the scripts measure
  them on the hardware they run on and never print fixed numbers.

## Layout

```
project_aether/
  src/custom_kernels/   triton/ (Triton 2.0), cuda/ (TMA, sm_90a), cutlass/, CMakeLists.txt
  src/physics_engine/   DSMC, Navier-Stokes, ablation, chemistry, PIC, fusion audit, GPU pipeline
  src/surrogate_models/ Galerkin transformer, PINN loss, block-sparse attention, FP8, MOTR, trajectory model
  src/distributed_infra/ paged cache, prefill/decode pools, scheduler, SCCL synthesis, mesh sharding
  scripts/              phase2..phase5 benchmarks and surrogate training (python -m scripts.<name>)
  tests/                pytest suite (GPU tests skip without CUDA)
  data/sample_clip/     hypervelocity-impact shadowgraph frames
```
