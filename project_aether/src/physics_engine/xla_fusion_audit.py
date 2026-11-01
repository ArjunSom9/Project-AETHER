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
"""Audits XLA kernel fusion from the compiled (post-optimization) HLO.

Report Part III, 3.2 states that the jitted timestep's HLO dumps were audited
to confirm every fusible stage fused, and that memory traffic per timestep
fell 3-4x relative to an unfused eager kernel sequence. This module provides
the measurements that audit relies on:

  * `analyze`: bytes accessed according to XLA's cost analysis of the
    optimized executable (fusion inputs and outputs only; fused intermediates
    never touch device memory), plus fusion and kernel counts parsed from the
    entry computation of the optimized HLO.
  * `eager_bytes`: the same function executed as an unfused eager kernel
    sequence, i.e. every JAX primitive is its own kernel that reads all of its
    operands from and writes all of its results to device memory.

XLA's fusion decisions are backend specific: the CPU backend, for example,
outlines the Threefry RNG and scatter loops instead of fusing them, so
reductions measured on CPU understate what XLA:GPU achieves on H100.

Typical usage example:

  report = xla_fusion_audit.analyze(my_jittable_fn, *example_args)
  baseline = xla_fusion_audit.eager_bytes(my_jittable_fn, *example_args)
  print(baseline / report.bytes_accessed)
"""

import re
from typing import Any, Callable, NamedTuple

import jax
from jax import core as jax_core

# Matches `%name = type fusion(...)` lines in the entry computation.
_FUSION_RE = re.compile(r'=\s*[^=]*?\bfusion\(')
_INSTRUCTION_RE = re.compile(r'^\s*(ROOT\s+)?%\S+\s*=\s*\S')
# Instructions that do not launch work on the device.
_FREE_OPS = ('parameter(', 'constant(', 'get-tuple-element(', 'tuple(',
             'bitcast(', 'after-all(', 'partition-id(', 'replica-id(')


class CostReport(NamedTuple):
    """Cost of one compiled XLA executable.

    Attributes:
      bytes_accessed: Bytes read plus written according to XLA's cost model.
      flops: Floating-point operations according to XLA's cost model.
      num_fusions: Fusion instructions in the entry computation.
      num_kernels: Entry-computation instructions that launch device work
        (fusions plus unfused ops such as sort, gather, or custom calls).
      hlo_text: Optimized HLO module text, for manual inspection.
    """
    bytes_accessed: float
    flops: float
    num_fusions: int
    num_kernels: int
    hlo_text: str


def _cost_dict(compiled: Any) -> dict[str, float]:
    # jax 0.4.14 returns a list with one dict per HLO module; later releases
    # return the dict directly.
    cost = compiled.cost_analysis()
    if isinstance(cost, (list, tuple)):
        cost = cost[0] if cost else {}
    return dict(cost or {})


def _entry_computation(hlo_text: str) -> list[str]:
    lines = hlo_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('ENTRY'):
            body = []
            for entry_line in lines[i + 1:]:
                if entry_line.startswith('}'):
                    break
                body.append(entry_line)
            return body
    return []


def count_kernels(hlo_text: str) -> tuple[int, int]:
    """Counts (fusions, device-work instructions) in the entry computation."""
    fusions = kernels = 0
    for line in _entry_computation(hlo_text):
        if not _INSTRUCTION_RE.match(line):
            continue
        if any(op in line for op in _FREE_OPS):
            continue
        kernels += 1
        if _FUSION_RE.search(line):
            fusions += 1
    return fusions, kernels


def analyze(fn: Callable[..., Any], *args: Any) -> CostReport:
    """Compiles `fn` for `args` and reports its optimized cost.

    Args:
      fn: A JAX-traceable function. Static configuration should be bound
        with `functools.partial` beforehand.
      *args: Example arguments (arrays or pytrees of arrays).

    Returns:
      The cost report of the optimized executable.
    """
    compiled = jax.jit(fn).lower(*args).compile()
    cost = _cost_dict(compiled)
    hlo_text = compiled.as_text()
    num_fusions, num_kernels = count_kernels(hlo_text)
    return CostReport(
        bytes_accessed=float(cost.get('bytes accessed', 0.0)),
        flops=float(cost.get('flops', 0.0)),
        num_fusions=num_fusions,
        num_kernels=num_kernels,
        hlo_text=hlo_text)


# Call-like primitives whose bodies are inlined, so that each primitive inside
# counts as its own kernel. Control-flow primitives (while, scan, cond) are
# kept whole: eagerly they also run as a single compiled program.
_INLINE_PARAMS = {
    'pjit': 'jaxpr',
    'closed_call': 'call_jaxpr',
    'core_call': 'call_jaxpr',
    'custom_jvp_call': 'call_jaxpr',
    'custom_vjp_call_jaxpr': 'fun_jaxpr',
    'remat': 'jaxpr',
    'checkpoint': 'jaxpr',
}


def _var_bytes(var: Any) -> int:
    if isinstance(var, jax_core.Literal):
        return 0
    aval = var.aval
    if not hasattr(aval, 'dtype'):
        return 0
    return int(aval.size) * aval.dtype.itemsize


def _jaxpr_bytes(jaxpr: Any) -> int:
    total = 0
    for eqn in jaxpr.eqns:
        param = _INLINE_PARAMS.get(eqn.primitive.name)
        if param is not None and param in eqn.params:
            inner = eqn.params[param]
            total += _jaxpr_bytes(getattr(inner, 'jaxpr', inner))
            continue
        total += sum(_var_bytes(v) for v in eqn.invars)
        total += sum(_var_bytes(v) for v in eqn.outvars)
    return total


def eager_bytes(fn: Callable[..., Any], *args: Any) -> float:
    """Bytes moved when `fn` runs as an unfused eager kernel sequence.

    Every primitive in the (inlined) jaxpr of `fn` is treated as one kernel
    that reads its operands from and writes its results to device memory.

    Args:
      fn: A JAX-traceable function with static configuration already bound.
      *args: Example arguments.

    Returns:
      Total bytes read plus written.
    """
    closed = jax.make_jaxpr(fn)(*args)
    return float(_jaxpr_bytes(closed.jaxpr))
