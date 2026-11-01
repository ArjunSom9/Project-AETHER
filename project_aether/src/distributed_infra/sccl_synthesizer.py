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
"""Topology-aware synthesis of collective algorithms (report Part VI, 6.3).

Implements the synthesis approach of SCCL (Z. Cai et al., "Synthesizing
Optimal Collective Algorithms", PPoPP 2021): a collective is a set of data
chunks with pre- and postconditions (which ranks hold / need each chunk), and
a k-synchronous algorithm moves chunks over the topology's links in S steps.
Step s lasts r_s rounds, and in one round a group of links (e.g. one GPU's
NVSwitch port) carries at most `capacity` chunks. For fixed (S, R = sum r_s)
the existence of an algorithm is an SMT problem, solved here with Z3. Sweeping
(S, R) yields the latency/bandwidth Pareto frontier SCCL reports.

Search is constrained with communication sketches in the spirit of TACCL
(A. Shah et al., NSDI 2023): restricting which links may be used (e.g. only
rail-aligned InfiniBand links between nodes) and requiring the schedule to be
invariant under rank symmetries (e.g. swapping the two nodes).

Synthesized algorithms are executed in JAX with `shard_map` and
`lax.ppermute` (`allgather`, `allreduce`, `alltoall`), so they can replace
the stock collectives inside sharded programs. `allreduce` is SCCL's
composition of a reduce-scatter (the allgather schedule inverted) and an
allgather. `select_algorithm` picks, per message size, the candidate with the
lowest alpha-beta cost, the size-dependent switching SCCL describes.

Typical usage example:

  topo = dgx_cluster(num_nodes=2, gpus_per_node=4)
  algo = synthesize(allgather_collective(8), topo, num_steps=3, num_rounds=3,
                    sketch=rail_sketch(topo, gpus_per_node=4))
  out = jax.jit(shard_map(lambda x: allgather(x, algo, 'r'), ...))(x)
"""

from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import z3

Link = tuple[int, int]


class LinkGroup(NamedTuple):
    """A bandwidth constraint: at most capacity * r_s chunks per step s.

    Attributes:
      links: Directed links sharing the bandwidth.
      capacity: Chunks the group carries per round.
    """
    links: tuple[Link, ...]
    capacity: int


class Topology(NamedTuple):
    """Directed links between ranks plus their bandwidth constraints."""
    name: str
    num_ranks: int
    links: tuple[Link, ...]
    groups: tuple[LinkGroup, ...]


def dgx_cluster(num_nodes: int = 2, gpus_per_node: int = 4,
                nvlink_capacity: int = 2,
                ib_capacity: int = 1) -> Topology:
    """NVSwitch nodes connected by rail-aligned InfiniBand links.

    Inside a node every GPU pair is linked through NVSwitch; each GPU's
    egress and ingress ports are one link group each (SCCL's DGX-2 model).
    Between nodes, GPU g of one node connects to GPU g of every other node
    (one HCA per GPU), each link its own, slower group.

    Args:
      num_nodes: Number of nodes.
      gpus_per_node: GPUs per node.
      nvlink_capacity: Chunks per round through one GPU's NVSwitch port.
      ib_capacity: Chunks per round over one InfiniBand link.

    Returns:
      The topology.
    """
    links, groups = [], []
    for node in range(num_nodes):
        ranks = range(node * gpus_per_node, (node + 1) * gpus_per_node)
        intra = [(a, b) for a in ranks for b in ranks if a != b]
        links.extend(intra)
        for r in ranks:
            groups.append(LinkGroup(tuple(l for l in intra if l[0] == r),
                                    nvlink_capacity))
            groups.append(LinkGroup(tuple(l for l in intra if l[1] == r),
                                    nvlink_capacity))
    for a in range(num_nodes):
        for b in range(num_nodes):
            if a == b:
                continue
            for g in range(gpus_per_node):
                link = (a * gpus_per_node + g, b * gpus_per_node + g)
                links.append(link)
                groups.append(LinkGroup((link,), ib_capacity))
    return Topology(f'dgx{num_nodes}x{gpus_per_node}',
                    num_nodes * gpus_per_node, tuple(links), tuple(groups))


def ring_topology(num_ranks: int) -> Topology:
    """Unidirectional ring, one chunk per link per round."""
    links = tuple((r, (r + 1) % num_ranks) for r in range(num_ranks))
    return Topology(f'ring{num_ranks}', num_ranks, links,
                    tuple(LinkGroup((l,), 1) for l in links))


class Collective(NamedTuple):
    """Chunks with the ranks holding them before and needing them after.

    Attributes:
      name: 'allgather' or 'alltoall'.
      num_ranks: Number of ranks.
      chunks_per_rank: Chunks each rank contributes.
      precondition: For every chunk, the ranks that start with it.
      postcondition: For every chunk, the ranks that must end with it.
    """
    name: str
    num_ranks: int
    chunks_per_rank: int
    precondition: tuple[frozenset[int], ...]
    postcondition: tuple[frozenset[int], ...]

    @property
    def num_chunks(self) -> int:
        return len(self.precondition)

    def permute_chunk(self, chunk: int, sigma: Sequence[int]) -> int:
        """Image of `chunk` under the rank permutation `sigma`."""
        if self.name == 'allgather':
            owner, index = divmod(chunk, self.chunks_per_rank)
            return sigma[owner] * self.chunks_per_rank + index
        src, dst = divmod(chunk, self.num_ranks)
        return sigma[src] * self.num_ranks + sigma[dst]


def allgather_collective(num_ranks: int,
                         chunks_per_rank: int = 1) -> Collective:
    everyone = frozenset(range(num_ranks))
    pre = tuple(frozenset({c // chunks_per_rank})
                for c in range(num_ranks * chunks_per_rank))
    return Collective('allgather', num_ranks, chunks_per_rank, pre,
                      tuple(everyone for _ in pre))


def alltoall_collective(num_ranks: int) -> Collective:
    pre = tuple(frozenset({c // num_ranks}) for c in range(num_ranks**2))
    post = tuple(frozenset({c % num_ranks}) for c in range(num_ranks**2))
    return Collective('alltoall', num_ranks, num_ranks, pre, post)


class Sketch(NamedTuple):
    """TACCL-style communication sketch restricting the search.

    Attributes:
      allowed_links: If set, only these links may carry chunks.
      symmetries: Rank permutations under which the schedule must be
        invariant.
    """
    allowed_links: frozenset[Link] | None = None
    symmetries: tuple[tuple[int, ...], ...] = ()


def rail_sketch(topology: Topology, gpus_per_node: int,
                rails: Sequence[int] | None = None) -> Sketch:
    """Sketch for `dgx_cluster`: chosen inter-node rails, node symmetry.

    Args:
      topology: A `dgx_cluster` topology.
      gpus_per_node: GPUs per node.
      rails: GPU indices whose InfiniBand rails may carry inter-node traffic
        (TACCL's inter-node sketch); all rails if None.

    Returns:
      A sketch allowing every NVSwitch link plus the chosen rails, and
      requiring invariance under rotating whole nodes (a symmetry of the
      topology), which divides the number of free variables by the number of
      nodes.
    """
    num_nodes = topology.num_ranks // gpus_per_node
    rails = set(range(gpus_per_node) if rails is None else rails)

    def allowed(link):
        same_node = link[0] // gpus_per_node == link[1] // gpus_per_node
        return same_node or link[0] % gpus_per_node in rails

    sigma = tuple(((r // gpus_per_node + 1) % num_nodes) * gpus_per_node +
                  r % gpus_per_node for r in range(topology.num_ranks))
    return Sketch(
        allowed_links=frozenset(l for l in topology.links if allowed(l)),
        symmetries=(sigma,) if num_nodes > 1 else ())


class Send(NamedTuple):
    chunk: int
    src: int
    dst: int


class Algorithm(NamedTuple):
    """A synthesized k-synchronous algorithm.

    Attributes:
      collective: The collective it implements.
      topology: Name of the topology it was synthesized for.
      steps: Sends of every step.
      rounds: Rounds of every step.
    """
    collective: Collective
    topology: str
    steps: tuple[tuple[Send, ...], ...]
    rounds: tuple[int, ...]

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def num_rounds(self) -> int:
        return sum(self.rounds)

    def cost(self, buffer_bytes: float, alpha: float, beta: float) -> float:
        """Alpha-beta time: S * alpha + (R / C) * buffer_bytes * beta.

        Args:
          buffer_bytes: Size of the full (gathered) buffer.
          alpha: Per-step latency [s].
          beta: Inverse bandwidth of one link [s/byte].

        Returns:
          Estimated time [s].
        """
        chunk_bytes = buffer_bytes / self.collective.num_chunks
        return self.num_steps * alpha + self.num_rounds * chunk_bytes * beta


def synthesize(collective: Collective, topology: Topology, num_steps: int,
               num_rounds: int, sketch: Sketch = Sketch(),
               timeout_ms: int = 120_000) -> Algorithm | None:
    """Finds an algorithm with exactly S steps and R rounds, if one exists.

    Args:
      collective: Collective to implement.
      topology: Target topology.
      num_steps: Number of synchronous steps S.
      num_rounds: Total rounds R (R >= S).
      sketch: Search restrictions.
      timeout_ms: Z3 time limit.

    Returns:
      The algorithm, or None if the problem is unsatisfiable or timed out.
    """
    if num_rounds < num_steps:
        return None
    num_ranks, num_chunks = topology.num_ranks, collective.num_chunks
    links = [l for l in topology.links
             if sketch.allowed_links is None or l in sketch.allowed_links]
    link_index = {l: i for i, l in enumerate(links)}
    never = num_steps + 1
    solver = z3.Solver()
    solver.set('timeout', timeout_ms)

    time = [[z3.Int(f't_{c}_{r}') for r in range(num_ranks)]
            for c in range(num_chunks)]
    send = [[z3.Bool(f's_{c}_{i}') for i in range(len(links))]
            for c in range(num_chunks)]
    rounds = [z3.Int(f'r_{s}') for s in range(1, num_steps + 1)]

    for c in range(num_chunks):
        pre, post = collective.precondition[c], collective.postcondition[c]
        for r in range(num_ranks):
            if r in pre:
                solver.add(time[c][r] == 0)
                continue
            solver.add(time[c][r] >= 1, time[c][r] <= never)
            if r in post:
                solver.add(time[c][r] <= num_steps)
            incoming = [send[c][i] for i, (_, d) in enumerate(links) if d == r]
            # A chunk present at r arrived over exactly one incoming link.
            solver.add(z3.Implies(time[c][r] <= num_steps, z3.Or(incoming)
                                  if incoming else z3.BoolVal(False)))
            if incoming:
                solver.add(z3.AtMost(*incoming, 1))
        for i, (s, d) in enumerate(links):
            solver.add(z3.Implies(send[c][i],
                                  z3.And(time[c][s] < time[c][d],
                                         time[c][d] <= num_steps)))

    for r in rounds:
        solver.add(r >= 1)
    solver.add(z3.Sum(rounds) == num_rounds)
    for group in topology.groups:
        members = [link_index[l] for l in group.links if l in link_index]
        if not members:
            continue
        for step, r_s in enumerate(rounds, start=1):
            load = [z3.If(z3.And(send[c][i], time[c][links[i][1]] == step),
                          1, 0)
                    for c in range(num_chunks) for i in members]
            solver.add(z3.Sum(load) <= group.capacity * r_s)

    for sigma in sketch.symmetries:
        for c in range(num_chunks):
            c2 = collective.permute_chunk(c, sigma)
            for r in range(num_ranks):
                solver.add(time[c][r] == time[c2][sigma[r]])
            for i, (s, d) in enumerate(links):
                image = (sigma[s], sigma[d])
                if image in link_index:
                    solver.add(send[c][i] == send[c2][link_index[image]])
                else:
                    solver.add(z3.Not(send[c][i]))

    if solver.check() != z3.sat:
        return None
    model = solver.model()
    steps = [[] for _ in range(num_steps)]
    for c in range(num_chunks):
        for i, (s, d) in enumerate(links):
            if z3.is_true(model.eval(send[c][i], model_completion=True)):
                t = model.eval(time[c][d]).as_long()
                steps[t - 1].append(Send(c, s, d))
    return Algorithm(collective, topology.name,
                     tuple(tuple(step) for step in steps),
                     tuple(model.eval(r).as_long() for r in rounds))


def pareto_frontier(collective: Collective, topology: Topology,
                    max_steps: int, max_extra_rounds: int = 4,
                    sketch: Sketch = Sketch()) -> list[Algorithm]:
    """Latency-optimal to bandwidth-optimal algorithms (SCCL's sweep).

    For S = 1..max_steps, finds the fewest rounds R in [S, S +
    max_extra_rounds] and keeps the (S, R) points not dominated by a
    smaller S.
    """
    frontier, best_rounds = [], None
    for steps in range(1, max_steps + 1):
        for rounds in range(steps, steps + max_extra_rounds + 1):
            if best_rounds is not None and rounds >= best_rounds:
                break
            algo = synthesize(collective, topology, steps, rounds, sketch)
            if algo is not None:
                frontier.append(algo)
                best_rounds = rounds
                break
    return frontier


def select_algorithm(candidates: Sequence[Algorithm], buffer_bytes: float,
                     alpha: float, beta: float) -> Algorithm:
    """The candidate with the lowest alpha-beta cost for this message size."""
    return min(candidates, key=lambda a: a.cost(buffer_bytes, alpha, beta))


# ------------------------------------------------------------------------------
# Execution with shard_map + ppermute.
# ------------------------------------------------------------------------------


class _Matching(NamedTuple):
    """Sends with every rank sending and receiving at most once."""
    perm: tuple[Link, ...]
    send_chunk: np.ndarray  # Chunk each rank sends (0 if idle).
    recv_chunk: np.ndarray  # Chunk each rank receives (-1 if idle).


def _matchings(sends: Sequence[Send], num_ranks: int) -> list[_Matching]:
    """Splits one step into rounds of a partial permutation each."""
    remaining = list(sends)
    out = []
    while remaining:
        used_src, used_dst, chosen, rest = set(), set(), [], []
        for s in remaining:
            if s.src in used_src or s.dst in used_dst:
                rest.append(s)
                continue
            used_src.add(s.src)
            used_dst.add(s.dst)
            chosen.append(s)
        send_chunk = np.zeros((num_ranks,), np.int32)
        recv_chunk = np.full((num_ranks,), -1, np.int32)
        for s in chosen:
            send_chunk[s.src] = s.chunk
            recv_chunk[s.dst] = s.chunk
        out.append(_Matching(tuple((s.src, s.dst) for s in chosen),
                             send_chunk, recv_chunk))
        remaining = rest
    return out


def _run_schedule(buffer: jax.Array, steps: Sequence[Sequence[Send]],
                  num_ranks: int, axis_name: str, reduce: bool) -> jax.Array:
    rank = jax.lax.axis_index(axis_name)
    for step in steps:
        for m in _matchings(step, num_ranks):
            payload = buffer[jnp.asarray(m.send_chunk)[rank]]
            received = jax.lax.ppermute(payload, axis_name, m.perm)
            chunk = jnp.asarray(m.recv_chunk)[rank]
            index = jnp.maximum(chunk, 0)
            update = buffer[index] + received if reduce else received
            buffer = jnp.where(chunk >= 0, buffer.at[index].set(update),
                               buffer)
    return buffer


def allgather(local: jax.Array, algo: Algorithm, axis_name: str) -> jax.Array:
    """Synthesized allgather inside `shard_map`.

    Args:
      local: This rank's chunks, (chunks_per_rank, ...).
      algo: An allgather algorithm.
      axis_name: Mesh axis whose ranks take part.

    Returns:
      All chunks in rank order, (num_chunks, ...).
    """
    col = algo.collective
    rank = jax.lax.axis_index(axis_name)
    buffer = jnp.zeros((col.num_chunks,) + local.shape[1:], local.dtype)
    own = rank * col.chunks_per_rank + jnp.arange(col.chunks_per_rank)
    buffer = buffer.at[own].set(local)
    return _run_schedule(buffer, algo.steps, col.num_ranks, axis_name, False)


def allreduce(x: jax.Array, algo: Algorithm, axis_name: str) -> jax.Array:
    """Synthesized sum-allreduce inside `shard_map`.

    The allgather schedule run backwards with accumulation is a
    reduce-scatter (every chunk flows up its broadcast tree to its owner);
    the allgather then redistributes the reduced chunks.

    Args:
      x: This rank's contribution; its leading dimension must be divisible
        by the number of chunks.
      algo: An allgather algorithm (its inverse is the reduce-scatter).
      axis_name: Mesh axis whose ranks take part.

    Returns:
      The sum over ranks, same shape as `x`.
    """
    col = algo.collective
    chunks = x.reshape((col.num_chunks, -1) + x.shape[1:])
    reversed_steps = [[Send(s.chunk, s.dst, s.src) for s in step]
                      for step in reversed(algo.steps)]
    reduced = _run_schedule(chunks, reversed_steps, col.num_ranks, axis_name,
                            True)
    gathered = _run_schedule(reduced, algo.steps, col.num_ranks, axis_name,
                             False)
    return gathered.reshape(x.shape)


def alltoall(x: jax.Array, algo: Algorithm, axis_name: str) -> jax.Array:
    """Synthesized all-to-all inside `shard_map`.

    Args:
      x: Row d of this rank's input is destined for rank d,
        shape (num_ranks, ...).
      algo: An alltoall algorithm.
      axis_name: Mesh axis whose ranks take part.

    Returns:
      Row s holds what rank s sent to this rank, shape (num_ranks, ...).
    """
    n = algo.collective.num_ranks
    rank = jax.lax.axis_index(axis_name)
    buffer = jnp.zeros((n * n,) + x.shape[1:], x.dtype)
    buffer = buffer.at[rank * n + jnp.arange(n)].set(x)
    buffer = _run_schedule(buffer, algo.steps, n, axis_name, False)
    return buffer[jnp.arange(n) * n + rank]


def make_allreduce(algo: Algorithm) -> Callable[[jax.Array, str], jax.Array]:
    """Adapts `algo` to the `fn(x, axis_name)` signature of `lax.psum`."""
    return lambda x, axis_name: allreduce(x, algo, axis_name)
