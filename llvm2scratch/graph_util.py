from dataclasses import dataclass, field

from scipy.sparse import csc_array
import igraph as ig
import numpy as np
import itertools
import math

@dataclass
class NodeInfo:
  depends: set[str]
  modifies: set[str]
  calls: set[str]
  direct_modifies: set[str]
  direct_calls: set[str]

@dataclass
class CallGraphAnalysis:
  entrypoint: str
  info: dict[str, NodeInfo]
  analyzed: set[str] = field(default_factory=set)

  def analyzeNode(self, name: str) -> bool:
    changed = False
    info = self.info[name]
    for callee in info.direct_calls:
      callee_info = self.info[callee]

      if callee not in self.analyzed:
        self.analyzed.add(callee)
        changed = self.analyzeNode(callee) or changed
        callee_info = self.info[callee]

      new_modifies = callee_info.modifies - info.modifies
      new_depends = (callee_info.depends - info.direct_modifies) - \
        info.depends
      new_calls = callee_info.calls - info.calls

      if len(new_modifies | new_depends | new_calls) > 0:
        changed = True

        info.modifies |= new_modifies
        info.depends |= new_depends
        info.calls |= new_calls

    self.info[name] = info

    return changed

  def analyze(self):
    while self.analyzeNode(self.entrypoint):
      self.analyzed = set()

def findNodesWithCycle(graph: dict[str, list[str]]) -> set[str]:
  nodes = list(graph.keys())
  node_idx = {n: i for i, n in enumerate(nodes)}
  edges = [(node_idx[u], node_idx[v]) for u in graph for v in graph[u]]
  g = ig.Graph(directed=True)
  g.add_vertices(len(nodes))
  g.add_edges(edges)

  sccs = g.components(mode="STRONG")
  result = {
    nodes[v]
    for comp in sccs
    if len(comp) > 1
    for v in comp
  }

  self_loops = {nodes[e.tuple[0]] for e in g.es if e.tuple[0] == e.tuple[1]}
  return result | self_loops

def selectCycleChecks(graph: dict[str, list[str]]) -> list[str]:
  # TODO: if graph has a very large amount of cycles, then use findNodesWithCycle instead
  nodes = sorted(graph.keys())
  node_idx = {n: i for i, n in enumerate(nodes)}
  edges = [(node_idx[u], node_idx[v]) for u in graph for v in graph[u]]

  g = ig.Graph(directed=True)
  g.add_vertices(len(nodes))
  g.add_edges(edges)

  cycles = g.simple_cycles()

  if not cycles: return []
  # Any node that can call itself must have a stack check; therefore we can ignore cycles
  # containing that node
  self_loop_nodes = {c[0] for c in cycles if len(c) == 1}
  remaining = [c for c in cycles if not self_loop_nodes.intersection(c)]

  if not remaining:
    return [nodes[i] for i in self_loop_nodes]

  # Avoid counting number of nodes if it is expensive to do so
  if len(remaining) < 1000:
    calc_exact = len(set().union(*[set(c) for c in remaining])) <= 15
  else:
    calc_exact = False

  if calc_exact:
    greedy_nodes = set(minHittingSetExact(remaining))
  else:
    greedy_nodes = set(greedyHittingSet(remaining, nodes))

  return [nodes[i] for i in self_loop_nodes | greedy_nodes]

def minHittingSetExact(cycles: list[list[int]]) -> list[int]:
  if not cycles: return []
  cycle_sets = [set(c) for c in cycles]
  all_nodes = sorted(set().union(*cycle_sets))
  for r in range(1, len(all_nodes) + 1):
    for comb in itertools.combinations(all_nodes, r):
      s = set(comb)
      if all(s & c for c in cycle_sets):
        return list(comb)
  return all_nodes

def greedyHittingSet(cycles: list[list[int]], nodes: list[str]) -> list[int]:
  if not cycles: return []
  n_nodes = len(nodes)
  lengths = np.array([len(c) for c in cycles], dtype=np.int32)
  rows = np.repeat(np.arange(len(cycles), dtype=np.int32), lengths)
  flat = list(itertools.chain.from_iterable(cycles))
  cols = np.array(flat, dtype=np.int32)
  subsets = csc_array(
    (np.ones(len(rows), dtype=np.float32), (rows, cols)),
    shape=(len(cycles), n_nodes)
  )

  slice_col = lambda j: subsets.indices[slice(*subsets.indptr[[j, j+1]])]
  n, j = len(cycles), n_nodes
  set_counts = np.asarray(subsets.sum(axis=0)).ravel()
  point_cover, soln = np.zeros(n, dtype=bool), np.zeros(j, dtype=bool)

  while not np.all(point_cover):
    opt_s = np.argmax(set_counts)
    point_cover[slice_col(opt_s)] = True
    set_counts = np.maximum((~point_cover).view(np.uint8) @ subsets, 0)
    soln[opt_s] = True

  return np.flatnonzero(soln).tolist()

def unavoidableNodes(graph: dict[str, list[str]], source: str, target: str) -> set[str]:
  nodes = list(graph.keys())
  node_idx = {n: i for i, n in enumerate(nodes)}
  edges = [(node_idx[u], node_idx[v]) for u in graph for v in graph[u]]

  g = ig.Graph(directed=True)
  g.add_vertices(len(nodes))
  g.add_edges(edges)

  source_idx = node_idx[source]
  target_idx = node_idx.get(target, None)

  if target_idx is None:
    return {source}

  idom = g.dominator(source_idx, mode="OUT")

  if idom[target_idx] == -1:
    return {source}

  unavoidable = set()
  node = target_idx
  while node != source_idx:
    if math.isnan(node):
      break
    unavoidable.add(nodes[node])
    node = idom[node]
  unavoidable.add(source)

  return unavoidable
