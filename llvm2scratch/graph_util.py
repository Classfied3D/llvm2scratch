from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from copy import deepcopy

import networkx as nx
import itertools

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

def minHittingSetExact(cycles: list[list[str]]) -> list[str]:
  """Exact minimum hitting set by brute-force (use only for small instances)."""
  if not cycles:
    return []
  cycle_sets = [set(c) for c in cycles]
  nodes = sorted(set().union(*cycle_sets))
  for r in range(1, len(nodes) + 1):
    for comb in itertools.combinations(nodes, r):
      s = set(comb)
      if all(s & c for c in cycle_sets):
        return list(s)
  return nodes # fallback: return all nodes if nothing found (shouldn't happen)

def greedyHittingSet(cycles: list[list[str]]) -> list[str]:
  """Greedy heuristic hitting set: iteratively choose node that covers most remaining cycles."""
  if not cycles:
    return []
  cycle_sets = [set(c) for c in cycles]
  uncovered = cycle_sets.copy()
  selected: set[str] = set()

  while uncovered:
    # count frequency
    freq: dict[str, int] = {}
    for c in uncovered:
      for n in c:
        freq[n] = freq.get(n, 0) + 1
    # pick node with max frequency (tie-breaker: stable via sorted)
    node = max(sorted(freq.keys()), key=lambda x: freq[x])
    selected.add(node)
    # remove covered cycles
    uncovered = [c for c in uncovered if node not in c]

  return list(selected)

def findAllCycles(graph: dict[str, list[str]]) -> list[list[str]]:
  g = nx.DiGraph(graph)
  return list(nx.simple_cycles(g))

def selectCycleChecks(cycles: list[list[str]], exact_threshold_nodes: int = 15) -> list[str]:
  if not cycles:
    return []
  cycle_nodes = set().union(*[set(c) for c in cycles])
  if len(cycle_nodes) <= exact_threshold_nodes:
    return minHittingSetExact(cycles)
  else:
    return greedyHittingSet(cycles)

def unavoidableNodes(graph: dict[str, list[str]], source: str, target: str) -> set[str]:
  g = nx.DiGraph(graph)
  paths = list(nx.all_simple_paths(g, source, target))
  if not paths:
    return set()
  return set.intersection(*[set(p) for p in paths])
