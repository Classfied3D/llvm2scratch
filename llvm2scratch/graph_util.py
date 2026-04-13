from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from copy import deepcopy
import networkx as nx
import itertools

@dataclass
class NodeInfo:
  might_call: set[str]
  might_modify: set[str]
  dependent: set[str]
  use_counts: Counter[str] = field(default_factory=Counter)

def analyzeCallGraph(fn_info: dict[str, NodeInfo]) -> dict[str, NodeInfo]:
  g = nx.DiGraph()
  for name, info in fn_info.items():
    g.add_node(name)
    for callee in info.might_call:
      g.add_edge(name, callee)

  tc = nx.transitive_closure(g)

  result = deepcopy(fn_info)
  for name in fn_info:
    for callee in tc.successors(name):
      callee_info = fn_info[callee]
      result[name].might_call |= callee_info.might_call
      result[name].might_modify |= callee_info.might_modify
      result[name].dependent |= callee_info.dependent
      result[name].use_counts += callee_info.use_counts

  return result

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
