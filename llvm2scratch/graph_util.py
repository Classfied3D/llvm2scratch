from collections import defaultdict
import itertools

def tarjan_scc(graph: dict[str, list[str]]) -> list[list[str]]:
  """Tarjan's algorithm: returns list of SCCs (each SCC is a list of nodes)."""
  index = 0
  indices: dict[str, int] = {}
  lowlink: dict[str, int] = {}
  stack: list[str] = []
  on_stack: set[str] = set()
  sccs: list[list[str]] = []

  def strongconnect(v: str):
    nonlocal index
    indices[v] = index
    lowlink[v] = index
    index += 1
    stack.append(v)
    on_stack.add(v)

    for w in graph.get(v, []):
      if w not in indices:
        strongconnect(w)
        lowlink[v] = min(lowlink[v], lowlink[w])
      elif w in on_stack:
        lowlink[v] = min(lowlink[v], indices[w])

    if lowlink[v] == indices[v]:
      comp: list[str] = []
      while True:
        w = stack.pop()
        on_stack.remove(w)
        comp.append(w)
        if w == v:
          break
      sccs.append(comp)

  for node in graph:
    if node not in indices:
      strongconnect(node)

  return sccs

def enumerate_cycles_in_scc(graph: dict[str, list[str]], scc_nodes: list[str]) -> list[list[str]]:
  """
  Enumerate all elementary cycles contained entirely inside scc_nodes.
  Iterative DFS starting from each start node s (ordered) and only
  exploring nodes whose index >= index(s) prevents duplicates.
  Each returned cycle is a list of nodes [s, ..., x] meaning s -> ... -> x -> s.
  """
  cycles: list[list[str]] = []
  if not scc_nodes:
    return cycles

  # Deterministic order
  nodes = list(scc_nodes)
  node_index = {n: i for i, n in enumerate(nodes)}
  # Pre-filter adjacency to only include edges inside the SCC
  adj = {n: [w for w in graph.get(n, []) if w in node_index] for n in nodes}

  n = len(nodes)
  for s_idx in range(n):
    start = nodes[s_idx]

    # If self-loop, record it
    if start in adj.get(start, []):
      cycles.append([start])

    # Iterative DFS stack: tuples (current, neighbor_iter, path)
    # We'll simulate recursion while enforcing neighbor_index >= s_idx
    path: list[str] = [start]
    in_path: set[str] = {start}
    # neighbor iterators stack; each element is iter(list_of_neighbors)
    neigh_iters: list = [iter(adj.get(start, []))]

    while neigh_iters:
      try:
        neighbor = next(neigh_iters[-1])
      except StopIteration:
        # finished neighbours for current node -> backtrack
        neigh_iters.pop()
        if path:
          in_path.remove(path.pop())
        continue

      # only consider neighbors with index >= s_idx to avoid duplicates
      if node_index[neighbor] < s_idx:
        continue

      if neighbor == start:
        # Found a cycle: the current path already contains start..current,
        # so record a copy of path (start .. current). This cycle represents start->...->current->start.
        cycles.append(path.copy())
        continue

      if neighbor in in_path:
        # would create non-elementary cycle (revisit) -> skip
        continue

      # go deeper
      path.append(neighbor)
      in_path.add(neighbor)
      neigh_iters.append(iter(adj.get(neighbor, [])))

    # done exploring cycles that have minimal node index == s_idx
  return cycles

def find_all_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
  """
  Find all elementary cycles in the graph.
  We compute SCCs first, then enumerate cycles only inside SCCs of size>1
  or single node with self-loop.
  """
  cycles: list[list[str]] = []
  sccs = tarjan_scc(graph)
  for scc in sccs:
    if len(scc) == 1:
      v = scc[0]
      if v in graph.get(v, []):  # self-loop
        cycles.append([v])
    else:
      cycles.extend(enumerate_cycles_in_scc(graph, scc))
  return cycles

def minimum_hitting_set_exact(cycles: list[list[str]]) -> list[str]:
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
  return nodes  # fallback: return all nodes if nothing found (shouldn't happen)

def greedy_hitting_set(cycles: list[list[str]]) -> list[str]:
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

def select_cycle_checks(cycles: list[list[str]], exact_threshold_nodes: int = 15) -> list[str]:
  """
  - Enumerates ALL cycles in the graph (guaranteed).
  - If unique nodes involved in cycles <= exact_threshold_nodes -> compute exact minimum hitting set.
  - Else -> use greedy heuristic to produce a (fast) hitting set.
  Returns a list of nodes that hit every cycle.
  """
  # If no cycles, nothing to do
  if not cycles:
    return []

  # All nodes that appear in any cycle
  cycle_nodes = set().union(*[set(c) for c in cycles])

  if len(cycle_nodes) <= exact_threshold_nodes:
    return minimum_hitting_set_exact(cycles)
  else:
    return greedy_hitting_set(cycles)


def find_paths(graph: dict[str, list[str]], start: str, end: str, path: list[str] | None = None) -> list[list[str]]:
  """Return all simple paths from start to end in a graph stored as dict[str, list[str]]."""
  if path is None:
    path = []
  path = path + [start]

  if start == end:
    return [path]

  paths: list[list[str]] = []
  for neighbor in graph.get(start, []): # safely handle nodes with no outgoing edges
    if neighbor not in path: # avoid cycles
      new_paths = find_paths(graph, neighbor, end, path)
      for p in new_paths:
        paths.append(p)
  return paths


def unavoidable_nodes(graph: dict[str, list[str]], source: str, target: str) -> set[str]:
  """Find nodes that appear in every path from (and including) source to target."""
  paths: list[list[str]] = find_paths(graph, source, target)
  if not paths:
    return set()

  # Convert each path into a set
  path_sets: list[set[str]] = [set(p) for p in paths]

  # Intersection across all paths
  unavoidable: set[str] = set.intersection(*path_sets)

  return unavoidable
