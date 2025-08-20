from collections import defaultdict, deque

def strongly_connected_components(graph: dict[str, list[str]]) -> list[list[str]]:
  """Tarjan's algorithm to find strongly connected components"""
  index = 0
  stack = []
  indices = {}
  lowlink = {}
  on_stack = set()
  sccs = []

  def strongconnect(node):
    nonlocal index
    indices[node] = index
    lowlink[node] = index
    index += 1
    stack.append(node)
    on_stack.add(node)

    for neighbor in graph.get(node, []):
      if neighbor not in indices:
        strongconnect(neighbor)
        lowlink[node] = min(lowlink[node], lowlink[neighbor])
      elif neighbor in on_stack:
        lowlink[node] = min(lowlink[node], indices[neighbor])

    # If node is a root of an SCC
    if lowlink[node] == indices[node]:
      scc = []
      while True:
        w = stack.pop()
        on_stack.remove(w)
        scc.append(w)
        if w == node:
          break
      sccs.append(scc)

  for node in graph:
    if node not in indices:
      strongconnect(node)

  return sccs

def select_minimum_checks_scc(graph: dict[str, list[str]]) -> list[str]:
  """Select one representative node per SCC that contains a cycle.
  Returns a list of nodes where stack checks should be placed."""
  sccs = strongly_connected_components(graph)
  checks = []

  for scc in sccs:
    if len(scc) > 1:
      # multi-node SCC -> definite recursion
      checks.append(scc[0])
    elif len(scc) == 1:
      node = scc[0]
      if node in graph.get(node, []): # self-loop
        checks.append(node)

  return checks

def topological_sort(graph: dict[str, list[str]]):
  indegree = defaultdict(int)
  for u in graph:
    for v in graph[u]:
      indegree[v] += 1
    indegree.setdefault(u, 0)
  
  q = deque([u for u in indegree if indegree[u] == 0])
  order = []
  while q:
    u = q.popleft()
    order.append(u)
    for v in graph.get(u, []):
      indegree[v] -= 1
      if indegree[v] == 0:
        q.append(v)
  return order

def longest_path_dag(graph: dict[str, list[str]], start: str, end: str):
  """Find the longest path between two paths"""
  order = topological_sort(graph)
  dist = {u: float("-inf") for u in order}
  parent = {u: None for u in order}
  
  dist[start] = 0
  
  for u in order:
    if dist[u] != float("-inf"):
      for v in graph.get(u, []):
        if dist[u] + 1 > dist[v]:
          dist[v] = dist[u] + 1
          parent[v] = u
  
  # Reconstruct path
  if dist[end] == float("-inf"):
    return None, float("-inf")
  
  path = []
  node = end
  while node is not None:
    path.append(node)
    node = parent[node]
  path.reverse()
  
  return path, dist[end]