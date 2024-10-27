import networkx as nx
import Kayles

# Maps graph hashes to grundy values
grundy_cache = {}

# Get the grundy value of an arbitrary graph
def grundy(graph: nx.Graph) -> int:
  # The base case is an empty graph (grundy 0)
  if len(graph.edges) == 0:
    return 0
  
  # XOR the Grundy values of each connected component
  components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
  if len(components) > 1:
    ret = 0
    for component in components:
      ret ^= grundy(component)
    return ret
  
  # Check the cache to avoid recomputation
  graph_hash = nx.weisfeiler_lehman_graph_hash(graph)
  if graph_hash in grundy_cache:
    return grundy_cache[graph_hash]

  # The grundy values of accessible states
  grundys = set()

  # Remove a single edge and trim orphaned vertices
  for e in graph.edges:
    new_graph = graph.copy()
    new_graph.remove_edge(e[0], e[1])
    if new_graph.degree[e[0]] == 0:
      new_graph.remove_node(e[0])
    if new_graph.degree[e[1]] == 0:
      new_graph.remove_node(e[1])
    grundys.add(grundy(new_graph))

  # Remove a vertex and trim orphaned vertices
  for v in graph.nodes:
    new_graph = graph.copy()
    neighbors = [e[1] for e in graph.edges(v)]
    new_graph.remove_node(v)
    for n in neighbors:
      if new_graph.degree[n] == 0:
        new_graph.remove_node(n)
    grundys.add(grundy(new_graph))

  # The grundy value is the mex of the computed grundys
  ret = Kayles.mex(grundys)
  grundy_cache[graph_hash] = ret
  return ret

# Generate a graph that is a path with n edges with a fork coming off of each vertex
def path_of_forks(n: int) -> nx.Graph:
  G: nx.Graph = nx.path_graph(n=n+1)
  for v in range(n+1):
    G.add_nodes_from([-3 * (v+1), -3 * (v+1) + 1, -3 * (v+1) + 2])
    G.add_edges_from([
      [v, -3 * (v+1)],
      [-3 * (v+1), -3 * (v+1) + 1],
      [-3 * (v+1), -3 * (v+1) + 2]
    ])
  return G

# Generate a graph that is path_of_forks(n) with num_spoons of the forks converted to spoons
def path_of_fork_spoons(n: int, num_spoons: int) -> nx.Graph:
  G = path_of_forks(n)
  G.add_edges_from([[-3 * (v+1) + 1, -3 * (v+1) + 2] for v in range(min(n+1, num_spoons))])
  return G

if __name__ == "__main__":
  for i in range(1000):
    G = nx.path_graph(n=i+1)
    print(i, grundy(G), Kayles.grundy(i))
    assert grundy(G) == Kayles.grundy(i)