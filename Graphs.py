import networkx as nx
import Kayles
import matplotlib.pyplot as plt
from tabulate import tabulate
from networkx_viewer import Viewer

# The file in which grundy_cache is stored
CACHE_FILENAME = "data/graph-cache"

# How many graphs to compute before updating the cache file
CACHE_BUFFER_SIZE = 1000

# Maps graph hashes to lists of arrays [G,g] of a graph or sparse6 graph representation and a grundy value
grundy_cache = {}

# A list of lines to add to the cache file when next updating it
cache_buffer = []

# Represents a single move in a game
class Move:
  # The graph given to the next player
  graph: nx.Graph

  # The first vertex involved
  vertex0: int

  # The second vertex, in the case of removing an edge
  vertex1: int | None

  # The grundy value of the graph given by this move
  grundy: int

  # The english string of this move (e.g. "vertex 0", "edge 0-1")
  english_str: str

  # The shorthand string of this move (e.g. "0", "0 1")
  short_str: str

  def __init__(self, graph: nx.Graph, vertex0: int, vertex1: int | None = None):
    self.graph = graph
    self.vertex0 = vertex0
    self.vertex1 = vertex1
    self.grundy = grundy(graph)
    self.english_str = f"vertex {vertex0}" if vertex1 is None else f"edge {vertex0}-{vertex1}"
    self.short_str = str(vertex0) if vertex1 is None else f"{vertex0} {vertex1}"

# Write a single value to the grundy cache
def write_to_cache(graph: nx.Graph, grundy: int, graph_hash: str=None):
  if graph_hash is None:
    graph_hash = nx.weisfeiler_lehman_graph_hash(graph)

  if graph_hash in grundy_cache:
    grundy_cache[graph_hash].append([graph, grundy])
  else:
    grundy_cache[graph_hash] = [[graph, grundy]]

  string_rep = nx.to_sparse6_bytes(graph).decode("utf-8")

  cache_buffer.append(f'{graph_hash},{grundy},{string_rep}')

  if len(cache_buffer) >= CACHE_BUFFER_SIZE:
    flush_cache_buffer()
  
# Append the cache buffer to the cache file and clear the buffer
def flush_cache_buffer():
  with open(CACHE_FILENAME, "a+") as f:
    f.writelines(cache_buffer)
  cache_buffer.clear()

# Load grundy_cache from a file
def load_cache():
  global grundy_cache
  grundy_cache = {}

  print("Loading cache...")
  try:
    with open(CACHE_FILENAME) as f:
      lines = f.readlines()
      for line in lines:
        split = line.strip().split(",")
        grundy = int(split[1])
        if split[0] in grundy_cache:
          grundy_cache[split[0]].append([split[2].encode("utf-8"), grundy])
        else:
          grundy_cache[split[0]] = [[split[2].encode("utf-8"), grundy]]
  except OSError:
    pass
  print("Cache loaded")

# Get all states accessible from a graph
def get_moves(graph: nx.Graph) -> list[Move]:
  accessible_states = []

  # Remove a single edge and trim orphaned vertices
  for e in graph.edges:
    new_graph = graph.copy()
    new_graph.remove_edge(e[0], e[1])
    if new_graph.degree[e[0]] == 0:
      new_graph.remove_node(e[0])
    if new_graph.degree[e[1]] == 0:
      new_graph.remove_node(e[1])
    accessible_states.append(Move(new_graph, e[0], e[1]))

  # Remove a vertex and trim orphaned vertices
  for v in graph.nodes:
    new_graph = graph.copy()
    neighbors = [e[1] for e in graph.edges(v)]
    new_graph.remove_node(v)
    for n in neighbors:
      if new_graph.degree[n] == 0:
        new_graph.remove_node(n)
    accessible_states.append(Move(new_graph, v, None))
  
  return accessible_states

# Get all states accessible from a graph, removing isomorphic states
def get_moves_unique(graph: nx.Graph) -> list[Move]:
  isoclasses = {}
  
  for m in get_moves(graph):
    hash = nx.weisfeiler_lehman_graph_hash(m.graph)
    if hash in isoclasses:
      if any(nx.is_isomorphic(m.graph, g.graph) for g in isoclasses[hash]):
        isoclasses[hash].append(m)
    else:
      isoclasses[hash] = [m]

  return [iclass[0] for iclass in isoclasses.values()]

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
    for cache_line in grundy_cache[graph_hash]:
      if type(cache_line[0]) is bytes:
        cache_line[0] = nx.from_sparse6_bytes(cache_line[0])
      if nx.is_isomorphic(cache_line[0], graph):
        return cache_line[1]

  # The grundy values of accessible states
  grundys = set(m.grundy for m in get_moves(graph))

  # The grundy value is the mex of the computed grundys
  ret = Kayles.mex(grundys)
  write_to_cache(graph, ret, graph_hash)
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

# Generate a graph that is two paths with n edges connected like a ladder
def ladder(n: int) -> nx.Graph:
  G: nx.Graph = nx.path_graph(n=n+1)
  nx.add_path(G, [-i-1 for i in range(n+1)])
  G.add_edges_from([(i, -i-1) for i in range(n+1)])
  return G

# Generate a graph that is a path with n edges with one terminal vertex in a cycle of length m
def lollipop(n: int, m: int) -> nx.Graph:
  if m <= 2:
    raise ValueError("A cycle must be of length at least 3")
  G: nx.Graph = nx.path_graph(n=n+1)
  nx.add_path(G, [-i for i in range(m)])
  G.add_edge(-m+1, 0)
  return G

# Generate a graph that is a path with n edges connected on either side to cycles of length m and k
def barbell(n: int, m: int, k: int) -> nx.Graph:
  if m <= 2 or k <= 2:
    raise ValueError("A cycle must be of length at least 3")
  G: nx.Graph = nx.path_graph(n=n+1)
  nx.add_path(G, [-i for i in range(m)])
  G.add_edge(-m+1, 0)
  nx.add_path(G, [i for i in range(n, n+k)])
  G.add_edge(n+k-1, n)
  return G

# Generate a path of length n where nodes are also connected to neighbors a distance of 2 away
def double_path(n: int) -> nx.Graph:
  G: nx.Graph = nx.path_graph(n=n+1)
  G.add_edges_from([(i, i+2) for i in range(n-1)])
  return G

# Generate a graph that looks like |_|_|_|_|_| (n=5)
# `closed_right` and `closed_left` determine whether the last edges is present
def spikes(n: int, closed_left: bool=True, closed_right: bool=True) -> nx.Graph:
  G: nx.Graph = nx.path_graph(n=n+1)
  new_node_num = n+1 if closed_left and closed_right else n if closed_left or closed_right else n-1
  G.add_nodes_from([n+i+1 for i in range(new_node_num)])
  G.add_edges_from([(i if closed_left else i+1,n+i+1) for i in range(new_node_num)])
  return G

# Generate a fork graph
def fork(n: int) -> nx.Graph:
  G: nx.Graph = nx.path_graph(n=n+2)
  G.add_node(-1)
  G.add_edge(-1, 1)
  return G

# Generate a spoon graph
def spoon(n: int) -> nx.Graph:
  G: nx.Graph = nx.path_graph(n=n+2)
  G.add_node(-1)
  G.add_edge(-1, 0)
  G.add_edge(-1, 1)
  return G

# Visually display a graph
def show_graph(graph: nx.Graph, positions: dict = None, with_labels: bool = True, save: bool = False, **kwargs):
  draw_kwargs = {
    "font_weight": 'bold',
  }
  draw_kwargs = dict(draw_kwargs, **kwargs)

  if positions is not None:
    nx.draw(graph, with_labels=with_labels, pos=positions, **draw_kwargs)
  elif nx.is_planar(graph):
    nx.draw_planar(graph, with_labels=with_labels, **draw_kwargs)
  else:
    nx.draw(graph, with_labels=with_labels, **draw_kwargs)

  # Scale axes equally
  plt.gca().set_aspect('equal', adjustable='box')

  if save:
    plt.savefig("out.png", transparent=True, bbox_inches="tight", pad_inches=0)
  
  plt.show()

# Return the best grundy value from a file of graph6-format graphs
def get_best_from_file(filename: str):
  max_so_far = 0
  with open(filename) as f:
    while f.readable():
      line = f.readline().strip()
      if len(line) == 0:
        break
      G = nx.from_graph6_bytes(line.encode("utf-8"))
      if grundy(G) > max_so_far:
        max_so_far = grundy(G)
  return max_so_far

# Prompt the user for a move and return it.
def prompt_user_move(graph: nx.Graph, positions: dict = None) -> Move:
  print("Input a move as either a single number ('0') for a vertex or two numbers ('0 1') for an edge.")
  print("Type 'show' to show the graph again.")
  
  while True:
    move_str = input("Type move selection: ")

    if move_str.lower() in ["show", "s"]:
      show_graph(graph, positions)
      continue

    move_split = move_str.split(" ")

    try:
      move_nums = [int(s) for s in move_split]
      if len(move_nums) == 1:
        if move_nums[0] not in graph.nodes:
          print(f"{n} is not a node in the graph.")
          continue
        neighbors = [e[1] for e in graph.edges(move_nums[0])]
        new_graph = graph.copy()
        new_graph.remove_node(move_nums[0])
        for n in neighbors:
          if new_graph.degree[n] == 0:
            new_graph.remove_node(n)
        return Move(new_graph, move_nums[0])
      elif len(move_nums) == 2:
        if not graph.has_edge(move_nums[0], move_nums[1]):
          print(f"{move_nums[0]} {move_nums[1]} is not an edge in the graph.")
          continue
        new_graph = graph.copy()
        new_graph.remove_edge(move_nums[0], move_nums[1])
        if new_graph.degree[move_nums[0]] == 0:
          new_graph.remove_node(move_nums[0])
        if new_graph.degree[move_nums[1]] == 0:
          new_graph.remove_node(move_nums[1])
        return Move(new_graph, move_nums[0], move_nums[1])
      else:
        print("Your move must be one or two integers separated by a space.")
        continue
    except ValueError:
      print("Your move should be either one or two integers separated by a space.")
      continue

# Play a game against the computer using a text interface
def play_game(graph: nx.Graph, user_first: bool, positions: dict = None, always_select_computer_move: bool = False):
  # true if it's the user's turn, false otherwise
  user_turn = user_first

  positions = positions if positions is not None else nx.planar_layout(graph) if nx.is_planar(graph) else nx.spring_layout(graph)

  while len(graph.edges) > 0:
    show_graph(graph, positions)
    if user_turn:
      move_strs = [f"{g.english_str} ({g.grundy})" for g in get_moves_unique(graph)]
      print(f"Available moves: {', '.join(move_strs)}")
      move = prompt_user_move(graph, positions)
      graph = move.graph
      print(f"Player removes {move.english_str}")
      user_turn = False
    else:
      moves = get_moves_unique(graph)
      grundys = [move.grundy for move in moves]
      min_grundy = min(grundys)
      optimal_moves = moves if min_grundy > 0 else [m for m in moves if m.grundy == 0]

      selected_move = optimal_moves[0]
      if len(optimal_moves) > 1 or always_select_computer_move:
        print(f"Optimal computer moves are: {', '.join(f'{m.english_str} ({m.grundy})' for m in optimal_moves)}")
        print("Select the move the computer will play.")
        selected_move = prompt_user_move(graph, positions)

      print(f"Computer removes {selected_move.english_str}")
      graph = selected_move.graph
      user_turn = True
  
  if user_turn:
    print("Computer wins!")
  else:
    print("You win!")

# Load the grundy cache
load_cache()

try:
  if __name__ == "__main__":  
    n = 3
    m = 5
    positions = {i:[0, i] for i in range(n)}
    positions |= { i: [1,i-n-0.5] for i in range(n, n+m-1) }
    positions |= { n+m-1: [-1,1] }

    while True:
      play_game(nx.complete_bipartite_graph(3, 5), False, positions=positions)
except KeyboardInterrupt:
  pass

print("Writing to cache file...")
flush_cache_buffer()