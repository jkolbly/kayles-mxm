import Kayles

# Get the Grundy value of a path with n edges
def grundy_path(n):
  return Kayles.grundy(n)

# A cache of grundy values for an n-fork indexed by n.
# -1 for values that have not been calculated
fork_cache = []

# Get the Grundy value of a fork graph with n edges to the right of the fork.
# This is also the 3-star (n,1,1)
def grundy_fork(n):
  global fork_cache

  # If necessary, allocate a larger cache
  if n >= len(fork_cache):
    fork_cache += [-1] * (n + 1 - len(fork_cache))

  # Return cached value if it exists
  if fork_cache[n] >= 0:
    return fork_cache[n]
  
  # The Grundy values of accessible states
  grundys = set()

  # Remove the middle vertex of the fork
  grundys.add(grundy_path(max(n-1, 0)))

  # Remove one of the split edges of the fork
  grundys.add(grundy_path(n+1))

  # Remove one edge anywhere along the fork's handle
  for i in range(n):
    grundys.add(grundy_path(i) ^ grundy_fork(n-i-1))

  # Remove one vertex anywhere along the fork's handle
  for i in range(n-1):
    grundys.add(grundy_path(i) ^ grundy_fork(n-i-2))

  # Cache this value for later
  fork_cache[n] = Kayles.mex(grundys)

  return fork_cache[n]

# A cache of grundy values for an n-spoon indexed by n.
# -1 for values that have not been calculated
spoon_cache = []

# Get the Grundy value of a spoon graph with n edges to the right of the triangle.
def grundy_spoon(n):
  global spoon_cache

  # If necessary, allocate a larger cache
  if n >= len(spoon_cache):
    spoon_cache += [-1] * (n + 1 - len(spoon_cache))

  # Return cached value if it exists
  if spoon_cache[n] >= 0:
    return spoon_cache[n]
  
  # The Grundy values of accessible states
  grundys = set()

  # Remove the leftmost edge of the spoon
  grundys.add(grundy_fork(n))

  # Remove one of the side edges of the spoon
  grundys.add(grundy_path(n+2))

  # Remove the central vertex
  grundys.add(grundy_path(1) ^ grundy_path(max(0, n-1)))

  # Remove one of the side edges of the spoon
  grundys.add(grundy_path(n+1))

  # Remove one edge anywhere along the spoon's handle
  for i in range(n):
    grundys.add(grundy_path(i) ^ grundy_spoon(n-i-1))

  # Remove one vertex anywhere along the spoon's handle
  for i in range(n-1):
    grundys.add(grundy_path(i) ^ grundy_spoon(n-i-2))

  # Cache this value for later
  spoon_cache[n] = Kayles.mex(grundys)

  return spoon_cache[n]

# A cache of grundy values for stars indexed by the string "length_1,...,length_n".
# where length_1 through length_n are the lengths of each path in ascending order
star_cache = {}

# Get the Grundy value of a star (multiple paths connected at a single vertex).
# Lengths of paths, in edges, are given by *lengths
def grundy_star(*lengths):
  global star_cache

  # Sort lengths and remove all but nonzero lengths
  trimmed_lengths = sorted(length for length in lengths if length > 0)
  string_rep = ",".join(str(l) for l in trimmed_lengths)

  # Check if we've already computed this
  if string_rep in star_cache:
    return star_cache[string_rep]
  
  # Check if this is empty
  if len(trimmed_lengths) == 0:
    return 0
  
  # Check if this is a path
  if len(trimmed_lengths) == 1:
    return grundy_path(trimmed_lengths[0])
  
  # The set of grundy values of accessible states
  grundys = set()

  # Remove the center vertex
  grundys.add(grundy_star(*(length - 1 for length in trimmed_lengths)))

  # Remove one edge from a side
  for i, length in enumerate(trimmed_lengths):
    for n in range(length):
      grundys.add(grundy_path(n) ^ grundy_star(*(l if k != i else l - n - 1 for k, l in enumerate(trimmed_lengths))))

  # Remove one vertex from a side
  for i, length in enumerate(trimmed_lengths):
    for n in range(length - 1):
      grundys.add(grundy_path(n) ^ grundy_star(*(l if k != i else l - n - 2 for k, l in enumerate(trimmed_lengths))))

  # Cache this value for later
  star_cache[string_rep] = Kayles.mex(grundys)

  return star_cache[string_rep]

if __name__ == "__main__":
  for n in range(1001):
    print(n, grundy_fork(n))
  
  for n in range(1001):
    print(n, grundy_spoon(n))