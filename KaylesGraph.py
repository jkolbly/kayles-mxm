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

if __name__ == "__main__":
  for n in range(1001):
    print(n, grundy_fork(n))