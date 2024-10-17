import itertools

# The smallest value for which a single line of n stones is periodic.
GRUNDY_PERIOD_START = 72

# A single period of the Grundy value for a single line of n stones.
GRUNDY_PERIOD = [4,1,2,8,1,4,7,2,1,8,2,7]

# A cache of grundy values for state n indexed by n.
# -1 for values that have not been calculated
grundy_cache = [0]

# Calculate the Grundy number for the state n.
# That is, the state with a single line of n adjacent stones.
def simple_grundy(n):
  # For large n, this is periodic. No need to compute manually.
  if n >= GRUNDY_PERIOD_START:
    return GRUNDY_PERIOD[n % 12]

  global grundy_cache

  # If necessary, reallocate a larger Grundy cache
  if n >= len(grundy_cache):
    grundy_cache += [-1] * (n + 1 - len(grundy_cache))

  # Return cached value if it exists
  if grundy_cache[n] >= 0:
    return grundy_cache[n]

  # The grundy numbers of accessible states
  # Uses a set for O(1) lookup when finding the mex
  grundys = set()

  # Check states that are removals of 1
  for i in range((n - 1) // 2 + 1):
    grundys.add(grundy(i, n - i - 1))

  # Check states that are removals of 2
  for i in range((n - 2) // 2 + 1):
    grundys.add(grundy(i, n - i - 2))

  # Cache this value for later
  grundy_cache[n] = mex(grundys)

  return grundy_cache[n]

# Find the minimum excluded natural number from a set
def mex(s):
  for i in itertools.count(start=0):
    if i not in s:
      return i

# Calculate the Grundy number for a state.
# The state is specified as a list of integers representing
# lengths of sequences of adjacent stones.
def grundy(*state):
  # Calculation is simple... XOR all Grundy values of sub-sequences
  ret = 0
  for n in state:
    ret ^= simple_grundy(n)
  return ret

if __name__ == "__main__":
  for i in range(201):
    print(i, grundy(i))