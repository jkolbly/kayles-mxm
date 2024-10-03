import itertools

# Calculate the Grundy number for the state n.
# That is, the state with a single line of n adjacent stones.
def simple_grundy(n):
  if n == 0:
    return 0

  # The grundy numbers  of accessible states
  # Uses a set for O(1) lookup when finding the mex
  grundys = set()

  # Check states that are removals of 1
  for i in range((n - 1) // 2 + 1):
    grundys.add(grundy(i, n - i - 1))

  # Check states that are removals of 2
  for i in range((n - 2) // 2 + 1):
    grundys.add(grundy(i, n - i - 2))

  return mex(grundys)

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