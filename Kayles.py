# Calculate the Grundy number for the state n.
# That is, the state with a single line of n adjacent stones.
def simple_grundy(n):
  pass

# Calculate the Grundy number for a state.
# The state is specified as a list of integers representing
# lengths of sequences of adjacent stones.
def grundy(*state):
  # Calculation is simple... XOR all Grundy values of sub-sequences
  ret = 0
  for n in state:
    ret ^= simple_grundy(n)
  return ret