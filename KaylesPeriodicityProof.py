from Kayles import *

# Returns true if the following theorem is proven and false otherwise:
# The Grundy numbers for a Kayles game of length n are periodic starting from
# n=72 with period 12.
#
# The proof is inductive (base case n<144) as follows:
#   - Let n>=144 with 12|n
#   - We show that grundy(n+offset) = grundy(144+offset):
#     - We find moves that give a grundy number of k for all k < grundy(144+offset)
#     - We show that no move gives a grundy number of grundy(144+offset)
def prove_kayles_periodicity(verbose=False):
  # The period of the game
  period = 12

  # The size of the base case
  base_length = 144

  # The grundy values for the base case.
  base_case = [grundy(n) for n in range(base_length)]

  # The largest value that appears in the base case
  max_grundy = max(base_case)

  # The expected grundy values we will check against.
  # By the inductive hypothesis, we will also use these when computing grundy values of moves.
  expected_grundys = [grundy(base_length + offset) for offset in range(period)]

  # Maps the grundy values that appear only in the base case to a list of indices at which
  # they appear.
  unique_values = { k: [i for i, g in enumerate(base_case) if g == k] for k in base_case if k not in expected_grundys }

  # We will verify that grundy(n+offset) = grundy(144+offset)
  for offset in range(period):
    # This should match grundy(n+offset)
    expected = expected_grundys[offset]

    if verbose:
      print(f'{"-"*20}Showing that grundy(n+{offset})={expected}{"-"*20}')

    # We will verify that some move gives a grundy number of k by a linear search
    for k in range(expected):
      if verbose:
        print(f'Looking for a move that gives {k}')

      # We will break when we've found a move that gives k.
      # Note that it must appear early in the base case so that we don't illegally
      # treat the first half of the base case as periodic.
      for index, val in enumerate(base_case[:base_length//2-2]):
        if verbose:
          print(f' - Try removing ball at {index} (gives {index} and n+{offset}-{index+1})')

        # Try removing the ball at index.
        # This gives parts of length index and n+offset-index-1
        if val ^ expected_grundys[(offset-index-1) % 12] == k:
          if verbose:
            print(f'   - This works! ({val} ^ {expected_grundys[(offset-index-1) % 12]} = {k})')
          break

        if verbose:
          print(f' - Try removing balls at {index} and {index+1} (gives {index} and n+{offset}-{index+2})')

        # Try removing the balls at index and index+1.
        # This gives parts of length index and n+offset-index-2
        if val ^ expected_grundys[(offset-index-2) % 12] == k:
          if verbose:
            print(f'   - This works! ({val} ^ {expected_grundys[(offset-index-2) % 12]} = {k})')
          break
      else:
        # We failed to find a move giving k
        return False

    if verbose:
      print(f'Proving that no move gives {expected}')

    # The pairs of grundy values that would XOR to expected.
    # Note that the sorting is only so that we don't repeat pairs.
    bad_pairs = [[i, i ^ expected] for i in range(max_grundy + 1) if i >= i ^ expected]
    
    # We must check that all pairs give no bad cases
    for pair in bad_pairs:
      # If neither member of the pair has a finite number of occurrences,
      # this proof fails (but the theorem need not be false)
      if pair[0] not in unique_values and pair[1] not in unique_values:
        return False

      unique = pair[0] if pair[0] in unique_values else pair[1]
      nonunique = pair[1] if pair[0] in unique_values else pair[0]

      if verbose:
        print(f' - {expected} = {unique} ^ {nonunique}. Checking that no occurrences of {unique} give {nonunique}')

      # index is a position that has a grundy value equal to unique
      for index in unique_values[unique]:
        if verbose:
          print(f'   - Try removing ball at {index} (gives {index} and n+{offset}-{index+1})')
          print(f'     - grundy(n+{offset}-{index+1})={expected_grundys[(offset-index-1) % 12]}')

        # If we find that grundy(n+offset-index-1)=nonunique then we have found a move that gives a grundy value equal to expected
        if expected_grundys[(offset-index-1) % 12] == nonunique:
          return False
        
        if verbose:
          print(f'   - Try removing balls at {index} and {index+1} (gives {index} and n+{offset}-{index+2})')
          print(f'     - grundy(n+{offset}-{index+2})={expected_grundys[(offset-index-2) % 12]}')

        # If we find that grundy(n+offset-index-2)=nonunique then we have found a move that gives a grundy value equal to expected
        if expected_grundys[(offset-index-2) % 12] == nonunique:
          return False

      if verbose:
        print(f'   - None of the above are {nonunique} so no occurrences of {unique} give {nonunique}')

    if verbose:
      print(f'Thus, grundy(n+{offset})={expected}')

  # If nothing went wrong, the theorem is proved
  return True

if __name__ == "__main__":
  if prove_kayles_periodicity(verbose=True):
    print("The theorem holds")
  else:
    print("The theorem fails")