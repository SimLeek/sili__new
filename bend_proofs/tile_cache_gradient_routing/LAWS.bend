import Base

# LAWS: the concat/split gradient-routing fact underlying tile-cache
# exact credit (sili_peridot's model/tile_cache_credit.py). See README.md
# for scope -- what this does and does not establish.
#
# step_cached() builds `k_content_full = concat([cache_k_tensor, k_new],
# axis=0)` (and the same shape for v). When the overall loss's backward
# reaches this concat, autodiff hands it one combined gradient `dy` sized
# to the WHOLE concatenated list, and the concat's own backward rule
# must route each half of `dy` back to the matching input -- the cached
# block gets `dy`'s first `length(cache)` entries, the new row gets the
# rest. If that routing were off by even one position, `cache_k_tensor
# .grad` (what tile_cache_credit.py reads back out) would silently hold
# the WRONG per-row values -- plausible-looking numbers, not a crash,
# exactly the kind of bug an empirical "it ran without crashing, and
# quality went up" test cannot rule out on its own.
#
# This file states that exact routing fact as a law: splitting an
# appended list back apart at the same boundary used to build it
# recovers the two original pieces, in order, with none dropped,
# duplicated, or swapped. `F32` stands for one row of a K/V vector's
# gradient (the real engine's rows are arrays of F32; this proof reasons
# over the routing of WHOLE rows, see README.md's scope note on why rows
# are the right unit here, not individual floats).

law take_append_left:
  for a: List<F32>
  for b: List<F32>
  {List.take(&1, F32, List.append(&1, F32, a, b), List.length(&1, F32, a)) == a : List<F32>}

law drop_append_right:
  for a: List<F32>
  for b: List<F32>
  {List.drop(&1, F32, List.append(&1, F32, a, b), List.length(&1, F32, a)) == b : List<F32>}

# NOTE (not a law here -- see README.md's scope section): the
# two-cache-use case (cache_k_tensor feeds BOTH pass 1 and pass 2 of
# step_cached(), via the same `k_content_full`) relies additionally on
# the standard multivariate chain rule's sum rule for a fanned-out
# node -- that the total derivative through a value used in two places
# is the SUM of the two paths' local derivatives. That is a standard
# calculus fact taken as given (exactly as the existing Lean proof in
# this repo takes its own non-independence hypothesis as externally
# supplied, never derived from nothing), not something re-derived here.
# What IS in scope and proven above is that each individual path's
# local contribution is routed to the correct rows before being summed
# -- i.e. that `_acc`'s two separate calls each add the RIGHT numbers,
# not that addition itself is the right combining rule (that part is
# calculus, not a list-indexing fact, and float addition is an opaque
# host primitive here, not something Bend's checker can reduce
# symbolically for abstract variables).
