import Base

# LAWS: the mathematical content behind "synaptic hold/credit (e-prop-
# style eligibility traces) approximates true BPTT, exactly in the
# fully-sparse limit, and the error depends only on the number of
# active intermediate ticks, not the total horizon length." See
# README.md for the full derivation and scope.
#
# Setup (derived in README.md, stated here only as the data definitions
# the laws below need): for a residual state update h_t = h_{t-1} + D_t,
# the true BPTT gradient credit flowing back from time T to a synapse
# active at time t is scaled by the full Jacobian product
# PI_{k=t+1}^{T} (I + J_k), where J_k is D_k's own Jacobian w.r.t.
# h_{k-1}. Synaptic-hold/e-prop's approximation replaces this product
# with the identity. The standard distributive/powerset expansion
# PI(1+x_k) = SUM over subsets S of PI_{k in S} x_k (provable over any
# commutative semiring, hence over Nat here, sidestepping the need for
# real-number symbolic arithmetic Bend's checker can't reduce) is
# exactly the error term's structure: J_k=0 (no state change that
# tick -- sparse activity) makes every subset containing k vanish, so
# the error collapses to subsets of ONLY the active ticks.
#
# List.product_plus_one(xs) models PI_{k} (1+x_k) for a list of
# (abstract, Nat-valued) per-tick Jacobian-correction magnitudes.
# List.powerset_products(xs) is the list of PI_S over every subset S
# (the powerset, order-independent for this purpose). List.sum sums a
# list; List.sum_scaled/scale_list are fused helpers avoiding a
# quantity mismatch with Base's List.map (see README.md).

def List.sum(a, xs: List<a, Nat>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case h <> t:
      Nat.add(h, List.sum(a, t))

def List.sum_scaled(a, +h: Nat, xs: List<a, Nat>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case e <> t:
      Nat.add(Nat.mul(h, e), List.sum_scaled(a, h, t))

def List.scale_list(+h: Nat, xs: +List<Nat>) -> +List<Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case e <> t:
      Nat.mul(h, e) <> List.scale_list(h, t)

def List.powerset_products(xs: +List<Nat>) -> +List<Nat>:
  match xs:
    case Nil{}:
      [1n]
    case h <> +t:
      List.append(&2, Nat, List.powerset_products(t), List.scale_list(h, List.powerset_products(t)))

def List.product_plus_one(xs: +List<Nat>) -> Nat:
  match xs:
    case Nil{}:
      1n
    case h <> t:
      Nat.mul(1n+h, List.product_plus_one(t))

# THE MAIN THEOREM: PI(1+x_k) over a list of per-tick Jacobian
# corrections equals the sum, over every subset of ticks, of the
# product of just that subset's corrections. This is the exact
# mathematical content of "the eligibility-trace estimate (which
# assumes the Jacobian product is the identity) differs from the true
# BPTT product by a sum over NONEMPTY subsets" -- and since any subset
# containing an inactive (x_k=0) tick contributes zero (Nat.mul
# absorbs 0), the error depends only on the active ticks, never on how
# many inactive ticks sit in between them.
law powerset_expansion:
  for +xs: +List<Nat>
  {List.product_plus_one(xs) == List.sum(&2, List.powerset_products(xs)) : Nat}

# THE KEY COROLLARY, made concrete: n consecutive fully-inactive ticks
# (every correction magnitude exactly 0) give product_plus_one == 1n
# EXACTLY -- not approximately, not "small error" -- regardless of n.
# This is "exact in the fully-sparse limit, independent of horizon
# length" as a literal, checked equality, not an asymptotic claim.
law all_inactive_is_exact:
  for n: Nat
  {List.product_plus_one(List.replicate(Nat, n, 0n)) == 1n : Nat}
