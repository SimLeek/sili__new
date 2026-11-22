import Base

# LAWS: why a gradient-independent (noise-scored) top-k selection of memory
# dims gives an UNBIASED sparse credit, up to the constant scale k/d. See
# README.md for the setting (sili_quartz e-BPTT credit_density/credit_score).
#
# Sel.choose(k, n) is the binomial coefficient C(n, k) (k first: Bend matches binders in order), defined by Pascal's
# rule (matching on k first so C(n, 0) = 1 reduces for any n).

def Sel.choose(k: Nat, +n: Nat) -> Nat:
  match k:
    case 0n:
      1n
    case 1n++q:
      match n:
        case 0n:
          0n
        case 1n++m:
          Nat.add(Sel.choose(q, m), Sel.choose(1n+q, m))

# LAW 1, THE MAIN THEOREM (absorption / equal coverage). Among the
# C(n+1, k+1) ways to keep k+1 of n+1 memory dims, a given dim is kept in
# C(n, k) of them -- the same count for every dim. Counting (dim, kept-set)
# incidences both ways: (n+1) dims each in C(n, k) sets, or C(n+1, k+1) sets
# each holding k+1 dims. So when the selection does not depend on G and the
# score is exchangeable across dims, every dim is kept with the same
# probability (k+1)/(n+1), and the expected sparse credit is exactly that
# fraction of the dense credit, dim by dim.
law absorption:
  for +n: Nat
  for +k: Nat
  {Nat.mul(1n+n, Sel.choose(k, n)) == Nat.mul(1n+k, Sel.choose(1n+k, 1n+n)) : Nat}
