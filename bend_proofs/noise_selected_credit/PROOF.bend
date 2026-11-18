import ./LAWS.bend as Laws
import ../lib/NAT.bend as Arith
import Base

# PROOF of LAWS.bend. Induction on n, cases on k; the step uses the law at
# (m, q) and (m, q+1) (n-structural, any k). See ../lib/NAT.bend for the
# right-to-left rewrite convention.
#
# Step, with a = C(m,q), b = C(m,q+1), c = C(m+1,q+1) = a + b, e = C(m+1,q+2):
#   (q+2)(c + e) = (c + (q+1)c) + (q+2)e          distribute
#                = (c + (m+1)a) + (m+1)b          law at (m,q), (m,q+1)
#                = (a+b) + ((m+1)a + (m+1)b)      c = a + b, reassociate
#                = (m+2)(a + b)                   distribute

def Laws.absorption(n, k):
  match n:
    case 0n:
      match k:
        case 0n:
          {==}
        case 1n++q:
          Equal.sym(Nat, Nat.mul(1n+1n+q, 0n), 0n, Arith.mul_zero_r(1n+1n+q))
    case 1n++m:
      match k:
        case 0n:
          %Laws.absorption(m, 0n) : {1n+Nat.mul(1n+m, 1n) == 1n+_ : Nat}
          {==}
        case 1n++q:
          %Arith.mul_distrib_l(1n+1n+q, Laws.Sel.choose(1n+q, 1n+m), Laws.Sel.choose(1n+1n+q, 1n+m)) : {Nat.mul(1n+1n+m, Laws.Sel.choose(1n+q, 1n+m)) == _ : Nat}
          %Laws.absorption(m, q) : {Nat.mul(1n+1n+m, Laws.Sel.choose(1n+q, 1n+m)) == Nat.add(Nat.add(Laws.Sel.choose(1n+q, 1n+m), _), Nat.mul(1n+1n+q, Laws.Sel.choose(1n+1n+q, 1n+m))) : Nat}
          %Laws.absorption(m, 1n+q) : {Nat.mul(1n+1n+m, Laws.Sel.choose(1n+q, 1n+m)) == Nat.add(Nat.add(Laws.Sel.choose(1n+q, 1n+m), Nat.mul(1n+m, Laws.Sel.choose(q, m))), _) : Nat}
          %Arith.mul_distrib_l(1n+m, Laws.Sel.choose(q, m), Laws.Sel.choose(1n+q, m)) : {Nat.add(Nat.add(Laws.Sel.choose(q, m), Laws.Sel.choose(1n+q, m)), _) == Nat.add(Nat.add(Nat.add(Laws.Sel.choose(q, m), Laws.Sel.choose(1n+q, m)), Nat.mul(1n+m, Laws.Sel.choose(q, m))), Nat.mul(1n+m, Laws.Sel.choose(1n+q, m))) : Nat}
          Arith.add_assoc_sym(Nat.add(Laws.Sel.choose(q, m), Laws.Sel.choose(1n+q, m)), Nat.mul(1n+m, Laws.Sel.choose(q, m)), Nat.mul(1n+m, Laws.Sel.choose(1n+q, m)))

def main() -> Nat:
  Laws.Sel.choose(2n, 4n)
