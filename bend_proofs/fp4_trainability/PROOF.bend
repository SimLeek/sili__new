import ./LAWS.bend as Laws
import ../lib/NAT.bend as Arith
import Base

# PROOF of LAWS.bend. See ../lib/NAT.bend for the right-to-left rewrite
# convention.
#
# sr_unbiased, induction on a, with W = d + e and Y = (m + W)*d:
#   (1+m)e + (1+m+W)d = (e + me) + (d + Y)
#                     = (d + e) + (me + Y)       middle swap, add_comm
#                     = W + (m + d)W             induction hypothesis
#                     = (1 + m + d)W

def Laws.sr_unbiased(a, d, e):
  match a:
    case 0n:
      Arith.mul_comm(Nat.add(d, e), d)
    case 1n++m:
      %Laws.sr_unbiased(m, d, e) : {Nat.add(Nat.add(e, Nat.mul(m, e)), Nat.add(d, Nat.mul(Nat.add(m, Nat.add(d, e)), d))) == Nat.add(Nat.add(d, e), _) : Nat}
      %Arith.add_comm(e, d) : {Nat.add(Nat.add(e, Nat.mul(m, e)), Nat.add(d, Nat.mul(Nat.add(m, Nat.add(d, e)), d))) == Nat.add(_, Nat.add(Nat.mul(m, e), Nat.mul(Nat.add(m, Nat.add(d, e)), d))) : Nat}
      Arith.add_middle_swap(e, Nat.mul(m, e), d, Nat.mul(Nat.add(m, Nat.add(d, e)), d))

# sr_second_moment: distribute d*e over (d + e), then reassociate each term.
def Laws.sr_second_moment(d, e):
  %Arith.mul_distrib_l(Nat.mul(d, e), d, e) : {Nat.add(Nat.mul(e, Nat.mul(d, d)), Nat.mul(d, Nat.mul(e, e))) == _ : Nat}
  %Arith.mul_assoc(d, e, e) : {Nat.add(Nat.mul(e, Nat.mul(d, d)), _) == Nat.add(Nat.mul(Nat.mul(d, e), d), Nat.mul(Nat.mul(d, e), e)) : Nat}
  %Equal.sym(Nat, Nat.mul(Nat.mul(d, e), d), Nat.mul(d, Nat.mul(e, d)), Arith.mul_assoc(d, e, d)) : {Nat.add(Nat.mul(e, Nat.mul(d, d)), Nat.mul(Nat.mul(d, e), e)) == Nat.add(_, Nat.mul(Nat.mul(d, e), e)) : Nat}
  %Arith.mul_swap_l(e, d, d) : {Nat.add(Nat.mul(e, Nat.mul(d, d)), Nat.mul(Nat.mul(d, e), e)) == Nat.add(_, Nat.mul(Nat.mul(d, e), e)) : Nat}
  {==}

# inc_value: every case but the carry computes; the carry D4 -> D0 uses the
# hypothesis on the tail and 5(1 + V) = 5 + 5V.
def Laws.inc_value(xs):
  match xs:
    case Nil{}:
      {==}
    case Con{h, t}:
      match h:
        case Laws.D0{}:
          {==}
        case Laws.D1{}:
          {==}
        case Laws.D2{}:
          {==}
        case Laws.D3{}:
          {==}
        case Laws.D4{}:
          %Equal.sym(Nat, Laws.Fp4.value(Laws.Fp4.inc(t)), 1n+Laws.Fp4.value(t), Laws.inc_value(t)) : {Nat.mul(5n, _) == 1n+Nat.add(4n, Nat.mul(5n, Laws.Fp4.value(t))) : Nat}
          %Arith.mul_succ_r_sym(5n, Laws.Fp4.value(t)) : {_ == 1n+Nat.add(4n, Nat.mul(5n, Laws.Fp4.value(t))) : Nat}
          {==}

def Laws.encode_value(n):
  match n:
    case 0n:
      {==}
    case 1n++p:
      %Laws.encode_value(p) : {Laws.Fp4.value(Laws.Fp4.inc(Laws.Fp4.encode(p))) == 1n+_ : Nat}
      Laws.inc_value(Laws.Fp4.encode(p))

def main() -> Nat:
  Laws.Fp4.value(Laws.Fp4.encode(37n))
