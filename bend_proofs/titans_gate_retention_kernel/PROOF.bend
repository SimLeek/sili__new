import ./LAWS.bend as Laws
import ../lib/NAT.bend as Arith
import Base

# PROOF of the four laws in LAWS.bend. The Nat semiring scaffolding these
# rest on lives in ../lib/NAT.bend (shared with the other proofs rather
# than copied in); that file also documents the right-to-left rewrite
# direction, which is what decides between a lemma and its _sym twin.

# LAW 1. Induction on n. The base case is just add_comm on b + eta. The
# step rewrites the definitional side by the hypothesis, distributes both
# sides, and then needs exactly mul_swap_l (to exchange the two gate
# factors) and add_swap_r (to exchange the two pure powers) -- which is
# precisely why the exchange is possible at all.
def Laws.kernel_recurrence_exchanged(b, e, n):
  match n:
    case 0n:
      Arith.add_comm(Nat.mul(e, 1n), Nat.mul(b, 1n))
    case 1n++p:
      %Laws.kernel_recurrence_exchanged(b, e, p) : {Nat.add(Nat.mul(e, Nat.add(Nat.mul(b, Laws.Kernel.w(b, e, p)), Laws.Kernel.pow(e, 1n+p))), Nat.mul(b, Laws.Kernel.pow(b, 1n+p))) == Nat.add(Nat.mul(b, _), Nat.mul(e, Laws.Kernel.pow(e, 1n+p))) : Nat}
      %Arith.mul_distrib_l(e, Nat.mul(b, Laws.Kernel.w(b, e, p)), Laws.Kernel.pow(e, 1n+p)) : {Nat.add(_, Nat.mul(b, Laws.Kernel.pow(b, 1n+p))) == Nat.add(Nat.mul(b, Nat.add(Nat.mul(e, Laws.Kernel.w(b, e, p)), Laws.Kernel.pow(b, 1n+p))), Nat.mul(e, Laws.Kernel.pow(e, 1n+p))) : Nat}
      %Arith.mul_distrib_l(b, Nat.mul(e, Laws.Kernel.w(b, e, p)), Laws.Kernel.pow(b, 1n+p)) : {Nat.add(Nat.add(Nat.mul(e, Nat.mul(b, Laws.Kernel.w(b, e, p))), Nat.mul(e, Laws.Kernel.pow(e, 1n+p))), Nat.mul(b, Laws.Kernel.pow(b, 1n+p))) == Nat.add(_, Nat.mul(e, Laws.Kernel.pow(e, 1n+p))) : Nat}
      %Arith.mul_swap_l(e, b, Laws.Kernel.w(b, e, p)) : {Nat.add(Nat.add(Nat.mul(e, Nat.mul(b, Laws.Kernel.w(b, e, p))), Nat.mul(e, Laws.Kernel.pow(e, 1n+p))), Nat.mul(b, Laws.Kernel.pow(b, 1n+p))) == Nat.add(Nat.add(_, Nat.mul(b, Laws.Kernel.pow(b, 1n+p))), Nat.mul(e, Laws.Kernel.pow(e, 1n+p))) : Nat}
      Arith.add_swap_r(Nat.mul(e, Nat.mul(b, Laws.Kernel.w(b, e, p))), Nat.mul(e, Laws.Kernel.pow(e, 1n+p)), Nat.mul(b, Laws.Kernel.pow(b, 1n+p)))

# LAW 2, THE MAIN THEOREM. Induction on n: the hypothesis turns w(e,b,p)
# into w(b,e,p) inside the recurrence, and Law 1 then identifies the two
# one-tick steps, so the sides meet.
def Laws.kernel_symmetry(b, e, n):
  match n:
    case 0n:
      {==}
    case 1n++p:
      %Laws.kernel_symmetry(b, e, p) : {Nat.add(Nat.mul(b, Laws.Kernel.w(b, e, p)), Laws.Kernel.pow(e, 1n+p)) == Nat.add(Nat.mul(e, _), Laws.Kernel.pow(b, 1n+p)) : Nat}
      %Laws.kernel_recurrence_exchanged(b, e, p) : {_ == Nat.add(Nat.mul(e, Laws.Kernel.w(b, e, p)), Laws.Kernel.pow(b, 1n+p)) : Nat}
      {==}

# LAW 3. Induction on n. At b = 1 the recurrence's mul(1, w) reduces to
# add(w, 0), so add_zero_sym strips it and geo's own recurrence remains.
def Laws.kernel_no_forget_is_geometric(e, n):
  match n:
    case 0n:
      {==}
    case 1n++p:
      %Arith.add_zero_sym(Laws.Kernel.w(1n, e, p)) : {Nat.add(_, Laws.Kernel.pow(e, 1n+p)) == Nat.add(Laws.Kernel.geo(e, p), Laws.Kernel.pow(e, 1n+p)) : Nat}
      %Laws.kernel_no_forget_is_geometric(e, p) : {Nat.add(Laws.Kernel.w(1n, e, p), Laws.Kernel.pow(e, 1n+p)) == Nat.add(_, Laws.Kernel.pow(e, 1n+p)) : Nat}
      {==}

# LAW 4. Immediate once mul(1, w) reduces: the increment is exactly
# eta^(n+1), so with alpha = 0 the kernel never decreases with age.
def Laws.kernel_age_monotone_without_forgetting(e, n):
  %Arith.add_zero(Laws.Kernel.w(1n, e, n)) : {Nat.add(Nat.add(Laws.Kernel.w(1n, e, n), 0n), Laws.Kernel.pow(e, 1n+n)) == Nat.add(_, Laws.Kernel.pow(e, 1n+n)) : Nat}
  {==}

# LAW 5. Induction on n. The step rewrites w by the hypothesis, moves the
# (n+1) factor past b with mul_swap_l, and finishes on add_comm -- so the
# closed form is exactly what the one-tick recurrence does on the diagonal.
def Laws.kernel_diagonal(b, n):
  match n:
    case 0n:
      {==}
    case 1n++p:
      %Laws.kernel_diagonal(b, p) : {Nat.add(Nat.mul(b, Laws.Kernel.pow(b, p)), Nat.mul(1n+p, Nat.mul(b, Laws.Kernel.pow(b, p)))) == Nat.add(Nat.mul(b, _), Nat.mul(b, Laws.Kernel.pow(b, p))) : Nat}
      %Arith.mul_swap_l(b, 1n+p, Laws.Kernel.pow(b, p)) : {Nat.add(Nat.mul(b, Laws.Kernel.pow(b, p)), _) == Nat.add(Nat.mul(b, Nat.mul(1n+p, Laws.Kernel.pow(b, p))), Nat.mul(b, Laws.Kernel.pow(b, p))) : Nat}
      Arith.add_comm(Nat.mul(b, Laws.Kernel.pow(b, p)), Nat.mul(b, Nat.mul(1n+p, Laws.Kernel.pow(b, p))))

def main() -> Nat:
  Laws.Kernel.w(1n, 2n, 3n)
