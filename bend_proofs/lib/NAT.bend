import Base

# Shared Nat semiring lemmas for the proofs in ../. Extracted because the
# same add_* block had been copy-pasted between proofs, which is how two
# copies drift apart; anything proved here is proved once.
#
# ORIENTATION MATTERS, and is why several lemmas appear twice. A rewrite
# `%lemma : {goal with _}` replaces an occurrence of the lemma's RIGHT-hand
# side with its LEFT-hand side, so rewriting runs right-to-left. To move a
# goal's subterm X into Y you need the lemma whose RHS is X, which is why
# add_assoc/add_assoc_sym, add_zero/add_zero_sym and
# mul_distrib_r/mul_distrib_r_sym all exist: same fact, opposite usable
# direction. Picking the wrong one is the single most common error here,
# and the checker reports it as "expected <goal> / observed <annotation>".
#
# Nat.add recurses on its FIRST argument and Nat.mul on its first too
# (Nat.mul(1n+p, b) = Nat.add(b, Nat.mul(p, b))), so `match` on the first
# argument is what makes these reduce. A variable used twice in a body
# needs `+` on its binder, and a matched successor reused twice needs the
# `1n++p` reusable pattern.

def add_assoc(a: Nat, b: Nat, c: Nat) -> {Nat.add(Nat.add(a, b), c) == Nat.add(a, Nat.add(b, c)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_assoc(p, b, c) : {1n+Nat.add(Nat.add(p, b), c) == 1n+_ : Nat}
      {==}

def add_assoc_sym(a: Nat, b: Nat, c: Nat) -> {Nat.add(a, Nat.add(b, c)) == Nat.add(Nat.add(a, b), c) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_assoc_sym(p, b, c) : {1n+Nat.add(p, Nat.add(b, c)) == 1n+_ : Nat}
      {==}

def add_zero(a: Nat) -> {Nat.add(a, 0n) == a : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_zero(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}

def add_zero_sym(a: Nat) -> {a == Nat.add(a, 0n) : Nat}:
  %add_zero(a) : {_ == Nat.add(a, 0n) : Nat}
  {==}

def add_succ_r(a: Nat, b: Nat) -> {Nat.add(a, 1n+b) == 1n+Nat.add(a, b) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_succ_r(p, b) : {1n+Nat.add(p, 1n+b) == 1n+_ : Nat}
      {==}

def add_succ_r_sym(a: Nat, b: Nat) -> {1n+Nat.add(a, b) == Nat.add(a, 1n+b) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_succ_r_sym(p, b) : {1n+1n+Nat.add(p, b) == 1n+_ : Nat}
      {==}

def add_comm(a: Nat, +b: Nat) -> {Nat.add(a, b) == Nat.add(b, a) : Nat}:
  match a:
    case 0n:
      %add_zero(b) : {_ == Nat.add(b, 0n) : Nat}
      {==}
    case 1n++p:
      %add_succ_r_sym(b, p) : {1n+Nat.add(p, b) == _ : Nat}
      %add_comm(p, b) : {1n+Nat.add(p, b) == 1n+_ : Nat}
      {==}

def add_middle_swap(+a: Nat, +b: Nat, +X: Nat, +Y: Nat) -> {Nat.add(Nat.add(a, b), Nat.add(X, Y)) == Nat.add(Nat.add(a, X), Nat.add(b, Y)) : Nat}:
  %add_assoc_sym(a, b, Nat.add(X, Y)) : {_ == Nat.add(Nat.add(a, X), Nat.add(b, Y)) : Nat}
  %add_assoc(b, X, Y) : {Nat.add(a, _) == Nat.add(Nat.add(a, X), Nat.add(b, Y)) : Nat}
  %add_comm(X, b) : {Nat.add(a, Nat.add(_, Y)) == Nat.add(Nat.add(a, X), Nat.add(b, Y)) : Nat}
  %add_assoc_sym(X, b, Y) : {Nat.add(a, _) == Nat.add(Nat.add(a, X), Nat.add(b, Y)) : Nat}
  %add_assoc(a, X, Nat.add(b, Y)) : {_ == Nat.add(Nat.add(a, X), Nat.add(b, Y)) : Nat}
  {==}

# Pull the second addend out front. add_middle_swap with a leading 0.
def add_swap_l(+a: Nat, +b: Nat, +c: Nat) -> {Nat.add(a, Nat.add(b, c)) == Nat.add(b, Nat.add(a, c)) : Nat}:
  add_middle_swap(0n, a, b, c)

# Exchange the last two addends. add_middle_swap with a trailing 0.
def add_swap_r(+x: Nat, +p: Nat, +q: Nat) -> {Nat.add(Nat.add(x, p), q) == Nat.add(Nat.add(x, q), p) : Nat}:
  %add_zero(q) : {Nat.add(Nat.add(x, p), _) == Nat.add(Nat.add(x, q), p) : Nat}
  %add_zero(p) : {Nat.add(Nat.add(x, p), Nat.add(q, 0n)) == Nat.add(Nat.add(x, q), _) : Nat}
  add_middle_swap(x, p, q, 0n)

def mul_zero_r(h: Nat) -> {Nat.mul(h, 0n) == 0n : Nat}:
  match h:
    case 0n:
      {==}
    case 1n+p:
      %mul_zero_r(p) : {Nat.mul(p, 0n) == _ : Nat}
      {==}

def mul_one_r(h: Nat) -> {Nat.mul(h, 1n) == h : Nat}:
  match h:
    case 0n:
      {==}
    case 1n+p:
      %mul_one_r(p) : {1n+Nat.mul(p, 1n) == 1n+_ : Nat}
      {==}

# add(a, mul(a, b)) == mul(a, 1n+b), oriented so mul(a, 1n+b) is rewritable.
def mul_succ_r_sym(a: Nat, +b: Nat) -> {Nat.add(a, Nat.mul(a, b)) == Nat.mul(a, 1n+b) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n++p:
      %mul_succ_r_sym(p, b) : {1n+Nat.add(p, Nat.add(b, Nat.mul(p, b))) == 1n+Nat.add(b, _) : Nat}
      %add_swap_l(p, b, Nat.mul(p, b)) : {1n+Nat.add(p, Nat.add(b, Nat.mul(p, b))) == 1n+_ : Nat}
      {==}

def mul_comm(a: Nat, +b: Nat) -> {Nat.mul(a, b) == Nat.mul(b, a) : Nat}:
  match a:
    case 0n:
      %mul_zero_r(b) : {_ == Nat.mul(b, 0n) : Nat}
      {==}
    case 1n++p:
      %mul_succ_r_sym(b, p) : {Nat.add(b, Nat.mul(p, b)) == _ : Nat}
      %mul_comm(p, b) : {Nat.add(b, Nat.mul(p, b)) == Nat.add(b, _) : Nat}
      {==}

def mul_distrib_l(h: Nat, +a: Nat, +b: Nat) -> {Nat.add(Nat.mul(h, a), Nat.mul(h, b)) == Nat.mul(h, Nat.add(a, b)) : Nat}:
  match h:
    case 0n:
      {==}
    case 1n++p:
      %mul_distrib_l(p, a, b) : {Nat.add(Nat.add(a, Nat.mul(p, a)), Nat.add(b, Nat.mul(p, b))) == Nat.add(Nat.add(a, b), _) : Nat}
      add_middle_swap(a, Nat.mul(p, a), b, Nat.mul(p, b))

def mul_distrib_r(x: Nat, +y: Nat, +c: Nat) -> {Nat.mul(Nat.add(x, y), c) == Nat.add(Nat.mul(x, c), Nat.mul(y, c)) : Nat}:
  match x:
    case 0n:
      {==}
    case 1n++p:
      %add_assoc_sym(c, Nat.mul(p, c), Nat.mul(y, c)) : {Nat.add(c, Nat.mul(Nat.add(p, y), c)) == _ : Nat}
      %mul_distrib_r(p, y, c) : {Nat.add(c, Nat.mul(Nat.add(p, y), c)) == Nat.add(c, _) : Nat}
      {==}

def mul_distrib_r_sym(+x: Nat, +y: Nat, +c: Nat) -> {Nat.add(Nat.mul(x, c), Nat.mul(y, c)) == Nat.mul(Nat.add(x, y), c) : Nat}:
  %mul_distrib_r(x, y, c) : {_ == Nat.mul(Nat.add(x, y), c) : Nat}
  {==}

def mul_assoc(a: Nat, +b: Nat, +c: Nat) -> {Nat.mul(Nat.mul(a, b), c) == Nat.mul(a, Nat.mul(b, c)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n++p:
      %mul_distrib_r_sym(b, Nat.mul(p, b), c) : {_ == Nat.add(Nat.mul(b, c), Nat.mul(p, Nat.mul(b, c))) : Nat}
      %mul_assoc(p, b, c) : {Nat.add(Nat.mul(b, c), Nat.mul(Nat.mul(p, b), c)) == Nat.add(Nat.mul(b, c), _) : Nat}
      {==}

# Move one factor past another: mul(b, mul(e, w)) == mul(e, mul(b, w)).
def mul_swap_l(+b: Nat, +e: Nat, +w: Nat) -> {Nat.mul(b, Nat.mul(e, w)) == Nat.mul(e, Nat.mul(b, w)) : Nat}:
  %mul_assoc(b, e, w) : {_ == Nat.mul(e, Nat.mul(b, w)) : Nat}
  %mul_comm(e, b) : {Nat.mul(_, w) == Nat.mul(e, Nat.mul(b, w)) : Nat}
  %mul_assoc(e, b, w) : {Nat.mul(Nat.mul(e, b), w) == _ : Nat}
  {==}

def main() -> Nat:
  Nat.add(Nat.mul(2n, 3n), 1n)
