import ./LAWS.bend as Laws
import Base

# PROOF. Everything below Laws.powerset_expansion/Laws.all_inactive_is_exact
# is internal scaffolding (basic Nat/List arithmetic). These carry their
# own explicit return-type annotations rather than a separate `law`
# declaration -- Bend only requires a def matching each declared law,
# not the reverse, but a def with no law in scope needs its own `->`.

def add_assoc(a: Nat, b: Nat, c: Nat) -> {Nat.add(Nat.add(a, b), c) == Nat.add(a, Nat.add(b, c)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_assoc(p, b, c) : {1n+Nat.add(Nat.add(p, b), c) == 1n+_ : Nat}
      {==}

def add_zero(a: Nat) -> {Nat.add(a, 0n) == a : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_zero(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
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

def add_assoc_sym(a: Nat, b: Nat, c: Nat) -> {Nat.add(a, Nat.add(b, c)) == Nat.add(Nat.add(a, b), c) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %add_assoc_sym(p, b, c) : {1n+Nat.add(p, Nat.add(b, c)) == 1n+_ : Nat}
      {==}

def add_middle_swap(+a: Nat, +b: Nat, +X: Nat, +Y: Nat) -> {Nat.add(Nat.add(a, b), Nat.add(X, Y)) == Nat.add(Nat.add(a, X), Nat.add(b, Y)) : Nat}:
  %add_assoc_sym(a, b, Nat.add(X, Y)) : {_ == Nat.add(Nat.add(a, X), Nat.add(b, Y)) : Nat}
  %add_assoc(b, X, Y) : {Nat.add(a, _) == Nat.add(Nat.add(a, X), Nat.add(b, Y)) : Nat}
  %add_comm(X, b) : {Nat.add(a, Nat.add(_, Y)) == Nat.add(Nat.add(a, X), Nat.add(b, Y)) : Nat}
  %add_assoc_sym(X, b, Y) : {Nat.add(a, _) == Nat.add(Nat.add(a, X), Nat.add(b, Y)) : Nat}
  %add_assoc(a, X, Nat.add(b, Y)) : {_ == Nat.add(Nat.add(a, X), Nat.add(b, Y)) : Nat}
  {==}

def mul_distrib_l(h: Nat, +a: Nat, +b: Nat) -> {Nat.add(Nat.mul(h, a), Nat.mul(h, b)) == Nat.mul(h, Nat.add(a, b)) : Nat}:
  match h:
    case 0n:
      {==}
    case 1n++p:
      %mul_distrib_l(p, a, b) : {Nat.add(Nat.add(a, Nat.mul(p, a)), Nat.add(b, Nat.mul(p, b))) == Nat.add(Nat.add(a, b), _) : Nat}
      add_middle_swap(a, Nat.mul(p, a), b, Nat.mul(p, b))

def mul_zero_r(h: Nat) -> {Nat.mul(h, 0n) == 0n : Nat}:
  match h:
    case 0n:
      {==}
    case 1n+p:
      mul_zero_r(p)

# Alternative formulation of the same list-scaling fact, via
# List.sum_scaled instead of List.scale_list -- not used by the main
# theorem below (which uses scale_list), kept as an independent
# cross-check of the same result.
def sum_map_scale_l(+h: Nat, +L: +List<Nat>) -> {Nat.mul(h, Laws.List.sum(&2, L)) == Laws.List.sum_scaled(&2, h, L) : Nat}:
  match L:
    case Nil{}:
      mul_zero_r(h)
    case e <> +t:
      %mul_distrib_l(h, e, Laws.List.sum(&2, t)) : {_ == Nat.add(Nat.mul(h, e), Laws.List.sum_scaled(&2, h, t)) : Nat}
      %Equal.sym(Nat, Nat.mul(h, Laws.List.sum(&2, t)), Laws.List.sum_scaled(&2, h, t), sum_map_scale_l(h, t)) : {Nat.add(Nat.mul(h, e), _) == Nat.add(Nat.mul(h, e), Laws.List.sum_scaled(&2, h, t)) : Nat}
      {==}

def sum_scale_list(+h: Nat, +L: +List<Nat>) -> {Nat.mul(h, Laws.List.sum(&2, L)) == Laws.List.sum(&2, Laws.List.scale_list(h, L)) : Nat}:
  match L:
    case Nil{}:
      mul_zero_r(h)
    case e <> +t:
      %mul_distrib_l(h, e, Laws.List.sum(&2, t)) : {_ == Nat.add(Nat.mul(h, e), Laws.List.sum(&2, Laws.List.scale_list(h, t))) : Nat}
      %Equal.sym(Nat, Nat.mul(h, Laws.List.sum(&2, t)), Laws.List.sum(&2, Laws.List.scale_list(h, t)), sum_scale_list(h, t)) : {Nat.add(Nat.mul(h, e), _) == Nat.add(Nat.mul(h, e), Laws.List.sum(&2, Laws.List.scale_list(h, t))) : Nat}
      {==}

def sum_append(+A: +List<Nat>, +B: +List<Nat>) -> {Nat.add(Laws.List.sum(&2, A), Laws.List.sum(&2, B)) == Laws.List.sum(&2, List.append(&2, Nat, A, B)) : Nat}:
  match A:
    case Nil{}:
      {==}
    case h <> +t:
      %sum_append(t, B) : {Nat.add(Nat.add(h, Laws.List.sum(&2, t)), Laws.List.sum(&2, B)) == Nat.add(h, _) : Nat}
      add_assoc(h, Laws.List.sum(&2, t), Laws.List.sum(&2, B))

def Laws.powerset_expansion(xs):
  match xs:
    case Nil{}:
      {==}
    case h <> +t:
      %Equal.sym(Nat, Laws.List.product_plus_one(t), Laws.List.sum(&2, Laws.List.powerset_products(t)), Laws.powerset_expansion(t)) : {Nat.mul(1n+h, _) == Laws.List.sum(&2, List.append(&2, Nat, Laws.List.powerset_products(t), Laws.List.scale_list(h, Laws.List.powerset_products(t)))) : Nat}
      %sum_append(Laws.List.powerset_products(t), Laws.List.scale_list(h, Laws.List.powerset_products(t))) : {Nat.add(Laws.List.sum(&2, Laws.List.powerset_products(t)), Nat.mul(h, Laws.List.sum(&2, Laws.List.powerset_products(t)))) == _ : Nat}
      %sum_scale_list(h, Laws.List.powerset_products(t)) : {Nat.add(Laws.List.sum(&2, Laws.List.powerset_products(t)), Nat.mul(h, Laws.List.sum(&2, Laws.List.powerset_products(t)))) == Nat.add(Laws.List.sum(&2, Laws.List.powerset_products(t)), _) : Nat}
      {==}

def Laws.all_inactive_is_exact(n):
  match n:
    case 0n:
      {==}
    case 1n++p:
      %Equal.sym(Nat, Nat.add(Laws.List.product_plus_one(List.replicate(Nat, p, 0n)), 0n), Laws.List.product_plus_one(List.replicate(Nat, p, 0n)), add_zero(Laws.List.product_plus_one(List.replicate(Nat, p, 0n)))) : {_ == 1n : Nat}
      Laws.all_inactive_is_exact(p)

def main() -> +List<Nat>:
  Laws.List.powerset_products([2n, 3n])
