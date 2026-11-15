import ./LAWS.bend as Laws
import ../lib/NAT.bend as Arith
import Base

# PROOF. Everything below Laws.powerset_expansion/Laws.all_inactive_is_exact
# is internal scaffolding (List arithmetic specific to this proof). The
# generic Nat semiring lemmas it uses now live in ../lib/NAT.bend, shared
# with the other proofs instead of copied into each; see that file for the
# right-to-left rewrite convention. Defs with no `law` in scope carry their
# own explicit `->`.

def sum_map_scale_l(+h: Nat, +L: +List<Nat>) -> {Nat.mul(h, Laws.List.sum(&2, L)) == Laws.List.sum_scaled(&2, h, L) : Nat}:
  match L:
    case Nil{}:
      Arith.mul_zero_r(h)
    case e <> +t:
      %Arith.mul_distrib_l(h, e, Laws.List.sum(&2, t)) : {_ == Nat.add(Nat.mul(h, e), Laws.List.sum_scaled(&2, h, t)) : Nat}
      %Equal.sym(Nat, Nat.mul(h, Laws.List.sum(&2, t)), Laws.List.sum_scaled(&2, h, t), sum_map_scale_l(h, t)) : {Nat.add(Nat.mul(h, e), _) == Nat.add(Nat.mul(h, e), Laws.List.sum_scaled(&2, h, t)) : Nat}
      {==}

def sum_scale_list(+h: Nat, +L: +List<Nat>) -> {Nat.mul(h, Laws.List.sum(&2, L)) == Laws.List.sum(&2, Laws.List.scale_list(h, L)) : Nat}:
  match L:
    case Nil{}:
      Arith.mul_zero_r(h)
    case e <> +t:
      %Arith.mul_distrib_l(h, e, Laws.List.sum(&2, t)) : {_ == Nat.add(Nat.mul(h, e), Laws.List.sum(&2, Laws.List.scale_list(h, t))) : Nat}
      %Equal.sym(Nat, Nat.mul(h, Laws.List.sum(&2, t)), Laws.List.sum(&2, Laws.List.scale_list(h, t)), sum_scale_list(h, t)) : {Nat.add(Nat.mul(h, e), _) == Nat.add(Nat.mul(h, e), Laws.List.sum(&2, Laws.List.scale_list(h, t))) : Nat}
      {==}

def sum_append(+A: +List<Nat>, +B: +List<Nat>) -> {Nat.add(Laws.List.sum(&2, A), Laws.List.sum(&2, B)) == Laws.List.sum(&2, List.append(&2, Nat, A, B)) : Nat}:
  match A:
    case Nil{}:
      {==}
    case h <> +t:
      %sum_append(t, B) : {Nat.add(Nat.add(h, Laws.List.sum(&2, t)), Laws.List.sum(&2, B)) == Nat.add(h, _) : Nat}
      Arith.add_assoc(h, Laws.List.sum(&2, t), Laws.List.sum(&2, B))

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
      %Equal.sym(Nat, Nat.add(Laws.List.product_plus_one(List.replicate(Nat, p, 0n)), 0n), Laws.List.product_plus_one(List.replicate(Nat, p, 0n)), Arith.add_zero(Laws.List.product_plus_one(List.replicate(Nat, p, 0n)))) : {_ == 1n : Nat}
      Laws.all_inactive_is_exact(p)

def main() -> +List<Nat>:
  Laws.List.powerset_products([2n, 3n])
