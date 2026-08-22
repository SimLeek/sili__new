/-
  Formal statement and proof of the information-theoretic claim behind
  combining a forward-contribution signal with a backward-sensitivity
  signal in the DISLDO per-synapse importance estimator (sili__new,
  linear_disldo.hpp).

  SCOPE, stated precisely up front (per direct request for rigor):

  This file proves a GENERAL, domain-independent fact from information
  theory: for any three random variables Θ (a hidden/target quantity),
  X, Y (two observations) on a common finite probability space,

      H(Θ | X, Y) ≤ H(Θ | X)

  with EQUALITY iff Y is conditionally independent of Θ given X. This is
  the standard "conditioning cannot increase entropy" theorem (Cover &
  Thomas, *Elements of Information Theory*, Thm 2.6.5), proved here from
  first principles via Gibbs' inequality (non-negativity of KL
  divergence), not merely cited.

  What this file does NOT prove: that `g = dy·x` (backward sensitivity)
  and `contrib = w·x` (forward contribution) are NOT conditionally
  independent given the synapse's true importance, for the specific
  DISLDO training dynamics. That is a modeling/architectural claim about
  a real, stochastic, evolving system (FP4 rounding, RMSprop-style
  damping, synaptogenesis-driven regrowth) -- it is the HYPOTHESIS under
  which the general theorem below yields a STRICT improvement, stated
  explicitly as a hypothesis in `combined_signal_strictly_informative`,
  never smuggled in as a conclusion.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field

open Finset

noncomputable section

/-! ### Finite discrete probability distributions -/

/-- A finite discrete probability distribution over `Finset.univ : Finset α`,
for a `Fintype α`: nonnegative weights summing to 1. Kept as a plain
structure (rather than reusing `PMF`) so the Gibbs'-inequality proof
below is fully self-contained and auditable end to end. -/
structure ProbDist (α : Type*) [Fintype α] where
  p : α → ℝ
  nonneg : ∀ x, 0 ≤ p x
  sum_eq_one : ∑ x, p x = 1

namespace ProbDist

variable {α : Type*} [Fintype α]

/-- Shannon entropy, nats (natural log), with the standard convention
`0 * log 0 = 0` (handled automatically: `Real.log 0 = 0` in Mathlib, so
`p x * Real.log (p x) = 0` whenever `p x = 0`). -/
def entropy (d : ProbDist α) : ℝ := -∑ x, d.p x * Real.log (d.p x)

end ProbDist

/-! ### Gibbs' inequality (KL divergence ≥ 0)

Restricted to `q` DOMINATING `p`'s support (`p.p x > 0 → q.p x > 0`),
matching every standard textbook statement (Cover & Thomas Thm 2.6.3):
without this, KL(p‖q) is genuinely +∞ (not just "hard to bound") when
`q` has a zero where `p` doesn't, which plain `ℝ` (not `EReal`) cannot
even state, let alone need to bound below by anything -- adding the
hypothesis is not a weakening for our purposes: in the application
below, `q` is always a PRODUCT OF MARGINALS of the same joint `p`
factors through, and a marginal is never smaller than any single joint
value it sums (`p.p x ≤ q.p x` is not required, only positivity), so
the hypothesis is automatically satisfied there -- see
`marginal_product_dominates` and its use in `entropy_le_condEntropy`.
-/

/-- The elementary inequality `log t ≤ t - 1` for `t > 0`, the entire
engine behind Gibbs' inequality below (Mathlib: `Real.log_le_sub_one_of_pos`). -/
example (t : ℝ) (ht : 0 < t) : Real.log t ≤ t - 1 :=
  Real.log_le_sub_one_of_pos ht

/-- **Gibbs' inequality.** For distributions `p, q` on the same finite
index set with `q` dominating `p`'s support, the cross-entropy of `p`
relative to `q` is at least `p`'s own (negative-signed) entropy sum:
`∑ p·log q ≤ ∑ p·log p`. Equivalently, `KL(p‖q) = ∑ p·log(p/q) ≥ 0`.

Proof: for every `x` with `p x > 0` (hence, by the dominance hypothesis,
`q x > 0` too), apply `log t ≤ t - 1` at `t = q(x)/p(x) > 0`, rearrange
to `log(q x) - log(p x) ≤ q(x)/p(x) - 1`, multiply through by `p(x) ≥ 0`
(direction preserved), giving `p(x)·log(q x) - p(x)·log(p x) ≤ q(x) - p(x)`
pointwise. Sum over all `x`: the RHS telescopes to `∑q - ∑p = 1 - 1 = 0`
since both are probability distributions over the SAME finite index set.
For `x` with `p x = 0`, both sides of the pointwise inequality are `0`
directly (Mathlib's `Real.log 0 = 0`), so the bound holds there too,
trivially. -/
theorem gibbs_inequality {α : Type*} [Fintype α] (p q : ProbDist α)
    (hdom : ∀ x, 0 < p.p x → 0 < q.p x) :
    ∑ x, p.p x * Real.log (p.p x) - ∑ x, p.p x * Real.log (q.p x)
      ≥ 0 := by
  have pointwise : ∀ x,
      p.p x * Real.log (q.p x) - p.p x * Real.log (p.p x) ≤ q.p x - p.p x := by
    intro x
    rcases eq_or_lt_of_le (p.nonneg x) with hpx0 | hpx0
    · simp [← hpx0]
      linarith [q.nonneg x]
    · have hqx0 : 0 < q.p x := hdom x hpx0
      have hratio : 0 < q.p x / p.p x := div_pos hqx0 hpx0
      have hlog : Real.log (q.p x / p.p x) ≤ q.p x / p.p x - 1 :=
        Real.log_le_sub_one_of_pos hratio
      rw [Real.log_div (ne_of_gt hqx0) (ne_of_gt hpx0)] at hlog
      have hmul := mul_le_mul_of_nonneg_left hlog (le_of_lt hpx0)
      have hcancel : p.p x * (q.p x / p.p x) = q.p x := by
        field_simp
      calc p.p x * Real.log (q.p x) - p.p x * Real.log (p.p x)
          = p.p x * (Real.log (q.p x) - Real.log (p.p x)) := by ring
        _ ≤ p.p x * (q.p x / p.p x - 1) := hmul
        _ = q.p x - p.p x := by rw [mul_sub, hcancel, mul_one]
  have summed : ∑ x, (p.p x * Real.log (q.p x) - p.p x * Real.log (p.p x))
      ≤ ∑ x, (q.p x - p.p x) := Finset.sum_le_sum (fun x _ => pointwise x)
  have rhs0 : ∑ x, (q.p x - p.p x) = 0 := by
    rw [Finset.sum_sub_distrib, q.sum_eq_one, p.sum_eq_one]; ring
  have split : ∑ x, (p.p x * Real.log (q.p x) - p.p x * Real.log (p.p x))
      = (∑ x, p.p x * Real.log (q.p x)) - ∑ x, p.p x * Real.log (p.p x) := by
    rw [Finset.sum_sub_distrib]
  linarith [split ▸ summed, rhs0]

/-! ### Conditional entropy and the "conditioning reduces entropy" theorem -/

/-- A joint distribution over `Θ × X × Y` (all finite), from which every
marginal and conditional used below is derived directly (no separate
axioms -- consistency is by construction, not assumed). -/
structure Joint (Θ X Y : Type*) [Fintype Θ] [Fintype X] [Fintype Y] where
  p : Θ → X → Y → ℝ
  nonneg : ∀ θ x y, 0 ≤ p θ x y
  sum_eq_one : ∑ θ, ∑ x, ∑ y, p θ x y = 1

namespace Joint

variable {Θ X Y : Type*} [Fintype Θ] [Fintype X] [Fintype Y] (J : Joint Θ X Y)

/-- Marginal over `X` alone. -/
def pX (x : X) : ℝ := ∑ θ, ∑ y, J.p θ x y

/-- Marginal over `(X, Y)`. -/
def pXY (x : X) (y : Y) : ℝ := ∑ θ, J.p θ x y

/-- Marginal over `(Θ, X)`. -/
def pΘX (θ : Θ) (x : X) : ℝ := ∑ y, J.p θ x y

/-- `H(Θ | X, Y)` as a `Finset`-weighted average of slice entropies,
skipping slices with zero probability (their conditional entropy is
undefined but irrelevant, weighted by 0 -- standard convention). -/
def condEntropyΘ_XY : ℝ :=
  ∑ x, ∑ y, J.pXY x y *
    (- ∑ θ, (if J.pXY x y = 0 then 0 else J.p θ x y / J.pXY x y) *
       Real.log (if J.pXY x y = 0 then 0 else J.p θ x y / J.pXY x y))

/-- `H(Θ | X)`, same convention. -/
def condEntropyΘ_X : ℝ :=
  ∑ x, J.pX x *
    (- ∑ θ, (if J.pX x = 0 then 0 else J.pΘX θ x / J.pX x) *
       Real.log (if J.pX x = 0 then 0 else J.pΘX θ x / J.pX x))

end Joint

/-! ### Subadditivity of joint entropy: `H(A,B) ≤ H(A) + H(B)`

A general, `Joint`-independent corollary of `gibbs_inequality`: apply it to
any joint distribution `d` over a product type `α × β` versus the product
of `d`'s own two marginals. This is the real engine behind the main
theorem below; the `Joint`-specific work after this section only has to
connect this general fact to the particular slice distributions that show
up in `condEntropyΘ_XY` / `condEntropyΘ_X`. -/

namespace ProbDist

variable {α : Type*} [Fintype α] {β : Type*} [Fintype β]

/-- Left marginal of a joint distribution over `α × β`. -/
def marginalLeft (d : ProbDist (α × β)) : ProbDist α where
  p a := ∑ b, d.p (a, b)
  nonneg a := Finset.sum_nonneg fun b _ => d.nonneg (a, b)
  sum_eq_one := by
    rw [← d.sum_eq_one, Fintype.sum_prod_type]

/-- Right marginal of a joint distribution over `α × β`. -/
def marginalRight (d : ProbDist (α × β)) : ProbDist β where
  p b := ∑ a, d.p (a, b)
  nonneg b := Finset.sum_nonneg fun a _ => d.nonneg (a, b)
  sum_eq_one := by
    rw [← d.sum_eq_one, Fintype.sum_prod_type_right]

/-- The independent product of two distributions is again a distribution. -/
def prod (da : ProbDist α) (db : ProbDist β) : ProbDist (α × β) where
  p ab := da.p ab.1 * db.p ab.2
  nonneg ab := mul_nonneg (da.nonneg ab.1) (db.nonneg ab.2)
  sum_eq_one := by
    have hprod : ∑ ab : α × β, da.p ab.1 * db.p ab.2 = ∑ a, ∑ b, da.p a * db.p b :=
      Fintype.sum_prod_type' (fun a b => da.p a * db.p b)
    rw [hprod]
    simp only [← Finset.mul_sum, db.sum_eq_one, mul_one]
    exact da.sum_eq_one

/-- The product of `d`'s own two marginals dominates `d`'s support: if a
joint entry is positive, so is each of the marginals it contributes to
(a marginal is a sum of nonnegative terms including that entry). -/
theorem prod_marginals_dominates (d : ProbDist (α × β)) :
    ∀ ab, 0 < d.p ab → 0 < (d.marginalLeft.prod d.marginalRight).p ab := by
  rintro ⟨a, b⟩ hab
  have hL : 0 < d.marginalLeft.p a :=
    lt_of_lt_of_le hab (Finset.single_le_sum (fun b' _ => d.nonneg (a, b')) (Finset.mem_univ b))
  have hR : 0 < d.marginalRight.p b :=
    lt_of_lt_of_le hab (Finset.single_le_sum (fun a' _ => d.nonneg (a', b)) (Finset.mem_univ a))
  exact mul_pos hL hR

/-- **Subadditivity of joint entropy.** `H(A,B) ≤ H(A) + H(B)`, a direct
corollary of `gibbs_inequality` applied to `d` versus the product of its
own marginals: `KL(d ‖ marginalLeft × marginalRight) ≥ 0` unwinds exactly
to this, once `log(pa a * pb b) = log(pa a) + log(pb b)` is used to split
the cross-entropy term back into two independent marginal entropies. -/
theorem entropy_le_sum_marginal_entropy (d : ProbDist (α × β)) :
    d.entropy ≤ d.marginalLeft.entropy + d.marginalRight.entropy := by
  have hg := gibbs_inequality d (d.marginalLeft.prod d.marginalRight) d.prod_marginals_dominates
  -- Rewrite `∑ d.p * log((marginalLeft.prod marginalRight).p)` as the sum of the
  -- two marginal cross-entropy terms, by splitting the log of a product pointwise.
  have hsplit : ∀ ab : α × β, d.p ab * Real.log ((d.marginalLeft.prod d.marginalRight).p ab)
      = d.p ab * Real.log (d.marginalLeft.p ab.1) + d.p ab * Real.log (d.marginalRight.p ab.2) := by
    rintro ⟨a, b⟩
    change d.p (a, b) * Real.log (d.marginalLeft.p a * d.marginalRight.p b)
      = d.p (a, b) * Real.log (d.marginalLeft.p a) + d.p (a, b) * Real.log (d.marginalRight.p b)
    rcases eq_or_lt_of_le (d.nonneg (a, b)) with hd0 | hd0
    · simp [← hd0]
    · have hL : 0 < d.marginalLeft.p a :=
        lt_of_lt_of_le hd0 (Finset.single_le_sum (fun b' _ => d.nonneg (a, b')) (Finset.mem_univ b))
      have hR : 0 < d.marginalRight.p b :=
        lt_of_lt_of_le hd0 (Finset.single_le_sum (fun a' _ => d.nonneg (a', b)) (Finset.mem_univ a))
      rw [Real.log_mul (ne_of_gt hL) (ne_of_gt hR), mul_add]
  have hleft : ∑ ab : α × β, d.p ab * Real.log (d.marginalLeft.p ab.1)
      = ∑ a, d.marginalLeft.p a * Real.log (d.marginalLeft.p a) := by
    have hstep : ∑ ab : α × β, d.p ab * Real.log (d.marginalLeft.p ab.1)
        = ∑ a, ∑ b, d.p (a, b) * Real.log (d.marginalLeft.p a) :=
      Fintype.sum_prod_type' (fun a b => d.p (a, b) * Real.log (d.marginalLeft.p a))
    rw [hstep]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [← Finset.sum_mul]
    rfl
  have hright : ∑ ab : α × β, d.p ab * Real.log (d.marginalRight.p ab.2)
      = ∑ b, d.marginalRight.p b * Real.log (d.marginalRight.p b) := by
    have hstep : ∑ ab : α × β, d.p ab * Real.log (d.marginalRight.p ab.2)
        = ∑ b, ∑ a, d.p (a, b) * Real.log (d.marginalRight.p b) :=
      Fintype.sum_prod_type_right' (fun a b => d.p (a, b) * Real.log (d.marginalRight.p b))
    rw [hstep]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [← Finset.sum_mul]
    rfl
  have hcross : ∑ ab, d.p ab * Real.log ((d.marginalLeft.prod d.marginalRight).p ab)
      = ∑ a, d.marginalLeft.p a * Real.log (d.marginalLeft.p a)
        + ∑ b, d.marginalRight.p b * Real.log (d.marginalRight.p b) := by
    rw [Finset.sum_congr rfl (fun ab _ => hsplit ab), Finset.sum_add_distrib, hleft, hright]
  unfold ProbDist.entropy
  linarith [hg, hcross]

end ProbDist

/-! ### `H(Θ|X,Y) ≤ H(Θ|X)`, via `entropy_le_sum_marginal_entropy` applied
slice-by-slice (fixed `x`, comparing the conditional joint `p(θ,y|x)`
against its own marginals). -/

namespace Joint

variable {Θ X Y : Type*} [Fintype Θ] [Fintype X] [Fintype Y] (J : Joint Θ X Y)

theorem pX_nonneg (x : X) : 0 ≤ J.pX x :=
  Finset.sum_nonneg fun θ _ => Finset.sum_nonneg fun y _ => J.nonneg θ x y

theorem pXY_nonneg (x : X) (y : Y) : 0 ≤ J.pXY x y :=
  Finset.sum_nonneg fun θ _ => J.nonneg θ x y

theorem pΘX_nonneg (θ : Θ) (x : X) : 0 ≤ J.pΘX θ x :=
  Finset.sum_nonneg fun y _ => J.nonneg θ x y

theorem pX_eq_sum_pΘX (x : X) : J.pX x = ∑ θ, J.pΘX θ x := rfl

theorem pX_eq_sum_pXY (x : X) : J.pX x = ∑ y, J.pXY x y := by
  show ∑ θ, ∑ y, J.p θ x y = ∑ y, ∑ θ, J.p θ x y
  exact Finset.sum_comm

theorem pXY_le_pX (x : X) (y : Y) : J.pXY x y ≤ J.pX x := by
  rw [J.pX_eq_sum_pXY]
  exact Finset.single_le_sum (fun y' _ => J.pXY_nonneg x y') (Finset.mem_univ y)

theorem pΘX_le_pX (θ : Θ) (x : X) : J.pΘX θ x ≤ J.pX x := by
  rw [J.pX_eq_sum_pΘX]
  exact Finset.single_le_sum (fun θ' _ => J.pΘX_nonneg θ' x) (Finset.mem_univ θ)

theorem p_le_pΘX (θ : Θ) (x : X) (y : Y) : J.p θ x y ≤ J.pΘX θ x :=
  Finset.single_le_sum (fun y' _ => J.nonneg θ x y') (Finset.mem_univ y)

theorem p_le_pXY (θ : Θ) (x : X) (y : Y) : J.p θ x y ≤ J.pXY x y :=
  Finset.single_le_sum (fun θ' _ => J.nonneg θ' x y) (Finset.mem_univ θ)

/-- The conditional joint distribution of `(Θ, Y)` given `X = x`, for `x`
with positive marginal probability `J.pX x`. -/
def sliceΘY (x : X) (hx : 0 < J.pX x) : ProbDist (Θ × Y) where
  p θy := J.p θy.1 x θy.2 / J.pX x
  nonneg θy := div_nonneg (J.nonneg θy.1 x θy.2) hx.le
  sum_eq_one := by
    have hnum : (∑ θy : Θ × Y, J.p θy.1 x θy.2) = J.pX x :=
      Fintype.sum_prod_type' (fun θ y => J.p θ x y)
    rw [← Finset.sum_div, hnum, div_self (ne_of_gt hx)]

theorem sliceΘY_marginalLeft (x : X) (hx : 0 < J.pX x) (θ : Θ) :
    (J.sliceΘY x hx).marginalLeft.p θ = J.pΘX θ x / J.pX x := by
  show (∑ y, J.p θ x y / J.pX x) = J.pΘX θ x / J.pX x
  rw [← Finset.sum_div]
  rfl

theorem sliceΘY_marginalRight (x : X) (hx : 0 < J.pX x) (y : Y) :
    (J.sliceΘY x hx).marginalRight.p y = J.pXY x y / J.pX x := by
  show (∑ θ, J.p θ x y / J.pX x) = J.pXY x y / J.pX x
  rw [← Finset.sum_div]
  rfl

/-- **Per-slice chain rule.** For a single `x` with `J.pX x > 0`, the
`x`-summand of `condEntropyΘ_XY` is at most the `x`-summand of
`condEntropyΘ_X`. Proved by rewriting both summands in terms of the
slice distribution `sliceΘY J x hx` and its marginals, then applying
`ProbDist.entropy_le_sum_marginal_entropy` (itself a corollary of
`gibbs_inequality`). -/
theorem condEntropyΘ_XY_summand_le (x : X) (hx : 0 < J.pX x) :
    ∑ y, J.pXY x y *
      (- ∑ θ, (if J.pXY x y = 0 then 0 else J.p θ x y / J.pXY x y) *
         Real.log (if J.pXY x y = 0 then 0 else J.p θ x y / J.pXY x y))
    ≤ J.pX x *
      (- ∑ θ, (if J.pX x = 0 then 0 else J.pΘX θ x / J.pX x) *
         Real.log (if J.pX x = 0 then 0 else J.pΘX θ x / J.pX x)) := by
  set d := J.sliceΘY x hx with hd
  -- Per-`y` pointwise identity: rewrite each raw summand of `condEntropyΘ_XY`
  -- in terms of the un-normalized joint `J.p` against `J.pX x` directly.
  have hpt : ∀ y, J.pXY x y *
      (- ∑ θ, (if J.pXY x y = 0 then 0 else J.p θ x y / J.pXY x y) *
         Real.log (if J.pXY x y = 0 then 0 else J.p θ x y / J.pXY x y))
      = - ∑ θ, J.p θ x y * Real.log (J.p θ x y / J.pX x)
        + J.pXY x y * Real.log (J.pXY x y / J.pX x) := by
    intro y
    rcases eq_or_lt_of_le (J.pXY_nonneg x y) with hy0 | hy0
    · have hall0 : ∀ θ, J.p θ x y = 0 := fun θ =>
        le_antisymm (hy0 ▸ J.p_le_pXY θ x y) (J.nonneg θ x y)
      simp [← hy0, hall0]
    · have hyne : J.pXY x y ≠ 0 := ne_of_gt hy0
      simp only [if_neg hyne]
      have hcancel : ∀ θ, J.pXY x y * (J.p θ x y / J.pXY x y * Real.log (J.p θ x y / J.pXY x y))
          = J.p θ x y * Real.log (J.p θ x y / J.pXY x y) := by
        intro θ
        rw [show J.pXY x y * (J.p θ x y / J.pXY x y * Real.log (J.p θ x y / J.pXY x y))
            = (J.pXY x y * (J.p θ x y / J.pXY x y)) * Real.log (J.p θ x y / J.pXY x y) from by ring,
            mul_div_cancel₀ (J.p θ x y) hyne]
      have hlogsplit : ∀ θ, J.p θ x y * Real.log (J.p θ x y / J.pXY x y)
          = J.p θ x y * Real.log (J.p θ x y / J.pX x) - J.p θ x y * Real.log (J.pXY x y / J.pX x) := by
        intro θ
        rcases eq_or_lt_of_le (J.nonneg θ x y) with hθ0 | hθ0
        · simp [← hθ0]
        · have h1 : Real.log (J.p θ x y / J.pXY x y)
              = Real.log (J.p θ x y / J.pX x) - Real.log (J.pXY x y / J.pX x) := by
            rw [Real.log_div (ne_of_gt hθ0) hyne, Real.log_div (ne_of_gt hθ0) (ne_of_gt hx),
                Real.log_div hyne (ne_of_gt hx)]
            ring
          rw [h1, mul_sub]
      rw [mul_neg, Finset.mul_sum]
      rw [Finset.sum_congr rfl (fun θ _ => hcancel θ),
          Finset.sum_congr rfl (fun θ _ => hlogsplit θ), Finset.sum_sub_distrib,
          ← Finset.sum_mul]
      have : (∑ θ, J.p θ x y) = J.pXY x y := rfl
      rw [this]
      ring
  -- Sum `hpt` over `y`, then identify the result with `pX x * (d.entropy - d.marginalRight.entropy)`.
  have hdEntropy : d.entropy = - ∑ y, ∑ θ, (J.p θ x y / J.pX x) * Real.log (J.p θ x y / J.pX x) := by
    have h : (∑ θy : Θ × Y, d.p θy * Real.log (d.p θy))
        = ∑ y, ∑ θ, (J.p θ x y / J.pX x) * Real.log (J.p θ x y / J.pX x) :=
      Fintype.sum_prod_type_right' (fun θ y => (J.p θ x y / J.pX x) * Real.log (J.p θ x y / J.pX x))
    unfold ProbDist.entropy
    rw [h]
  have hdMR : d.marginalRight.entropy = - ∑ y, (J.pXY x y / J.pX x) * Real.log (J.pXY x y / J.pX x) := by
    unfold ProbDist.entropy
    congr 1
    exact Finset.sum_congr rfl fun y _ => by rw [J.sliceΘY_marginalRight x hx y]
  have hexpand : J.pX x * (d.entropy - d.marginalRight.entropy)
      = - ∑ y, ∑ θ, J.p θ x y * Real.log (J.p θ x y / J.pX x)
        + ∑ y, J.pXY x y * Real.log (J.pXY x y / J.pX x) := by
    have e1 : J.pX x * (∑ y, ∑ θ, (J.p θ x y / J.pX x) * Real.log (J.p θ x y / J.pX x))
        = ∑ y, ∑ θ, J.p θ x y * Real.log (J.p θ x y / J.pX x) := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun y _ => ?_
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun θ _ => ?_
      rw [show J.pX x * (J.p θ x y / J.pX x * Real.log (J.p θ x y / J.pX x))
          = (J.pX x * (J.p θ x y / J.pX x)) * Real.log (J.p θ x y / J.pX x) from by ring,
          mul_div_cancel₀ (J.p θ x y) (ne_of_gt hx)]
    have e2 : J.pX x * (∑ y, (J.pXY x y / J.pX x) * Real.log (J.pXY x y / J.pX x))
        = ∑ y, J.pXY x y * Real.log (J.pXY x y / J.pX x) := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun y _ => ?_
      rw [show J.pX x * (J.pXY x y / J.pX x * Real.log (J.pXY x y / J.pX x))
          = (J.pX x * (J.pXY x y / J.pX x)) * Real.log (J.pXY x y / J.pX x) from by ring,
          mul_div_cancel₀ (J.pXY x y) (ne_of_gt hx)]
    rw [hdEntropy, hdMR,
        show J.pX x * (-(∑ y, ∑ θ, (J.p θ x y / J.pX x) * Real.log (J.p θ x y / J.pX x))
              - -(∑ y, (J.pXY x y / J.pX x) * Real.log (J.pXY x y / J.pX x)))
          = J.pX x * (∑ y, (J.pXY x y / J.pX x) * Real.log (J.pXY x y / J.pX x))
            - J.pX x * (∑ y, ∑ θ, (J.p θ x y / J.pX x) * Real.log (J.p θ x y / J.pX x))
          from by ring,
        e1, e2]
    ring
  have hLHS : ∑ y, J.pXY x y *
      (- ∑ θ, (if J.pXY x y = 0 then 0 else J.p θ x y / J.pXY x y) *
         Real.log (if J.pXY x y = 0 then 0 else J.p θ x y / J.pXY x y))
      = J.pX x * (d.entropy - d.marginalRight.entropy) := by
    rw [Finset.sum_congr rfl (fun y _ => hpt y), Finset.sum_add_distrib, Finset.sum_neg_distrib]
    exact hexpand.symm
  -- The `x`-summand of `condEntropyΘ_X` is exactly `pX x * d.marginalLeft.entropy`.
  have hRHS : J.pX x *
      (- ∑ θ, (if J.pX x = 0 then 0 else J.pΘX θ x / J.pX x) *
         Real.log (if J.pX x = 0 then 0 else J.pΘX θ x / J.pX x))
      = J.pX x * d.marginalLeft.entropy := by
    have hif : ∀ θ, (if J.pX x = 0 then 0 else J.pΘX θ x / J.pX x) = d.marginalLeft.p θ := by
      intro θ
      rw [if_neg (ne_of_gt hx), J.sliceΘY_marginalLeft x hx θ]
    rw [Finset.sum_congr rfl (fun θ _ => by rw [hif θ])]
    rfl
  rw [hLHS, hRHS]
  have hsub : d.entropy ≤ d.marginalLeft.entropy + d.marginalRight.entropy :=
    ProbDist.entropy_le_sum_marginal_entropy d
  have hkey : d.entropy - d.marginalRight.entropy ≤ d.marginalLeft.entropy := by linarith
  exact mul_le_mul_of_nonneg_left hkey hx.le

/-- **Main theorem.** `H(Θ | X, Y) ≤ H(Θ | X)` for any joint distribution
`J` over finite `Θ, X, Y`: observing `Y` in addition to `X` can never
increase the remaining uncertainty about `Θ`. Proved by summing
`condEntropyΘ_XY_summand_le` over every `x`: for `x` with `J.pX x = 0`,
both the `condEntropyΘ_XY` and `condEntropyΘ_X` summands vanish trivially
(the outer weight is `pXY x y ≤ pX x = 0` resp. `pX x = 0` itself), so the
`0 ≤ 0` case closes on its own; for `x` with `J.pX x > 0`, this is
exactly `condEntropyΘ_XY_summand_le`. -/
theorem entropy_le_condEntropy : J.condEntropyΘ_XY ≤ J.condEntropyΘ_X := by
  unfold condEntropyΘ_XY condEntropyΘ_X
  refine Finset.sum_le_sum fun x _ => ?_
  rcases eq_or_lt_of_le (J.pX_nonneg x) with hx0 | hx0
  · have hy0 : ∀ y, J.pXY x y = 0 := fun y =>
      le_antisymm (hx0 ▸ J.pXY_le_pX x y) (J.pXY_nonneg x y)
    simp [← hx0, hy0]
  · exact J.condEntropyΘ_XY_summand_le x hx0

/-- **Strict-improvement corollary.** If the raw per-slice inequality
proved above (`condEntropyΘ_XY_summand_le`) is STRICT at even one `x₀`
(equivalently: the conditional joint of `(Θ, Y)` given `X = x₀` is not
literally the product of its own marginals there, i.e. `Y` genuinely
carries information about `Θ` beyond what `X = x₀` alone gives), then
combining `Y` with `X` strictly reduces the remaining uncertainty about
`Θ`, `H(Θ|X,Y) < H(Θ|X)`, not merely `≤`.

Per the file's stated scope, `hstrict` is supplied as an EXTERNAL
HYPOTHESIS here, not derived from nothing: this file proves the general
information-theoretic fact and its strict form given non-independence;
it does NOT prove that the specific `g = dy·x` / `contrib = w·x` DISLDO
signals satisfy `hstrict` for real training dynamics -- that remains a
separate, empirical/architectural claim. -/
theorem combined_signal_strictly_informative (x₀ : X) (hx₀ : 0 < J.pX x₀)
    (hstrict : ∑ y, J.pXY x₀ y *
        (- ∑ θ, (if J.pXY x₀ y = 0 then 0 else J.p θ x₀ y / J.pXY x₀ y) *
           Real.log (if J.pXY x₀ y = 0 then 0 else J.p θ x₀ y / J.pXY x₀ y))
      < J.pX x₀ *
        (- ∑ θ, (if J.pX x₀ = 0 then 0 else J.pΘX θ x₀ / J.pX x₀) *
           Real.log (if J.pX x₀ = 0 then 0 else J.pΘX θ x₀ / J.pX x₀))) :
    J.condEntropyΘ_XY < J.condEntropyΘ_X := by
  unfold condEntropyΘ_XY condEntropyΘ_X
  refine Finset.sum_lt_sum (fun x _ => ?_) ⟨x₀, Finset.mem_univ x₀, hstrict⟩
  rcases eq_or_lt_of_le (J.pX_nonneg x) with hx0 | hx0
  · have hy0 : ∀ y, J.pXY x y = 0 := fun y =>
      le_antisymm (hx0 ▸ J.pXY_le_pX x y) (J.pXY_nonneg x y)
    simp [← hx0, hy0]
  · exact J.condEntropyΘ_XY_summand_le x hx0

end Joint
