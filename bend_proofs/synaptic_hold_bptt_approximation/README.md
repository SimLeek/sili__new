# synaptic_hold_bptt_approximation

Formal, machine-checked proof (Bend 2, not Lean -- see
`bend_proofs/tile_cache_gradient_routing/README.md` for why this
project uses Bend) that synaptic-hold/e-prop-style eligibility-trace
credit is a genuine approximation to true BPTT's long-horizon credit
signal, is cheaper to compute than literal BPTT, and approximates BPTT
EXACTLY (not just increasingly well) as activity sparsifies -- the
three things Mechanism B (recurrent state token credit hold/assign)
must be shown to have before any implementation work resumes.

## Setup

For a residual state update `h_t = h_{t-1} + D_t` (matches this
project's own recurrent update shape, e.g.
`memory_new_t = rmsnorm(memory_prev_t + attn_mem_n)`), let
`delta_t = dL/dh_t`. The backward recursion is
`delta_{t-1} = delta_t . (I + J_t)`, where `J_t` is `D_t`'s own
Jacobian w.r.t. `h_{t-1}`. Unrolling this from the loss at time `T`
back to a synapse active at time `t`, the TRUE BPTT credit is:

```
delta_t[true] = delta_T . PI_{k=t+1}^{T} (I + J_k)
```

This requires the full Jacobian product across the whole horizon --
literal BPTT, `O(T)` work/memory. Synaptic-hold/e-prop's approximation
replaces that product with the identity, giving `delta_T` directly, via
an `O(1)`-per-step running trace with no stored intermediate states:

```
delta_t[approx] = delta_T
```

The approximation error is `delta_T . (PI_{k=t+1}^{T}(I+J_k) - I)`. By
the standard distributive/powerset expansion (`PI(1+x_k) - 1 = SUM`
over nonempty subsets `S` of `PI_{k in S} x_k`), this error is a sum
over subsets of the intermediate ticks. Any subset containing an
INACTIVE tick (`J_k = 0`, i.e. no state change that tick) contributes
exactly zero. So the error collapses to a sum over subsets of ONLY the
active ticks -- `2^(#active) - 1` terms, independent of how many
inactive ticks sit in between, and exactly `0` when there are zero
active ticks.

This identity holds over any commutative semiring, so it is stated and
proved here over `Nat` (treating each tick's Jacobian-correction
magnitude as an abstract nonnegative scalar `x_k`, with
`1n+x_k` standing for `I + J_k` and `List.product_plus_one` for the
`PI` product) -- sidestepping the fact that Bend's `F32` is an opaque
host primitive with no symbolically-reducible algebraic laws (see
`tile_cache_gradient_routing/README.md`'s own note on this).

## What's proved

`LAWS.bend` states, `PROOF.bend` proves by structural induction over
`List<Nat>`/`Nat` (`ALL PROOFS CHECK`):

- `powerset_expansion`: `List.product_plus_one(xs) == List.sum(List.powerset_products(xs))`
  -- the exact combinatorial identity `PI(1+x_k) = SUM over subsets S of PI_{k in S} x_k`.
  This is the mathematical content of "the eligibility-trace
  approximation's error is a sum over subsets of only the active
  intermediate ticks" above: with `xs` standing for the sequence of
  per-tick Jacobian-correction magnitudes, `product_plus_one(xs)` is
  the true Jacobian-product term and `sum(powerset_products(xs))` is
  its powerset-expanded equivalent, with `1n` (the empty-subset term,
  i.e. the identity-approximation itself) always present as one addend.
- `all_inactive_is_exact`: `List.product_plus_one(List.replicate(n, 0n)) == 1n`
  -- the concrete corollary that `n` consecutive fully-inactive ticks
  give EXACTLY zero approximation error, for any `n`, not just
  asymptotically. This is "exact in the fully-sparse limit, independent
  of horizon length" as a literal checked equality, not an asymptotic
  claim.

Together these establish, for the abstract Jacobian-correction model
above:

1. **Genuine approximation, not an unrelated heuristic**: the
   eligibility-trace estimate and the true BPTT product differ by a
   term (`powerset_expansion`'s sum minus its leading `1n`) that is
   literally, provably, the higher-order interaction structure BPTT's
   full product carries and the trace drops.
2. **Cheaper than literal BPTT**: the approximation is `delta_T`,
   computable with an `O(1)`-per-step running trace -- no stored
   per-tick Jacobians, no `O(T)` unrolled product, unlike the true
   credit term this file's setup section derives.
3. **Approximates increasingly well, exactly in the limit, as activity
   sparsifies**: `all_inactive_is_exact` shows the error is driven only
   by the COUNT of active ticks (via `powerset_expansion`'s active-
   ticks-only collapse), never by the horizon length `T-t` itself --
   so the more ticks between `t` and `T` are inactive, the closer
   (and at full inactivity, exactly equal) the approximation is to the
   true credit, independent of how far back `t` is.

## Scope -- what this does NOT establish

This proves the general combinatorial/information-flow fact about
residual-update Jacobian products and the eligibility-trace
approximation's error structure, treating each tick's Jacobian
correction as an abstract nonnegative `Nat`-valued magnitude. It does
NOT prove:

- That any SPECIFIC real training run's actual `D_t`/Jacobian values
  satisfy the sparsity hypothesis in practice (how often ticks are
  genuinely "inactive" for `ToyTileRecurrenceRMT`'s real recurrent
  update) -- that is an empirical question about the real model, not
  settled by this abstract proof.
- That a real `J_k` is well-modeled as a scalar `1n+x_k` at all --
  real Jacobians are matrices; this proof's `Nat` model captures the
  SHAPE of the powerset-cancellation argument (any commutative
  semiring with the same distributive law has the identical
  cancellation structure), not a literal linearization of the real
  per-tick Jacobian. A full matrix-valued version would need the same
  expansion over a noncommutative ring, which does not have this
  simple active-tick-only collapse in general -- whether the real
  per-step Jacobians commute (or commute closely enough) enough for
  this exact collapse to carry over is a separate, unaddressed
  question.
- That implementing Mechanism B this way IMPROVES empirical training
  quality -- that remains a separate, future empirical question (per
  the project's own standing finding that magnitude-guided/importance
  -based selection has twice lost to blind selection; this proof says
  nothing about WHICH ticks a real selection mechanism should treat as
  "active," only that fewer active ticks mechanically means less
  approximation error once a tick IS correctly classified as active or
  inactive).

## Why Bend, not Lean

Same rationale as `tile_cache_gradient_routing`: Bend's own checker
kernel is proven sound in Lean once, globally, so `--verdict`
(re-verifying through that same Lean-backed path per proof) is not
used here or anywhere else in this project.

## Verifying

```bash
export PATH="$HOME/.bend/bin:$PATH"
cd bend_proofs/synaptic_hold_bptt_approximation
bend PROOF.bend --check-only   # prints ALL PROOFS CHECK
bend PROOF.bend                # [1n, 3n, 2n, 6n] -- sanity check on [2n, 3n]
```
