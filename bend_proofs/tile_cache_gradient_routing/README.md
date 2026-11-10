# tile_cache_gradient_routing

Formal, machine-checked proof (Bend 2, not Lean -- see note below) of the
list-indexing fact underlying `sili_peridot`'s tile-cache exact credit
mechanism (`model/tile_cache_credit.py`, `model/toy_tile_recurrence_rmt.py`'s
`step_cached()`).

## What's proved

`LAWS.bend` states, `PROOF.bend` proves by structural induction (both
`ALL PROOFS CHECK`):

- `take_append_left`: `List.take(append(a, b), length(a)) == a`
- `drop_append_right`: `List.drop(append(a, b), length(a)) == b`

That is: splitting an appended list back apart at the boundary used to
build it recovers the two original pieces exactly, in order, with
nothing dropped, duplicated, or shifted by one position.

## Why this matters

`step_cached()` builds `k_content_full = concat([cache_k_tensor,
k_new], axis=0)` (same shape for v). When the real training loop's
`loss.backward()` reaches this concat, autodiff hands it ONE combined
gradient sized to the whole concatenated list, and concat's own
backward rule must route each half back to the matching input -- the
cached block gets the gradient's first `length(cache)` rows, the new
row gets the rest. `tile_cache_credit.py` reads `cache_k_tensor.grad`
directly off this routing (confirmed empirically this session: `Tensor`
has no `requires_grad` gate, so a "detached" leaf still receives a real
gradient from any backward chain that reaches it -- see JOURNAL.md).
If the routing were off by even one position, `cache_k_tensor.grad`
would silently hold the WRONG per-row values: plausible-looking
numbers, no crash, nothing an empirical "it ran, and quality went up"
smoke test can rule out on its own. This proof rules that specific
failure mode out directly, rather than trusting it by inspection.

## Scope -- what this does NOT establish

This proves the list-indexing fact in the abstract (structural
induction over `List<F32>`, treating each F32 as an opaque element
never computed on). It does NOT prove:

- That `gaussian_attention`'s own backward (`_cpu.gaussian_attention_backward`,
  C++) computes the mathematically correct local Jacobian-vector
  product in the first place -- that remains an empirical/architectural
  trust boundary, same as this repo's Lean proof
  (`lean_proofs/importance_signal_information_gain`) explicitly declines
  to prove its own signals' real-training non-independence.
- That summing a value's gradient contributions across multiple uses
  (`cache_k_tensor` feeds both pass 1 and pass 2 of `step_cached()`) is
  the mathematically correct total derivative -- that is the standard
  multivariate chain rule's sum rule for a fanned-out node, taken as
  given, not re-derived here (see `LAWS.bend`'s own note on this).
  Float addition is an opaque host primitive in this proof, not
  something reduced symbolically for abstract variables -- there is no
  claim here about floating-point commutativity/associativity under
  real rounding.
- Anything about whether the resulting correction mechanism actually
  IMPROVES training quality -- that is a separate empirical question
  (see JOURNAL.md's statistical comparisons), unaffected by this proof
  either way.

## Why Bend, not Lean

This repo's existing formal proof (`lean_proofs/importance_signal_information_gain`)
uses Lean. This one uses Bend 2 instead, per direct instruction:
Bend's own checker kernel is itself proven sound in Lean (once,
globally -- `bend2/bendtt.lean`), so re-verifying each individual Bend
proof through that same Lean-backed kernel (`bend PROOF.bend --verdict`)
is redundant overhead that defeats the actual reason for using Bend
(order-of-magnitude faster proof checking) -- `--verdict` is not used
here or in any future Bend proof in this project.

## Verifying

```bash
curl -fsSL https://bend-lang.com/install.sh | sh   # one-time
export PATH="$HOME/.bend/bin:$PATH"
cd bend_proofs/tile_cache_gradient_routing
bend PROOF.bend --check-only   # prints ALL PROOFS CHECK
```
