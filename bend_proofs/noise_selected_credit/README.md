# noise_selected_credit

Why choosing the sparse credit dims with **noise** gives an unbiased credit,
and choosing them by **gradient magnitude** does not.

Run `bend PROOF.bend --check-only` (prints `ALL PROOFS CHECK`; one law). Nat
lemmas come from `../lib/NAT.bend`.

## Setting

sili_quartz's e-BPTT credit (`sili_quartz/models/ebptt.py`, branch
`research/e-bptt-fake-backprop`) routes the gradient G = dL/dM that reaches a
detached Titans memory into its write projections through two eligibility
traces: dL/dW_V = G·E_V, dL/dW_K = Gᵀ·E_K. That contraction, 2d³ per chunk, is
the most expensive part of the credit. `credit_density = p` keeps only k = p·d
of the d memory dims per batch row and skips the rest (cost 2p·d³). The kept
dims are the top k of a score:

- `grad`: |G| per dim
- `grad_noise`: |G| · Φ(z), with z a per-dim gaussian drifting over ~16 tokens
- `noise`: Φ(z) alone

Measured on ledger row 15 (state detached every chunk, p = 0.25, 3 seeds,
3000 steps): `noise` 3/3 seeds ≥ 0.97; `grad` 1/3; `grad_noise` 1/3; the dense
carry 3/3. Selection counts over 1000 steps: under `noise` every dim gets 1/64
of the selections (normalized entropy 1.000); under `grad` the top 8 dims take
29 % (uniform 12.5 %) and the run stays at chance.

## What's proved

`absorption`: `(n+1)·C(n, k) == (k+1)·C(n+1, k+1)` over Nat, with C defined by
Pascal's rule.

Read with d = n+1 memory dims and k+1 kept: count (dim, kept set) incidences
two ways. There are C(d, k+1) kept sets, each holding k+1 dims; and each of the d
dims is in C(d−1, k) of them, the **same number for every dim**. So if every
kept set is equally likely, every dim is kept with probability (k+1)/d. The
expected sparse credit is then exactly (k+1)/d times the dense credit, dim by
dim: an unbiased estimate up to a constant scale, which Adam's normalization
absorbs.

## What is assumed, not proved

- **Every kept set is equally likely.** True when the scores are i.i.d. across
  dims and independent of G, as Φ(z) is: top-k of exchangeable scores is a
  uniform k-subset. That probabilistic step is not in the Bend file; the file
  proves the counting that makes it give equal coverage.
- **Temporal persistence.** z drifts, so consecutive chunks keep mostly the same
  dims, copying biology (credit stays on similar inputs for a while). It is a
  separate property from the unbiasedness above and does not change it: each
  chunk is still marginally a uniform k-subset. Nothing here says what the
  persistence itself does.
- **Top-k by |G| is biased.** That is the contrast, not a theorem here: its kept
  set depends on G, so dims with persistently small |G| are kept less often and
  their credit is under-weighted every step. Stich, Cordonnier and Jaggi,
  "Sparsified SGD with Memory" (NeurIPS 2018), make the same distinction for
  gradient compression: random-k is unbiased and needs no correction; top-k
  needs an error-feedback memory or some coordinates are never updated.
