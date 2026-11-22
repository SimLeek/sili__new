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

## Why being unbiased matters (known theory, not proved here)

The proof shows the noise choice is unbiased. It does not show why that makes
training work. That part is standard stochastic-optimization theory for
compressed gradients. It is the nearest known model of this setup, not a proof
about it.

- **Unbiased sparsification is SGD with extra zero-mean noise.** Rand-k, scaled
  by d/k, satisfies E[C(g)] = g and E‖C(g) − g‖² = (d/k − 1)‖g‖². Every step
  points along the true gradient on average; only the variance grows, by the
  factor ω = d/k − 1 (3 at k/d = 1/4). SGD convergence results then hold with
  that larger variance term: the method converges, more slowly. That matches
  what was measured: noise p = 0.25 reaches the same accuracy as the dense
  credit, with a median of 1750 vs 1250 steps to 0.95.
- **Biased sparsification without a correction can fail to converge.** If
  E[C(g)] ≠ g, the error is systematic and does not average out over steps.
  Karimireddy et al. 2019 give simple convex problems where a biased
  compressor (signSGD) does not converge to the optimum. Stich et al. 2018
  note that top-k without error accumulation leaves some coordinates never
  updated.
- **Scope.** This is not literally gradient compression. Here the selection is
  over the inner index of the credit contraction (memory dims), the result
  goes into Adam rather than plain SGD, and each step's selection is made per
  batch row. The measured |G| stalls are consistent with the bias explanation,
  but are not shown to be caused by it.

## Error feedback: the known fix for biased top-k, and whether it beats random

**What it is.** Keep a residual of everything the compressor dropped and add it
back before the next compression:

    p_t = e_t + g_t,    send C(p_t),    e_{t+1} = p_t − C(p_t)

Nothing is lost permanently: a coordinate skipped now accumulates until it is
large enough to be kept. It dates to 1-bit SGD (Seide et al. 2014) and is
analysed by Stich, Cordonnier and Jaggi 2018 ("Sparsified SGD with Memory").
They show top-k or rand-k with this memory keeps SGD's convergence rate.
Karimireddy et al. 2019 show EF-SGD with any compressor of this kind matches
SGD's rate with no extra assumptions. EF21 (Richtárik, Sokolov and Fatkhullin,
NeurIPS 2021) compresses the change in a running gradient estimate instead.
It gets an O(1/T) nonconvex rate under standard assumptions (older EF
analyses gave O(1/T^(2/3)) and needed bounded gradients), and it outperforms
the original EF in their experiments.

**Is there a mathematical version that outperforms pure random?** In the
compression model, yes. Beznosikov, Horváth, Richtárik and Safaryan, "On
Biased Compression for Distributed Learning" (JMLR 2023), analyse biased
compressors directly. Top-k is always at least as contractive as rand-k (it
keeps at least a k/d share of ‖g‖², rand-k keeps exactly that on average).
Under their model of i.i.d. uniform entries, top-k has roughly (1 − k/d)²
times less variance than rand-k. With error feedback, top-k then converges at
least as fast in theory and usually faster in practice. Without error
feedback, the guarantees above do not apply to top-k.

**Why it is not a drop-in here.** Error feedback needs the residual of what
was skipped. In this credit step the skipped part is G·E over the unselected
memory dims. Computing it exactly costs the dense contraction, which is the
work the sparsity was meant to save. Storing the skipped columns of G and
applying them later is wrong, because E keeps changing as writes happen.
Keeping the residual at the weight-gradient level also needs the dense
product. A cheap, correct error feedback for this setting is an open
question; nothing tested here implements one. Random selection gets
unbiasedness for free, with no residual state, which is the practical reason
it is the default.

## References

- F. Seide, H. Fu, J. Droppo, G. Li, D. Yu. 1-bit stochastic gradient descent
  and its application to data-parallel distributed training of speech DNNs.
  Interspeech 2014.
- S. U. Stich, J.-B. Cordonnier, M. Jaggi. Sparsified SGD with Memory.
  NeurIPS 2018.
- S. P. Karimireddy, Q. Rebjock, S. U. Stich, M. Jaggi. Error Feedback Fixes
  SignSGD and other Gradient Compression Schemes. ICML 2019.
- P. Richtárik, I. Sokolov, I. Fatkhullin. EF21: A New, Simpler, Theoretically
  Better, and Practically Faster Error Feedback. NeurIPS 2021.
- A. Beznosikov, S. Horváth, P. Richtárik, M. Safaryan. On Biased Compression
  for Distributed Learning. JMLR 24, 2023.
