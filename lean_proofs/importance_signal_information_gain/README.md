# importance_signal_information_gain

Formal, machine-checked proof of the information-theoretic claim behind
combining a forward-contribution signal with a backward-sensitivity
signal in DISLDO's per-synapse importance estimator
(`sili__new/sili/lib/headers/linear_disldo.hpp`).

## What's proved

`SiliImportanceProof/ImportanceSignalInformationGain.lean` proves, from
first principles (Gibbs' inequality, itself proved here rather than
merely cited):

- `gibbs_inequality` -- non-negativity of KL divergence.
- `ProbDist.entropy_le_sum_marginal_entropy` -- subadditivity of joint
  entropy, `H(A,B) ≤ H(A) + H(B)`.
- `Joint.entropy_le_condEntropy` -- the main theorem: for any joint
  distribution over finite `Θ, X, Y`, `H(Θ|X,Y) ≤ H(Θ|X)`. Observing an
  additional signal can never increase uncertainty about a hidden
  quantity.
- `Joint.combined_signal_strictly_informative` -- the strict corollary:
  if the two signals are not conditionally independent (an explicit,
  externally-supplied hypothesis, never derived from nothing), combining
  them strictly reduces uncertainty.

All sorry-free; `#print axioms` on each theorem shows only the three
standard classical-logic axioms (`propext`, `Classical.choice`,
`Quot.sound`).

## Scope

This file proves the GENERAL, domain-independent information-theoretic
fact. It does NOT prove that DISLDO's specific `g = dy·x` (backward
sensitivity) and `contrib = w·x` (forward contribution) signals satisfy
the non-independence hypothesis for real training dynamics -- that is a
separate, empirical/architectural claim, out of scope for this proof.

## What this justifies in the C++ code

`linear_disldo.hpp`'s per-synapse importance update (and its mirror one
level up, `value_scale_importance`/`output_scale_importance` in
`delta_csr_types.hpp`) combines the two signals additively, SUM first
then square -- `(g+contrib)^2`, not `g^2+contrib^2` -- before feeding the
RMSprop-style second moment. Sum-then-square lets the two signals cancel
when they disagree in sign (evidence of noise: the synapse's current
value and the task's error signal conflict), while reinforcing when they
agree (evidence of genuine importance). This proof does not derive the
SPECIFIC combination formula (additive vs. sum-then-square vs. any other
consistent scheme) -- that is an engineering decision, made independently
and justified by direct algebraic and design analysis, not extracted
mechanically from the theorem. What the proof underwrites is the
qualitative claim used to justify combining the two signals at all:
doing so cannot increase (and, when they are dependent, strictly
decreases) the remaining uncertainty about the synapse's true
importance.

## Verifying

```bash
cd lean_proofs/importance_signal_information_gain
lake build SiliImportanceProof
```

Dependencies (Mathlib and its transitive sub-dependencies) are pinned in
`lake-manifest.json` to the exact commits Mathlib's own manifest
specifies for the `v4.33.0` toolchain (see `lean-toolchain`) -- not
arbitrary "latest" versions, which will fail to build (Lean core and
Batteries drift out of sync otherwise).
