# fp4 trainability: unbiased stochastic rounding and an exact size bound

Run `bend PROOF.bend --check-only` (prints `ALL PROOFS CHECK`; four laws). Nat
arithmetic only, shared lemmas from `../lib/NAT.bend`.

**Corrected 2026-11-22.** The first version of this README (PR #58) drew three
conclusions the laws do not support: that stored-fp4 training converges as SGD
does, that NVIDIA's NVFP4 result is evidence for it, and that it needs about
1.29x the parameters. All three are withdrawn below. The laws are unchanged.

Question (sili_quartz TODO 2b.1): can sili's fp4 setup (fp4 weights + fp4
optimizer state, Hadamard-product scales and additive low-rank branches, with
annealing) train and learn the recurrent MQAR? If low bits cannot match fp32
per parameter, how much larger must the fp4 model be?

## Setting (sili__new code)

Per synapse, sili stores two fp4 codes (E2M1: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4,
±6; `fp4quant.hpp`):

- **weight** `cw`, read as `W = Q ⊙ S + A`: the code Q times a rank-N scale
  `S(row,col) = Σ_k γ_k u_k(row) v_k(col)` (the Hadamard product), plus an
  additive low-rank branch `A = Σ_k u'_k(row) v'_k(col)` (`delta_csr_types.hpp`).
  A is needed structurally: the gradient through S is zero wherever Q is the
  zero code, and only an additive term can write a value there.
- **importance** `ci`, an RMSprop second moment with per-row and per-column
  scales (`importance_scale`, `output_importance_scale`).
- **update** (`BoundedRMSpropSynapsePolicy`):
  `ci ← β·ci + (1−β)(g² + contrib²)` and
  `Δcw = −lr · clip(g / (√ci + ε)) / S`.
  Both writes go through stochastic rounding. On the live path
  (`block4_codec.hpp`) these are the never-zero variants: weights use
  `fp4_quantize_stochastic_live`, importance uses
  `fp4_quantize_stochastic_live_nonneg`. Code 0 means "no synapse", so a live
  synapse never stores it. Construction and loading stay deterministic.
- Activations, the recurrent state and the additive/scale branches stay fp32.
  Only weights and importance are 4-bit.

That is 8 bits of state per synapse. bf16 weights with fp32 Adam use 80, or
48 with bf16 Adam moments. These are per synapse, not at equal quality.

## What is proved

| law | statement | meaning |
|---|---|---|
| `sr_unbiased` | `a·e + (a+w)·d = (a+d)·w`, with `w = d+e` | stochastic rounding of `x = a+d` on the cell `[a, a+w]` has expectation exactly `x` |
| `sr_second_moment` | `e·d² + d·e² = (d·e)·w` | the rounding error has variance `d·e`, at most `w²/4` |
| `inc_value` | `value(inc(xs)) = 1 + value(xs)` | base-5 carry is exact |
| `encode_value` | `value(encode(n)) = n` | every integer is exactly a list of digits 0–4, each an fp4 value |

## What it means for training

1. **Stochastic learning is required, not optional.** Under round-to-nearest,
   any update smaller than half the fp4 spacing is lost every time. Under
   stochastic rounding each single update is kept in expectation
   (`sr_unbiased`), with added variance `d·e ≤ w²/4` (`sr_second_moment`).
   This is a one-step statement. The trajectory matches fp32 in expectation
   only where the gradient is linear in the weight.
   - **The noise floor.** With no high-precision copy, that variance does not
     go away as lr shrinks (Li et al., NeurIPS 2017, Theorem 5). Measured with
     the real `fp4_quantize_stochastic_live` on `dw = -lr·(w - target)`:

     | target | rms error, lr .1 | .01 | .001 | .0001 | √(d·e) |
     |---|---|---|---|---|---|
     | 0.1 | 0.488 | 0.491 | 0.488 | 0.490 | 0.490 |
     | 0.3 | 0.401 | 0.405 | 0.404 | 0.416 | 0.400 |
     | 1.2 | 0.244 | 0.243 | 0.244 | 0.245 | 0.245 |
     | 3.4 | 0.490 | 0.492 | 0.488 | 0.489 | 0.490 |

     The long-run mean is on target (within 0.022 in every run). The error is
     in units of the scale S, so it can only come down through smaller S,
     averaging over more synapses (width), or something that anneals. That is
     what the Hadamard and additive LoRAs are for: they carry the low-rank part
     in fp32 and set S, leaving the full-rank residual to width.
   - **Crossing zero.** Start +1, target -1, 1000 runs each at lr .1, .01 and
     .001: `fp4_quantize_stochastic` and `_stochastic_live` cross in 1000/1000,
     `fp4_quantize_live` (nearest) and
     `block4_vec_quantize_stochastic_fp4_live` in 0/1000.
   - **The SIMD live quantizer must not be used or ported.** Despite its name,
     `block4_vec_quantize_stochastic_fp4_live` redirects to the value's own
     sign, so for |v| < 0.5 it is deterministic and biased by up to 0.5. Only a
     unit test calls it today: both kernels quantize with the scalar functions.
   - **Never-zero weights stay unbiased.** Below 0.5 in magnitude,
     `_live` rounds stochastically between −0.5 and +0.5 with
     `p(+0.5) = v + 0.5`. That is `sr_unbiased` on a cell of width 1 shifted
     by −0.5 (a shift keeps the expectation exact). The cost is variance: up
     to 1/4 there, against 1/16 on the [0, 0.5] cell when code 0 is allowed,
     so 4× the noise for near-zero weights.
   - **No live synapse can be exactly zero.** Every live synapse adds at
     least ±0.5·S, so a row with n irrelevant live synapses carries noise of
     variance about n·S²/4 in its pre-activation. The only ways to silence a
     synapse are to prune it, shrink the row/column scale (which shrinks every
     synapse sharing it), or cancel it with the additive branch. In biology
     too, a synapse either exists or is pruned.
   - The deterministic `fp4_quantize_live` (used outside the stochastic
     update path) is biased near zero, by design.
2. **The same holds for the 4-bit optimizer state.** The EMA increment
   `(1−β)g²` is tiny next to the fp4 spacing, so round-to-nearest would freeze
   `ci`. With stochastic rounding its expectation tracks the true EMA.
   - **Why 4 bits is enough for importance.** `ci` only sets each synapse's
     step size, and a rounded `ci` changes that step by a bounded factor.
     Preconditioned SGD converges for any preconditioner within fixed positive
     bounds. That last part is standard real analysis, assumed and not proved
     here.
   - **The real risk is range, not precision.** Within one row/column scale,
     E2M1 spans only 0.5 to 6 (12×).
     - *Below the range*, `_live_nonneg` rounds everything under 0.5 up to
       0.5, deterministically. So `ci` has a floor of 0.5·scale and never hits
       code 0. That removes the `g/ε` blow-up the allow-zero quantizer would
       have. It is a bias toward smaller steps: a synapse whose true `ci` is
       under the floor moves slower than RMSprop would move it, never in the
       wrong direction. The preconditioner stays within positive bounds, so
       convergence is assumed to hold, only slower for those synapses.
     - *Above the range*, codes saturate at 6: a deterministic clip of large
       `ci`, which gives larger steps than RMSprop.
     - **Measure the fraction of `ci` at the floor (code 1) and at saturation
       (code 7) during training.**
   - **Clips bias the step.** `max_abs_delta` (binding default 2) caps the
     normalized step and `max_ci` (100) caps the EMA. `max_abs_grad` is a
     no-op on the fp4 binding, which never passes it (1e30). Clipped SGD is biased: it converges
     to a neighbourhood whose size depends on the clip threshold against the
     gradient noise (Koloskova, Hendrikx & Stich, ICML 2023). So the clip
     needs its own check, not an assumption.
3. **The weights don't have to hold the bindings.** In the recurrent MQAR the
   key→value bindings live in the recurrent state (the memory matrix M), not in
   the weights. The weights only implement the write and read. Random sign
   projections, ±1 or sparse {−1, 0, +1} times one scale (Achlioptas 2003), are
   exactly representable in fp4 and give near-orthogonal keys. So binding
   capacity (titans measured ≈ d_k/4) is set by the key width and the *state*
   precision, not the weight precision.
   - Information theory bites in the state: N bindings over vocabulary V need
     `N·log₂V` bits held in `d_k·d_v` entries of M. Quantizing M (or a
     recurrent hybrid's linear-attention state) is the hard case.
4. **Whether annealing is needed is open.** Lowering lr cannot remove the
   noise floor, so annealing has to act on something else (scales, importance,
   which synapses update). It is untested for this, and separately needed
   against rank collapse.

## The comparative size

- **Proved upper bound** (`encode_value`): a b-bit fixed-point weight is
  exactly ⌈b / log₂5⌉ = ⌈b / 2.32⌉ fp4 synapses. Each carries one digit, and
  a rank-1 scale channel gives digit j the row scale 5^j. The digit codes carry
  the weight's sign.
  - 16-bit weights take 7 fp4 synapses, i.e. 28 bits of weight codes, 1.75×.
    This bound uses only 5 of the 15 codes.
  - Balanced signed digits −4…4 (9 codes, all exact in E2M1) give log₂9 =
    3.17 bits per synapse: 6 synapses for 16 bits. Same construction with
    signs; not proved here.
  - It is an *exact emulation* bound: it shows that precision can be bought
    with width, never that it must be.
- **There may be no single size equivalence.** Models split differently
  between precise work and decision (if-then) work: the bound above is the
  price of pure precision, and decision work can cost far less, so the ratio is
  a per-task measurement. The 1.29x first quoted
  here is withdrawn. It came from Kumar et al.'s weight-precision fit
  (`N_eff = N(1 − e^(−P/2.6745))`), where only the forward pass is quantized,
  so it does not describe training with nothing but the 4-bit code stored.
  Extra width does average the rounding noise, and can buy more if-then
  (decision-type) complexity. How much is needed has to be measured.

## Assumed, not proved

- That the noise floor is low enough, at a given width and scale, for the
  task. Nothing here proves end-to-end trainability of MQAR.
- The real-valued behaviour of SGD under a bounded, gradient-dependent
  preconditioner.
- Independent rounding draws. sili uses a per-thread xorshift64*, which is
  fine statistically but not cryptographic.
- That the clips' bias, the importance floor and the saturation at 6 leave
  training close enough to the unbiased case. They are tested below, not
  proved.
- That MQAR's key/value projections train to near-orthogonal keys, rather
  than being fixed random signs.
- The state precision (M stays fp32 in every run so far).

## Predicted test (sili_quartz 2b.2)

Recurrent MQAR, sili fp4 weights + fp4 importance, activations fp32. One
switch per arm:
1. live stochastic rounding (the default): expected to train, slower than fp32
   by the noise;
2. stochastic rounding allowing code 0 (`fp4_quantize_stochastic`): tests the
   never-zero choice (4× near-zero variance, no exact zero) against the
   dead-synapse problem it was added for;
3. clips off (`max_abs_delta`, `max_ci` at no-op values): tests the clip bias;
4. fp32 reference.

No round-to-nearest arm: `sr_unbiased` shows what it loses, and the
zero-crossing measurement above shows the stall (0/1000). Biology's synaptic
changes are stochastic too. NVFP4 is not evidence either way: it rounds weights
to nearest over an fp32 master copy.

Log the fraction of `ci` at the floor and at saturation, and the fraction of
weights at ±0.5. If arm 1 fails, these logs and arms 2–3 say whether the cause
is the never-zero choice, the clips, or something else (range, state
precision).

## References

- S. Gupta, A. Agrawal, K. Gopalakrishnan, P. Narayanan, "Deep Learning with
  Limited Numerical Precision", ICML 2015.
- T. Kumar et al., "Scaling Laws for Precision", ICLR 2025,
  arXiv:2411.04330 (Table 2 constants).
- NVIDIA, "Pretraining Large Language Models with NVFP4", arXiv:2509.25149.
  Master weights and optimizer state are fp32, weights round to nearest,
  stochastic rounding is applied to gradients only (Sec. 4, 4.1, 4.4). So it
  is not a result about stored fp4 weights.
- H. Li, S. De, Z. Xu, C. Studer, H. Samet, T. Goldstein, "Training Quantized
  Nets: A Deeper Understanding", NeurIPS 2017, arXiv:1706.02379.
- B. Chmiel et al., "FP4 All the Way" (2025), the format sili's table follows.
- A. Koloskova, H. Hendrikx, S. U. Stich, "Revisiting Gradient Clipping:
  Stochastic Bias and Tight Convergence Guarantees", ICML 2023.
- D. Achlioptas, "Database-friendly random projections", JCSS 2003.
