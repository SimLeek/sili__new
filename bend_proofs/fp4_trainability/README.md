# fp4 trainability: unbiased stochastic rounding and an exact size bound

Run `bend PROOF.bend --check-only` (prints `ALL PROOFS CHECK`; four laws). Nat
arithmetic only, shared lemmas from `../lib/NAT.bend`.

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
48 with bf16 Adam moments.

## What is proved

| law | statement | meaning |
|---|---|---|
| `sr_unbiased` | `a·e + (a+w)·d = (a+d)·w`, with `w = d+e` | stochastic rounding of `x = a+d` on the cell `[a, a+w]` has expectation exactly `x` |
| `sr_second_moment` | `e·d² + d·e² = (d·e)·w` | the rounding error has variance `d·e`, at most `w²/4` |
| `inc_value` | `value(inc(xs)) = 1 + value(xs)` | base-5 carry is exact |
| `encode_value` | `value(encode(n)) = n` | every integer is exactly a list of digits 0–4, each an fp4 value |

## What it means for training

1. **Stochastic learning is required, not optional.** Under round-to-nearest,
   any update smaller than half the fp4 spacing is lost every time, so the
   expected drift is zero and training stalls once steps get small. Under
   stochastic rounding, the stored weight follows the fp32 trajectory in
   expectation (`sr_unbiased`). Each step adds zero-mean noise of variance at
   most `w²/4` (`sr_second_moment`). Unbiased plus bounded variance is the
   standard SGD setting, so fp4 training converges as SGD does, only noisier.
   (Gupta et al. 2015; NVIDIA's NVFP4 pretraining uses stochastic rounding on
   gradients for the same reason.)
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
   - **Clips bias the step.** `max_abs_delta` caps the normalized step and
     `max_abs_grad` caps g before the EMA. Clipped SGD is biased: it converges
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
4. **Annealing is not needed for trainability.** It is needed for keeping
   capacity over long training: rank collapse, and the forced-firing /
   renorm results in sili_peridot.

## The comparative size

- **Proved upper bound** (`encode_value`): a b-bit fixed-point weight is
  exactly ⌈b / log₂5⌉ = ⌈b / 2.32⌉ fp4 synapses. Each carries one digit, and
  a rank-1 scale channel gives digit j the row scale 5^j. Sign uses a mirrored
  copy.
  - 16-bit weights take 7 fp4 synapses, i.e. 28 bits of weight codes, 1.75×.
    This bound uses only 5 of the 15 codes.
  - Balanced signed digits −4…4 (9 codes, all exact in E2M1) give log₂9 =
    3.17 bits per synapse: 6 synapses for 16 bits. Same construction with
    signs; not proved here.
  - It is an *exact emulation* bound: it shows that precision can be bought
    with width, never that it must be.
- **Measured** (Kumar et al., "Scaling Laws for Precision", ICLR 2025; their
  weight-only setting matches sili: weights 4-bit, activations high
  precision):
  `N_eff = N(1 − e^(−P/γ_w))`, with fitted γ_w = 2.6745 for integer training
  precision (2.8111 for their floating-point fit).
  - 4-bit weights: N_eff ≈ 0.776·N. 16-bit: 0.9975·N.
  - So 4-bit-weight training needs ≈ **1.29× the parameters** for the same
    loss.
  - Caveats: that is quantization-aware transformer pretraining. The
    constants depend on architecture and data. (The paper's "more than 4x
    larger" warning is for 4-bit activations/KV too, which sili does not do.)
- **Memory at equal quality.** sili's 8 bits/synapse × 1.29 ≈ 10.3 bits,
  vs 48–80 for bf16 + Adam.
  - The extra width can go to more units: more if-then (decision-type)
    complexity in place of per-weight precision.
  - Whether that beats the precision it replaces is the open empirical
    question for 2b.2.

## Assumed, not proved

- The real-valued convergence of SGD with unbiased bounded noise and a bounded
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
2. round-to-nearest on the update sites: expected to stall once steps fall
   below half the spacing;
3. stochastic rounding allowing code 0 (`fp4_quantize_stochastic`): tests the
   never-zero choice (4× near-zero variance, no exact zero) against the
   dead-synapse problem it was added for;
4. clips off (`max_abs_delta`, `max_abs_grad` at their no-op values): tests the
   clip bias;
5. fp32 reference.

Log the fraction of `ci` at the floor and at saturation, and the fraction of
weights at ±0.5. If arm 1 fails while arm 2's failure looks the same, the cause
is something other than rounding bias (range, clips, state precision).

## References

- S. Gupta, A. Agrawal, K. Gopalakrishnan, P. Narayanan, "Deep Learning with
  Limited Numerical Precision", ICML 2015.
- T. Kumar et al., "Scaling Laws for Precision", ICLR 2025,
  arXiv:2411.04330 (Table 2 constants).
- NVIDIA, "Pretraining Large Language Models with NVFP4", arXiv:2509.25149
  (12B hybrid Mamba-Transformer, 10T tokens. Stochastic rounding on gradients,
  random Hadamard transforms on weight-gradient GEMM inputs, 2D block scaling.
  They also keep ~15% of linear layers in BF16. sili does not plan to: biology
  has no such high-precision layers.)
- B. Chmiel et al., "FP4 All the Way" (2025), the format sili's table follows.
- A. Koloskova, H. Hendrikx, S. U. Stich, "Revisiting Gradient Clipping:
  Stochastic Bias and Tight Convergence Guarantees", ICML 2023.
- D. Achlioptas, "Database-friendly random projections", JCSS 2003.
