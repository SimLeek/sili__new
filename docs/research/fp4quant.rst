``fp4quant.hpp`` research notes
==================================

Companion doc to ``sili/lib/headers/fp4quant.hpp``. Source comments point
back here by anchor ID (``*ID:*`` marker under each heading below); this doc
links back to source by function name. See ``docs/research/delta_csr_types.rst``
for the pattern this follows (semantic dotted anchor IDs, visible ID markers,
frozen code snippets on real-bug/non-obvious-derivation sections).

.. _fp4_codec.bitshift_design:

Bit-shift codec: OCP MXFP4 E2M1 with a repurposed NaN slot
------------------------------------------------------------

*ID:* ``fp4_codec.bitshift_design``

``FP4_TABLE``'s 16 entries are exactly OCP MXFP4 E2M1 (2 exponent bits, 1
mantissa bit, bias 1) with one deliberate repurposing: slot 8
(sign=1,exp=00,mant=0, which raw E2M1 would read as -0.0) stores NaN instead.
That's a real, intentional difference from raw E2M1, not a bug to route
around -- every function in this file that touches slot 8 does so on purpose.

``fp4_decode_bits``/``fp4_encode_bits`` replace an older linear-scan-over-
``FP4_TABLE`` implementation with direct bit manipulation:

- **Decode**: E2M1's exponent/mantissa fields slot directly into an
  IEEE-754 float32's fields with a re-bias (E2M1 bias 1 -> float32 bias 127,
  so ``exp_field = e2m1_exp + 126``) and the single mantissa bit placed at
  float32's top mantissa bit -- exact, no rounding, verified bit-for-bit
  against ``FP4_TABLE`` for all 16 codes. Only ``exp==0`` (the subnormal
  slot: values 0/NaN/-0.5 depending on sign+mantissa) needs separate
  handling; every other code is one shift+mask+bitcast.
- **Encode**: nearest-value quantization via the standard low-precision-ML
  technique (also how real E2M1 hardware casts work) -- add a rounding bias
  at the mantissa truncation point and let integer addition's carry
  propagate the rounding through the exponent (1.75 -> 2.0 falls out of the
  carry automatically, no separate case needed), then saturate at the max
  representable magnitude (6.0). The one further special case is the
  ``[0.25, 1.0)`` magnitude range, which straddles the subnormal/normal
  boundary the generic carry trick doesn't span on its own.

Tie-breaking here (nearest, with exact float ties resolved by whichever way
the bit arithmetic naturally falls, e.g. midpoints round up in magnitude) is
NOT required to match the old linear-scan's tie convention (which favoured
lower table index on a tie) -- the table isn't a frozen external format, just
this codebase's own choice, and exact-tie floats essentially never occur in
real gradient-driven data. ``fp4_quantize()`` is now defined IN TERMS OF this
encoder, not the other way around; ``fp4_quantize(NaN) == 0`` is preserved
(every ``|NaN - table[i]|`` comparison in the old scan was false, so ``best``
never moved off its 0 initial value -- the new encoder special-cases NaN
input to return 0 directly to match).

.. code-block:: cpp

   // as of PR #45, fp4quant.hpp -- fp4_encode_bits' carry-propagation trick:
   uint32_t rounded = abits + (1u << 21);
   if (rounded > SIX) rounded = SIX;
   const uint32_t exp_field = (rounded >> 23) & 0xFFu;
   const uint32_t m = (rounded >> 22) & 1u;
   mag_code = ((exp_field - 126u) << 1) | m;
   // A near-zero input must land on code 0 regardless of sign -- never the
   // repurposed NaN slot (8 = sign 1, magnitude 0).
   if (mag_code == 0) return 0;

.. _fp4_quantize_live.never_zero_rationale:

``_live`` variants: why a live synapse's weight must never be code 0
-------------------------------------------------------------------------

*ID:* ``fp4_quantize_live.never_zero_rationale``

A LIVE synapse's weight (never importance, never a genuinely blank/
unallocated storage slot -- see ``block4.hpp``/``delta_csr_memory.hpp``'s own
"byte==0 means blank" convention, which these functions must never be used to
write) must never quantize to code 0: once a synapse's weight AND importance
both land on 0, ``disldo_backward`` treats the whole row/tile as having zero
live connections and excludes it from the normal per-synapse gradient path
entirely, recoverable only via a much slower separate dead-row bootstrap
fallback -- confirmed directly via ``tests/unit/test_scale_handling.cpp``'s
"near-autapse" tests. ``fp4_encode_bits_live``/``fp4_quantize_live`` redirect
the near-zero region to the smallest nonzero magnitude (code 1 = 0.5) instead
of code 0 -- codes go directly from -0.5 to +0.5 with nothing between.

**Deliberately biased near zero**: true values very close to 0 no longer
average to 0 (``E[quantized] != v`` in that region, unlike ``fp4_quantize``'s
own unbiased-elsewhere convention). This is an intentional tradeoff, not an
oversight -- confirmed empirically via a near-autapse harness across 6 seeds
x 4 targets including exactly 0.0: the resulting sign settles quickly and
stays put rather than thrashing, and overall accuracy is comparable to or
better than allowing code 0, because the baseline's occasional
multi-hundred-step "stuck dead synapse" episodes cost far more than this
variant's small steady-state discretization floor.

**NaN input**: a NaN weight reaching quantization is itself a distinct bug
class (upstream gradient corruption). The never-0 invariant shouldn't carve
out an exception for it, so ``fp4_encode_bits_live`` redirects NaN to a
signed code 1 rather than ``fp4_encode_bits``'s own code-0 convention for
NaN.

.. _fp4_quantize_stochastic.unbiased_rounding_rationale:

Stochastic rounding: why deterministic quantize can't train
------------------------------------------------------------------

*ID:* ``fp4_quantize_stochastic.unbiased_rounding_rationale``

``fp4_quantize()`` is deterministic nearest-neighbour: a gradient-driven
update too small to cross the midpoint between two ``FP4_TABLE`` entries is
silently discarded, every single time, with no memory of the near-miss --
there's no persistent float32 "master weight" anywhere in this storage (see
``linear_disldo.hpp``'s ``disldo_forward``/``disldo_backward``, which
dequantize the CURRENTLY-STORED code, add the update, and requantize
immediately). That's fine for one-shot construction/conversion
(nearest-neighbour is the best single-shot approximation) but makes
small/gradual training updates impossible: real per-synapse gradients here
are routinely too small to cross even the finest FP4 gap (0->0.5) in one
step, and with round-to-nearest they never accumulate toward doing so on a
later step either.

Standard fix from low-precision training literature (Gupta et al. 2015,
"Deep Learning with Limited Numerical Precision", and used by essentially
every training scheme with quantized weights and no master-weight copy,
including modern large-model FP4 training): STOCHASTIC rounding. Round up or
down with probability proportional to how close ``v`` is to each neighbour,
so ``E[quantized value] == v`` exactly. A single small update then has a
small but real, unbiased chance of flipping the stored code, and the
expected drift over many steps correctly tracks the true (unquantized)
gradient signal -- unlike round-to-nearest, which has no expected drift at
all for sub-half-gap updates.

Deliberately NOT a replacement for ``fp4_quantize()`` -- only the
gradient-driven update sites (``disldo_forward``'s importance update,
``disldo_backward``'s weight+importance update) should use the stochastic
variants; construction/loading/compact/synaptogenesis-insert must stay
exactly deterministic (repacking or reloading the same content must not
randomly perturb existing values).

.. _fp4_stochastic_rng.thread_local_design:

Stochastic-rounding RNG: thread-local, single draw per call
------------------------------------------------------------------

*ID:* ``fp4_stochastic_rng.thread_local_design``

``fp4_stochastic_rng_state()`` is a fast, thread-local, non-cryptographic
PRNG (xorshift64*) -- called from within OpenMP-parallelized per-synapse
loops, so a shared/global generator would mean either a data race or lock
contention on every single synapse update. Seeded once per thread from its
id by default; explicitly reseedable via ``fp4_seed_stochastic_rng()`` for
tests that need reproducibility, same precedent as ``EnergyDynamics``'s own
unseeded-by-default exploration noise (``np.random.seed(0)`` pinned by
callers that need it, see ``test_column_averaging_predictive.py``).
``fp4_seed_stochastic_rng()`` reseeds only the CALLING thread: fine for
single-threaded callers (tests, small examples), which get full
reproducibility this way. A multi-threaded (OpenMP) caller would need to
call it once per worker thread to pin all of them, which no caller currently
needs -- training runs are meant to be stochastic across threads too; this
is for unit-test determinism, not for controlling a real training run's
outcome.

``fp4_stochastic_next_u64()`` is shared by ``fp4_stochastic_uniform01()``
(top 24 bits) and ``fp4_quantize_stochastic()``'s dithered rounding
(different bit slices of the SAME draw, not a second RNG step) --
xorshift64*'s bits are well-mixed enough that slicing different ranges for
different purposes within one draw is fine, and it matters here: it keeps
stochastic quantize at exactly one RNG step per call.

.. _fp4_quantize_stochastic.dithered_rounding_design:

Dithered rounding: exploiting the mantissa/interpolation identity
------------------------------------------------------------------------

*ID:* ``fp4_quantize_stochastic.dithered_rounding_design``

``fp4_quantize_stochastic()`` is unbiased (``E[result] == v`` for ``v``
within the representable range ``[-6,6]``; clamps deterministically outside
it, same as ``fp4_quantize()`` would). Bit-shift/dithered-rounding
implementation, not a linear bracket scan over ``FP4_SORTED_IDX`` (that
table is kept for GPU/other-device use per direction, but is no longer this
function's own CPU path). Two regimes, matching ``fp4_encode_bits``'s own
split:

- ``|v| >= 1.0`` (the "normal" E2M1 region): the interpolation fraction
  between the two neighbouring representable values is EXACTLY the value's
  own discarded IEEE mantissa bits (below the one bit E2M1 keeps),
  normalized to [0,1) -- provably, algebraically, not an approximation (both
  endpoints of any such bracket share the same IEEE exponent, so the
  bracket's linear interpolation fraction and the mantissa's fractional
  position within that exponent coincide exactly). That makes the classic
  dithered-rounding trick exact here: add a UNIFORM RANDOM integer spanning
  the discarded bits' full range (not the deterministic encoder's fixed
  ``1<<21`` bias), then truncate -- integer addition's carry propagates a
  rounding UP through the exponent as needed, precisely like the
  deterministic version, just probabilistically instead of via a fixed
  round-to-nearest bias.
- ``|v| < 1.0`` (the subnormal/normal-transition region, magnitudes
  0/0.5/1.0 only): straddles E2M1's own subnormal boundary the same way
  ``fp4_encode_bits``'s 3-way split does, and doesn't have the "discarded
  mantissa bits = interpolation fraction" property (no shared exponent
  bracket to exploit) -- but the two sub-brackets here (``[0,0.5)`` and
  ``[0.5,1.0)``) are each linear in ``v`` directly, so the interpolation
  fraction is just a plain multiply (``2v``, or ``2v-1``), no bit tricks
  needed.

.. _fp4_quantize_stochastic_live.probability_weighted_redirect:

Live stochastic quantize: a genuine probability redirect, not a fixed outcome
--------------------------------------------------------------------------------

*ID:* ``fp4_quantize_stochastic_live.probability_weighted_redirect``

``fp4_quantize_stochastic_live()`` is NOT a collapse to a fixed "always code
1" outcome -- an earlier draft of this design got this wrong. The existing
``fp4_quantize_stochastic``'s ``abits < HALF_BITS`` bracket (``[0,0.5)``)
already does genuine probability-weighted rounding for ``v>=0`` (draws
``p_up=v*2``, picks code 0 vs 1 with probability continuously varying by
where ``v`` sits in the bracket). The live version generalizes that SAME
mechanism across the full SIGNED bracket ``[-0.5,+0.5)`` instead of
collapsing it: picks -0.5 vs +0.5 with probability proportional to where the
signed ``v`` falls across the doubled range (``p_pos = (v+0.5)/1.0``) -- near
``v=0`` this is close to a fair coin flip between signs; near +0.5 almost
always +0.5; near -0.5 almost always -0.5. Every bracket further from zero is
untouched and stays exactly as unbiased as ``fp4_quantize_stochastic``'s own.

.. _fp4_quantize_stochastic_live_nonneg.sign_never_flipped_bug:

Nonneg live quantize: why importance can never use the cross-sign redirect
---------------------------------------------------------------------------------

*ID:* ``fp4_quantize_stochastic_live_nonneg.sign_never_flipped_bug``

``fp4_quantize_stochastic_live_nonneg()`` exists for a LIVE synapse's
IMPORTANCE (or any other quantity that is mathematically always >= 0, e.g.
the ``ci`` accumulator fed into ``sqrt(ci)+eps`` damping throughout
``disldo_backward``/``sisldo_ops.hpp``).

**Real bug, found via direct question/regression during this session**:
``fp4_quantize_stochastic_live``'s cross-sign redirect (see
``fp4_quantize_stochastic_live.probability_weighted_redirect`` above) is
correct for WEIGHT, which can legitimately be positive or negative -- but
applying that SAME redirect to importance is a bug: importance near 0 would
then have up to a 50% chance of landing on the NEGATIVE code, and a negative
decoded importance makes every downstream ``sqrt(ci)`` call return NaN,
silently poisoning training.

Fix: sign is NEVER flipped in the nonneg variant (matches
``fp4_encode_bits_live``'s own deterministic behavior, which never had this
bug). In the ``[0,0.5)`` bracket there is no legal nonzero magnitude to
interpolate toward besides 0.5 itself (0 is forbidden, negative is
forbidden), so that bracket is deterministic: always ``mag_code=1`` -- the
same deliberate near-zero bias already accepted for the live design as a
whole (see ``fp4_quantize_live.never_zero_rationale`` above), just without
weight's cross-sign freedom that a nonnegative quantity must not have. Every
bracket above ``HALF_BITS`` is untouched, identical to
``fp4_quantize_stochastic_live``'s own (those never touch sign or produce
``mag_code 0`` anyway).

.. code-block:: cpp

   // as of PR #45, fp4quant.hpp -- the WRONG behavior for importance
   // (fp4_quantize_stochastic_live's cross-sign redirect, near v=0):
   const float signed_v = sign ? -av : av;
   const float p_pos = signed_v + 0.5f;
   const bool pick_pos = fp4_stochastic_uniform01() < p_pos;
   mag_code = 1u;
   s_out = pick_pos ? 0u : 1u;   // up to 50% chance of a NEGATIVE code --
                                 // poisons every downstream sqrt(ci) with NaN

   // fp4_quantize_stochastic_live_nonneg's fix -- sign is never touched:
   } else if (abits < HALF_BITS) {
       mag_code = 1u; // deterministic, sign stays whatever it was (always +)
