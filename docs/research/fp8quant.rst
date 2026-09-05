``fp8quant.hpp`` research notes
==================================

Companion doc to ``sili/lib/headers/fp8quant.hpp``. Source comments point
back here by anchor ID (``*ID:*`` marker under each heading below); this doc
links back to source by function name. Same pattern as
``docs/research/fp4quant.rst`` (semantic dotted anchor IDs, visible ID
markers, frozen code snippets on real-bug/non-obvious-derivation sections) --
fp8quant.hpp is FP4's E4M3 sibling, same codec/stochastic-rounding/
never-zero design space, different bit format, so several sections below
point back to fp4quant.rst instead of re-deriving shared reasoning.

.. _fp8_format.e4m3_design:

E4M3 format: OCP MX spec, repurposed NaN slot, and validation
------------------------------------------------------------------

*ID:* ``fp8_format.e4m3_design``

1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits -- matches the OCP
Microscaling FP8 spec / NVIDIA Transformer Engine's E4M3, not a from-scratch
format. NOT plain IEEE-754, though: like this codebase's own FP4 (E2M1 with
one repurposed NaN slot), the exponent field ``1111`` is NOT entirely
reserved for Inf/NaN. Only the single code ``S.1111.111`` is NaN; every other
``e=1111`` code is still a valid finite normal value, which extends the max
representable magnitude to 448.0 instead of IEEE's would-be smaller normal
ceiling. Subnormals live at ``e=0000``: value = ``m/8 * 2^-6`` = ``m * 2^-9``
(min positive normal is ``2^-6``, subnormal step is ``2^-9``).

Validated in Python (sili_peridot's toy-model quantization sweep, July-Aug
2026 session) with a per-row/rank-1 (row*col envelope) scale on top of this
raw code -- exactly the same mechanism as FP4's existing value_scale/
output_scale. Beat both a plain-int8 rank-1 scheme AND real FP4 across every
task family tested (single-cell tanh-RNN, deep causal-attention transformer,
tile-recurrence). Full writeup in sili_peridot's JOURNAL.md.

Same bit-shift/carry-propagation codec technique as ``fp4_encode_bits``/
``fp4_decode_bits`` (see ``fp4quant.rst``: ``fp4_codec.bitshift_design`` for
the full derivation of that trick) -- no table, no per-candidate branch. One
full byte per value means no nibble-packing/masking machinery is needed at
all here, simpler than FP4's ``ElemRef``/``Lane`` machinery.

.. code-block:: cpp

   // as of PR #45, fp8quant.hpp -- the repurposed-NaN-slot constants:
   constexpr uint32_t FP8_MAX_BITS = 0x43E00000u;      // bits_of(448.0f)
   constexpr uint32_t FP8_NAN_SLOT_BITS = 0x43F00000u; // bits_of(480.0f) in
                                                        // float32 terms --
                                                        // the reserved code

.. _fp8_encode_bits.subnormal_normal_boundary:

Subnormal encode: carrying across the subnormal/normal boundary
------------------------------------------------------------------

*ID:* ``fp8_encode_bits.subnormal_normal_boundary``

The subnormal region (``abits < FP8_MIN_NORMAL_BITS``) is nearest-of
``{0, 2^-9, ..., 7*2^-9}``. Each sub-bracket is linear in ``v`` directly (no
shared exponent to exploit for a bit trick), so a plain round suffices --
``m = round(v * 512)`` -- same reasoning ``fp4_encode_bits`` uses for its own
``<1.0`` region.

**Real edge case, distinct from FP4's own subnormal handling**: rounding can
carry PAST the largest subnormal (``7*2^-9 = 0.013672``) into the first
normal code (``2^-6 = 0.015625``, ``e4=1,m=0``) -- e.g. ``0.015367`` is
numerically nearer to ``0.015625`` (min normal) than to ``0.013672`` (max
subnormal). Clamping the rounded ``m`` to 7 in that situation would silently
pick the wrong neighbour instead of carrying across the subnormal/normal
boundary. In the normal-range branch, integer addition's carry propagates
this kind of boundary-crossing rounding automatically (the whole point of the
carry-propagation trick, see ``fp4_codec.bitshift_design``) -- but the
subnormal branch here computes ``m`` via a float multiply+round rather than
integer carry, so it can't fall out "for free" and needs an explicit
overflow check instead.

.. code-block:: cpp

   // as of PR #45, fp8quant.hpp -- fp8_encode_bits' subnormal carry fix:
   const uint32_t m = uint32_t(av * 512.0f + 0.5f);
   if (m > 7u) {
       // Rounding carried past the largest subnormal into the first
       // normal code (2^-6 = min normal, e4=1, m=0).
       return uint8_t((s << 7) | 0x08u);
   }
   return uint8_t((s << 7) | m);

.. _fp8_quantize_live.dual_zero_rationale:

Live variants: E4M3 has distinct +0 and -0, both must be avoided
------------------------------------------------------------------

*ID:* ``fp8_quantize_live.dual_zero_rationale``

Same never-zero invariant as ``fp4_quantize_live`` -- full rationale in
``fp4quant.rst``: ``fp4_quantize_live.never_zero_rationale`` (a live
synapse's weight quantizing to the block4/delta_csr "blank slot" sentinel
makes ``disldo_backward`` treat the whole row/tile as having zero live
connections, recoverable only via a much slower bootstrap fallback).

**New subtlety vs FP4**: ``FP4_TABLE`` has a single shared zero code (index
0) regardless of sign, so ``fp4_quantize_live`` only had to avoid one value.
E4M3 is a real floating-point format with DISTINCT ``+0`` (byte ``0x00``, the
actual block4/scattered blank-slot sentinel) and ``-0`` (byte ``0x80``,
decodes to the same ``0.0f`` value but is a different byte). Both must become
unreachable for a live weight, not just ``0x00`` -- a synapse could otherwise
silently collapse to a zero CONTRIBUTION via ``-0`` even though it wouldn't
trip the byte``==``0 liveness check (which only tests the raw byte, not the
decoded float).

Fix, in ``fp8_encode_bits_live``'s subnormal branch: if the rounded magnitude
``m`` comes out 0, redirect to ``m=1`` (the smallest nonzero magnitude,
``2^-9``) before applying the sign bit -- covers both ``+0`` and ``-0`` since
the redirect happens before sign is OR'd in.

.. _fp8_quantize_stochastic.dithered_rounding:

Stochastic quantize: same dithered-rounding technique as FP4, wider dither
--------------------------------------------------------------------------------

*ID:* ``fp8_quantize_stochastic.dithered_rounding``

Unbiased (``E[result] == v`` within ``[-448,448]``; clamps deterministically
outside it), same dithered-rounding technique as ``fp4_quantize_stochastic``
-- full derivation of why this is exact in the normal region is in
``fp4quant.rst``: ``fp4_quantize_stochastic.dithered_rounding_design``. A
uniform random dither spanning the discarded mantissa bits is added before
truncation; integer carry propagates a probabilistic round exactly the way
the deterministic version's fixed bias propagates a round-to-nearest.

Shares the caller's thread-local RNG state with FP4
(``fp4_stochastic_next_u64``/``fp4_stochastic_uniform01``, see
``fp4quant.rst``: ``fp4_stochastic_rng.thread_local_design``) -- one PRNG per
thread, not duplicated per format.

**Wider dither range than FP4**: E4M3 keeps 3 mantissa bits (discards 20 of
float32's 23 mantissa bits), vs FP4's 1 kept bit (discards 22) -- so the
dither mask here is ``0xFFFFF`` (spanning ``2^20``) vs FP4's ``0x3FFFFF``
(spanning ``2^22``).

Gradient-driven update sites only, same convention as FP4's
``set_stochastic`` -- construction/loading/compact must stay deterministic.

.. _fp8_quantize_stochastic_live.subnormal_signed_redirect:

Live stochastic quantize: widening only the sub-bracket that could hit +-0
--------------------------------------------------------------------------------

*ID:* ``fp8_quantize_stochastic_live.subnormal_signed_redirect``

Mirrors ``fp4_quantize_stochastic_live``'s probability-weighted redirect
(``fp4quant.rst``: ``fp4_quantize_stochastic_live.probability_weighted_redirect``),
but narrower in scope. FP4's redirect widens its ENTIRE ``[0,0.5)`` bracket
to a full signed choice, because that whole bracket is the only place a
result could land on the shared zero code. FP8's subnormal region instead has
8 distinct sub-brackets (``m_lo = 0..7``), and only the ``m_lo==0`` bracket
can produce ``+0`` or ``-0`` -- so only that one sub-bracket is widened, to a
full-signed-bracket probability-weighted choice between ``+2^-9`` and
``-2^-9`` (``p_pos = (v + 2^-9) / (2 * 2^-9)``). Every other sub-bracket
(``m_lo`` 1-7, and the entire normal region) is untouched and stays exactly
as unbiased as ``fp8_quantize_stochastic``'s own.

.. _fp8_quantize_stochastic_live_nonneg.sign_never_flipped:

Nonneg live quantize: mirroring FP4's importance sign-flip bug fix
------------------------------------------------------------------------

*ID:* ``fp8_quantize_stochastic_live_nonneg.sign_never_flipped``

Same bug, same fix as ``fp4_quantize_stochastic_live_nonneg`` -- full
writeup in ``fp4quant.rst``:
``fp4_quantize_stochastic_live_nonneg.sign_never_flipped_bug``.
``fp8_quantize_stochastic_live``'s ``m_lo==0`` cross-sign redirect is correct
for weight (legitimately signed) but wrong for importance: importance is
mathematically always ``>= 0`` and feeds ``sqrt(ci)+eps`` damping throughout
``disldo_backward``/``sisldo_ops.hpp``, and a negative decoded importance
makes every downstream ``sqrt(ci)`` call NaN.

Fix: sign is NEVER flipped in the nonneg variant. The ``m_lo==0`` sub-bracket
is deterministic (always ``m=1``, matching ``fp8_encode_bits_live``'s own
behavior) since there's no legal nonzero code below ``2^-9`` to interpolate
toward besides ``2^-9`` itself. Every other sub-bracket (``m_lo`` 1-7, and the
whole normal region) is untouched, identical to
``fp8_quantize_stochastic_live``'s own.

.. _fp8_bivalues.two_array_shape:

FP8BiValues: plain two-array storage, not nibble-packed
------------------------------------------------------------

*ID:* ``fp8_bivalues.two_array_shape``

Unlike ``FP4BiPacked``'s nibble-interleaved single byte array (two 4-bit
values packed per byte -- necessary because an FP4 code only needs 4 bits),
``FP8BiValues`` is two plain ``std::vector<uint8_t>`` arrays (``weights``,
``importance``). E4M3 already needs a whole byte per value, so there is
nothing left to pack, and no bit-spanning/masking machinery is needed.

Mirrors ``DeltaCSRBiValues<T>``'s plain two-array shape exactly, per direct
instruction ("the sparse part can take pretty close to the fp32 version") --
same structure, just narrower per-element storage than a float32 array.
``ValueAccessor<FP8BiValues>`` (``delta_csr_types.hpp``, alongside
``ValueAccessor<FP4BiPacked>``/``ValueAccessor<DeltaCSRBiValues<T>>``) is what
makes this a drop-in ``VALUES_TYPE`` for
``SparseLinearWeightsDelta``/``disldo_forward``/``disldo_backward``.
