``delta_csr_types.hpp`` research notes
=========================================

Companion doc to ``sili/lib/headers/delta_csr_types.hpp``. Source comments
point back here by anchor ID (``*ID:*`` marker under each heading below);
this doc links back to source by file/symbol name. See
``docs/research/linear_disldo.rst`` for the pattern this follows (semantic
dotted anchor IDs, visible ID markers, frozen code snippets on real-bug/
non-obvious-derivation sections).

Split out of ``sparse_struct.hpp`` to keep files under ~1k lines: core type
definitions only (``sparse_struct`` template, ``ValueAccessor<FP4BiPacked>``/
``ValueAccessor<FP8BiValues>``/``ValueAccessor<DeltaCSRBiValues<T>>``,
``DeltaCSRLayout``/``DeltaCSRRowCursor``/``DeltaCSRWeights``,
``SparseLinearWeightsDelta``). Free functions operating on these types live
in ``delta_csr_memory.hpp`` and ``sisldo_ops.hpp``; ``sparse_struct.hpp``
remains a valid umbrella include of all three. ``block4.hpp`` is included
partway through the file (right before ``SparseLinearWeightsDelta``, the
first thing that needs ``Block4Store``'s full definition) rather than at the
top, because ``Block4Store`` itself now reuses ``DeltaCSRLayout``/
``DeltaCSRRowCursor`` (defined earlier in this file) -- so it can't be
included any earlier without a circular dependency.

.. _value_accessor.live_variants:

``ValueAccessor``: ``set``/``set_stochastic``/``set_live``/``set_stochastic_live``
-------------------------------------------------------------------------------------

*ID:* ``value_accessor.live_variants``

Each storage type (``FP4BiPacked``, ``FP8BiValues``, the float32 fallback
``DeltaCSRBiValues<T>``) exposes four setters, not one, because two
independent axes matter for how a synapse's stored code is written:

- **Gradient-driven vs. reparametrization**: ``set_stochastic`` is for a
  real optimizer step (stochastic rounding reduces the systematic bias a
  deterministic round would introduce over many small steps); plain
  ``set`` is for a reparametrization that doesn't correspond to a gradient
  step at all (e.g. rescaling importance to a new per-row scale).
- **Live vs. not-yet-connected**: the ``_live`` variants exist because
  FP4/FP8's code-0 slot doubles as "no synapse here" for block4's dense
  tiles (see ``disldo_backward.was_live_gating`` in
  ``docs/research/linear_disldo.rst``) -- a genuinely live synapse must
  never be allowed to quantize back down to that same sentinel code, or it
  silently un-registers itself from block4's own sparse/dense bookkeeping.
  ``fp8_quantize_stochastic_live``'s importance path specifically uses the
  ``_nonneg`` variant (not the general live codec): importance is always
  >= 0 and feeds directly into ``sqrt(ci)``, so the general live codec's
  cross-sign "redirect near zero" behavior would NaN it.

The float32 fallback (``DeltaCSRBiValues<T>``) has no quantization at all,
so both axes collapse to a no-op passthrough to ``set()`` -- an exact 0.0f
float is a legitimate stored value there, not a byte-0 sentinel, so the
never-0 live-quantize invariant is meaningless for that storage. This keeps
callers that always call ``set_stochastic``/``set_live`` uniformly correct
regardless of which ``VALUES_TYPE`` they're templated on.

.. _amortized_decay.chunked_cursor:

Amortized decoupled decay + running stats
--------------------------------------------

*ID:* ``amortized_decay.chunked_cursor``

``apply_amortized_decay_stats`` touches only ``chunk_size`` synapses per
call via a persistent rolling cursor supplied by the caller -- it never
scans a whole layer in one call, the same amortization shape as
``synap_row_step``'s own per-row cursor. ``decay_factor`` is the caller's
job to derive (Python side): ``2^(-cycle_length/H)`` for a chosen half-life
``H`` in real training steps, where ``cycle_length = ceil(nnz/chunk_size)``,
since a given synapse is only actually touched once per ``cycle_length``
steps.

Decays the WEIGHT only, via ``get_imp``+``set_live`` (importance is a
significance signal, not a magnitude to shrink), preserving each
``VALUES_TYPE``'s own never-0 live-quantize invariant where one exists
(FP4/FP8) and passing through unchanged where it's meaningless (fp32). For
FP4/FP8 this decay is lower marginal value than for fp32: raw codes are
already magnitude-bounded by a small fixed decode table and can't blow up
the way a plain float can -- the real FP4/FP8 blow-up path is the SCALE
vector (covered by the scale policies' own NaN/Inf guards below), not the
per-synapse code. The running stats remain useful regardless, for uniform
health monitoring across all three storage types.

Stats fields (``mean_abs``/``rms``/``max_abs``/``n``) are only meaningful
for a just-FINISHED cycle -- accumulators reset after being read via the
caller's own ``sum_abs``/``sum_sq``/``max_abs``/``n`` references.
``cycle_complete=false`` means "still mid-cycle": this function zeroes
nothing in that case, but callers should gate on ``cycle_complete`` rather
than reading the numeric fields, which are stale/undefined until it flips
true.

.. _scale_policy.nan_inf_guard:

Scale-update policies: why every one guards against NaN/Inf
----------------------------------------------------------------

*ID:* ``scale_policy.nan_inf_guard``

``RMSpropScalePolicy``/``AdaMaxScalePolicy``/``AdamScalePolicy`` are
swappable in-place optimizers for ``value_scale``/``output_scale``
(``disldo_backward``'s scattered path, ``linear_disldo.hpp``). Template
parameter, not a runtime flag: each policy is a stateless struct with one
static ``update()``, so choosing a policy costs nothing at runtime (inlined,
no branch, no vtable), matching this codebase's existing ``VALUES_TYPE``
convention of separate hand-instantiated callers over one shared templated
implementation rather than one class runtime-branching on a stored enum.

Built to compare real update rules for ``value_scale``/``output_scale``
against a toy Python fake-quantize simulation's closed-form full-layer
refit (``sili_peridot``'s ``QuantizedDISLDOLayer32``/
``rank1_fake_quantize``) that reaches near-perfect accuracy on an
out-of-context recall task while the real RMSprop-based scale update
collapses -- see ``sili_peridot/JOURNAL.md``'s 2026-08-09 tile-recurrence
entries for the full investigation.

**Real bug, confirmed via direct diagnostic** (dense connectivity,
``sili_peridot/JOURNAL.md`` 2026-08-10): ``scale``/``scale_state`` update
INLINE every step and are never touched by any external gradient clip
(unlike an optimizer-managed parameter). A single overflowed ``g_agg``
(e.g. dense connectivity's much larger fan-in summing far more per-column
gradient terms than sparse ever does, narrowed from a double accumulator
to ``VALUE_TYPE`` with no range check upstream) turns ``scale_state``
permanently Inf (``beta2*Inf`` never decays back down), then
``sqrt(Inf)+eps -> Inf``, ``g_agg/Inf -> Inf`` or ``Inf/Inf -> NaN``, and
once ``scale`` is NaN every future ``beta2*NaN+...`` stays NaN forever.
``output_scale`` went NaN in lockstep with the whole model while the raw
stored weight code itself stayed correctly bounded. Fix: skip the update
entirely on a non-finite input/result rather than letting it corrupt
``scale``/``scale_state`` -- makes NaN structurally unreachable through
this path, not just unlikely. Every scale policy in this file (including
the ``Block4Vec`` SIMD synapse-policy specializations further down) carries
this same guard for the same reason.

.. code-block:: cpp

   // as of PR #45, delta_csr_types.hpp -- RMSpropScalePolicy::update:
   if (!std::isfinite(g_agg) || !std::isfinite(contrib_agg))
       return;
   // ... later, before committing the result:
   if (!std::isfinite(new_state)) return;
   if (!std::isfinite(state_hat)) return;
   if (!std::isfinite(new_scale)) return;

.. _scale_policy.contrib_agg_combination:

``contrib_agg``: combining gradient and forward-contribution signals
-------------------------------------------------------------------------

*ID:* ``scale_policy.contrib_agg_combination``

``contrib_agg`` is the row/column-aggregated forward-contribution signal
(sum of ``x*w``, mirroring per-synapse ``ci``'s own ``contrib = x*w`` term
-- see ``linear_disldo.hpp``'s additive combination and the Lean proof
``Joint.combined_signal_strictly_informative`` in
``lean_proofs/importance_signal_information_gain/SiliImportanceProof/
ImportanceSignalInformationGain.lean``). ``value_scale_importance``/
``output_scale_importance`` are the SAME kind of RMSprop second-moment
accumulator as ``ci``, just aggregated over a row/column instead of a
single synapse -- combined the same way ``ci`` does: SQUARE first, THEN
sum (``g_agg^2+contrib_agg^2``), not ``(g_agg+contrib_agg)^2``. This value
is the DIVISOR of the update step, so its job is safety, not just
importance-ranking: sum-then-square lets a large-magnitude disagreement
between ``g_agg`` and ``contrib_agg`` collapse the denominator toward zero
even though both signals are individually large, exploding the step --
square-then-sum is bounded below by ``max(g_agg,contrib_agg)^2``
regardless of sign, so a large disagreement damps the step instead of
amplifying it. The actual STEP still uses ``g_agg`` alone (unbiased:
``E[step]=0`` under zero-mean noise, only the magnitude ESTIMATE gets the
extra signal). Defaults to 0 for callers that don't have one, reproducing
plain RMSprop exactly.

.. _scale_policy.adam_bias_correction:

``step``: Adam-style bias correction, and the sign-flip bug it fixes
--------------------------------------------------------------------------

*ID:* ``scale_policy.adam_bias_correction``

``scale_state`` starts at 0, so on step 1 it's ``(1-beta2)*(g_agg^2 +
contrib_agg^2)`` -- badly SHRUNK toward zero, not the true magnitude
itself (Kingma & Ba 2015, sec 3). Dividing by ``(1-beta2^step)`` undoes
exactly that shrinkage: on step 1, ``state_hat = new_state/(1-beta2) =
g_agg^2+contrib_agg^2`` exactly, so the step size becomes normal-sized
instead of ~1/sqrt(1-beta2) (~31.6x at the default ``beta2=0.999``) too
large. This is purely an optimizer-internal correction, unrelated to a
model's own ``+b`` bias term.

**Real bug this fixes**: ``value_scale`` swinging sign in a single first
update (1.0 -> -3.1, lr=0.5), corrupting every synapse sharing that row
(both scattered and block4-owned, since they share the same
``value_scale[row]``) -- see ``test_disldo_block4_backward.cpp``'s
regression test. Applied ONLY to ``value_scale_importance``/
``output_scale_importance`` (row/column-level, one ``uint32_t`` counter
each -- cheap), NOT to per-synapse ``ci`` (which would need a counter the
same size as ``ci`` itself, doubling memory for an FP4/FP8 format where
every byte counts, for a self-limiting problem that doesn't compound
across a whole row the way ``value_scale``'s does).

.. _scale_policy.log_space_variant:

``log_space``: scale-invariant update for magnitude reparametrization
----------------------------------------------------------------------

*ID:* ``scale_policy.log_space_variant``

``log_space=false`` (default) uses a fixed-size additive ``eff_lr`` step,
which assumes scale stays near 1.0 -- a huge RELATIVE change once scale
has shrunk far below 1 (which magnitude-scale reparametrization
deliberately does) and negligible once scale has grown large. This mirrors
``update_cw``'s own ``scale_invariant`` fix (``PlainRMSpropSynapsePolicy``/
``BoundedRMSpropSynapsePolicy`` below), just applied to scale's OWN update
instead of the per-synapse weight update: ``d(loss)/d(log(scale)) =
d(loss)/d(scale)*scale = g_agg*scale`` (chain rule through
``scale=exp(log_scale)``); RMSprop-normalizing THAT keeps the step a fixed
RELATIVE (percentage) size regardless of scale's own magnitude, and scale
can never cross zero (``exp()>0``), unlike the additive step. Ported
verbatim from ``sili_peridot``'s torch prototype
(``toy_tile_recurrence_rmt_torch.py``'s ``_scale_update``,
``scale_invariant_chain_rule`` branch) -- see that module for the
validated-in-torch derivation this is a direct port of.

.. _scale_policy.adamax_design:

``AdaMaxScalePolicy``: running-max second moment, no sqrt
----------------------------------------------------------

*ID:* ``scale_policy.adamax_design``

Matches AdaMax's own decayed running-max second-moment tracker (Kingma &
Ba 2015, sec 7): ``scale_state`` tracks ``max(beta2*scale_state,
|g_agg|)`` -- growth is INSTANT (never lets ``scale_state`` fall below the
current gradient magnitude, matching the max-cover safety property a real
fixed-point scale needs: never let a stored value exceed what its levels
can represent), shrink is gradual (only the decay term reduces it, when
nothing larger has been seen recently). No sqrt needed -- ``scale_state``
is already in the same units as ``|g_agg|``, unlike RMSprop's ``g^2`` EMA.

``contrib_agg`` combines via ``max(|g_agg|,|contrib_agg|)``, not
``|g_agg+contrib_agg|`` -- same safety rationale as
``scale_policy.contrib_agg_combination`` above, but AdaMax's own state IS
an L-infinity (max) norm tracker already (that's what distinguishes it
from Adam's L2/RMSprop), so combining two signals via max is the same
combine rule the policy already uses for combining across TIME, just
applied across the two SIGNALS too. Summing before taking the magnitude
would have the identical cancellation hole as sum-then-square did for
RMSprop.

``step`` is accepted only for call-site signature compatibility with
``RMSpropScalePolicy::update`` -- unused here. Per Kingma & Ba 2015 sec 7,
AdaMax's running-MAX state doesn't have RMSprop's EMA cold-start
shrinkage problem (``max(0, combined_mag)`` on step 1 is already the true
value, not a shrunk fraction of it), so there's nothing for bias
correction to fix.

.. _scale_policy.adam_additive_branch:

``AdamScalePolicy``: real Adam for the AQRS additive branch
--------------------------------------------------------------

*ID:* ``scale_policy.adam_additive_branch``

AQRS additive branch's default optimizer (task #277, see
``sili_peridot/AQRS_DESIGN.md``): RMSprop's second-moment tracking (same
formula as ``RMSpropScalePolicy::update``) PLUS a genuine first-moment
(momentum) EMA, used in the step's NUMERATOR instead of the raw gradient.
Per direct instruction: default for the additive branch's own parameters
because Adam trains faster and more stably than RMSprop for a small
parameter count, while RMSprop stays available as an explicit alternative
(``RMSpropScalePolicy`` itself, unchanged) -- callers select between them
via the same template-parameter pattern already used everywhere else in
this file, not a runtime branch.

NOT a drop-in replacement for ``RMSpropScalePolicy::update``'s existing
call sites -- different signature (needs a SEPARATE ``momentum_state``
reference alongside ``scale_state``, since Adam genuinely needs two
independent EMAs, not one), hence a new sibling struct rather than an
extra parameter bolted onto the existing function. Can't simply call
``RMSpropScalePolicy::update`` internally either: Adam's second moment
must be computed from the RAW gradient, but the step's numerator must use
the momentum-SMOOTHED gradient -- ``RMSpropScalePolicy::update`` uses the
SAME ``g_agg`` value for both, so passing it either raw ``g`` or
momentum-smoothed ``g`` would get one of the two uses wrong. The
bias-correction PATTERN (not the full update) is intentionally similar to
``RMSpropScalePolicy::update``'s own inline version -- a small, deliberate
duplication of that specific ~4-line snippet (not refactored into one
shared helper, to avoid touching a working, tested function) -- everything
else (the first-moment EMA and combining both moments in the final step)
is genuinely new.

.. _scale_policy.no_scale_policy:

``NoScalePolicy``: a real ablation, not a stub
------------------------------------------------

*ID:* ``scale_policy.no_scale_policy``

``scale``/``scale_state`` are never touched, so ``scale`` stays at
whatever it was initialized to (``value_type(1)`` by default --
``SparseLinearWeightsDelta``'s ``value_scale.resize(n, value_type(1))``) --
i.e. ``value_scale[row]*output_scale[col]`` is permanently the identity
multiply, ``true_w == stored_w``. This exists as a direct real-hardware
test of the "zero trained scale" hypothesis (``sili_peridot``'s
``fixed_digit_residual_quantize``/``TrueMultiDigitLayer`` work) instead of
just fixing staleness elsewhere -- usable with ``DeferredScaleWrite`` true
or false, since there's nothing to defer either way.

.. _synapse_policy.overview:

Per-synapse ``ci``-update policy family: why it's templated at all
------------------------------------------------------------------------

*ID:* ``synapse_policy.overview``

Distinct from the ``ScalePolicy`` family above: those operate on
``value_scale``/``output_scale`` (one scalar per ROW/COLUMN). This family
operates on per-synapse ``ci`` (one scalar per SYNAPSE,
``linear_disldo.hpp``'s own RMSprop second-moment accumulator) --
historically hand-duplicated at ~8 call sites (6 scalar + 2 SIMD) rather
than templated. Any future change to this formula should go through ONE
template parameter, not shotgun surgery across every site again -- an
earlier revert already missed 2 SIMD sites once, caught only by
``test_block4_scattered_divergence.cpp``.

**Root cause this family exists to fix**: a plain RMSprop ``ci`` EMA
(``beta2=0.999``, no bias correction -- unlike ``value_scale_importance``/
``output_scale_importance``, per-synapse ``ci`` does not get Adam-style
bias correction) LAGS the true local gradient scale near a converged
solution. That lag is what keeps the step naturally small during a long
stable plateau (``ci`` stays elevated relative to the now-tiny residual
gradient). But ``ci`` eventually decays down to match the small residual
too -- once it does, ``g/sqrt(ci)`` stops shrinking with the error and
returns to ~full-lr-sized steps regardless of how small the actual error
is, causing overshoot, a larger resulting gradient, ``ci`` ratcheting back
up, and the cycle repeating/compounding. Confirmed directly: a
``DISLDOLayer32`` permutation-regression run stayed rock-stable at
``SSE=3.0000`` for 300+ steps then diverged past ``SSE=180`` by step
~405, at a CONSTANT lr, with ``ci`` visibly decaying from ~0.9 to ~0.66
right before the spike.

An lr-decay schedule (any monotonic-to-zero form, including
``1/(1+step)``) "fixes" this by eventually freezing the whole network --
incompatible with this project's lifelong-learning goal. Both policies
below are STATIONARY (no dependency on step count / wall clock), so they
stay compatible with an infinite training horizon.

Exactly two entry points per policy, each owning its full formula
end-to-end -- no math left inline at the call site, no third "combine"
wrapper. An earlier draft split the RMSprop division/damp-branch out into
the call site while only the floor and clip lived in the policy --
half-templated, still shotgun-surgery-prone for the actual optimizer math,
the opposite of the point. ``update_ci`` is the complete per-synapse
second-moment update; ``update_cw`` is the complete per-synapse weight
delta (the ``damp_by_importance`` branch, the division, and, for Bounded
only, a hard cap on ``|delta|`` independent of ``ci`` entirely). ``S`` is
the combined ``value_scale*out_scale`` factor some call sites multiply
into the delta (pass ``VALUE_TYPE(1)`` for sites that don't scale) -- note
``ci`` itself never sees ``S``, only the delta does, matching every
existing call site's own convention.

.. _synapse_policy.plain_reference:

``PlainRMSpropSynapsePolicy``: the bit-identical reference
---------------------------------------------------------------

*ID:* ``synapse_policy.plain_reference``

Reproduces ``linear_disldo.hpp``'s original (pre-fix) inline formula
exactly, bit-for-bit, on finite inputs -- kept for explicit opt-in and as
the reference the ``BoundedRMSpropSynapsePolicy`` fix was checked against,
but no longer ``disldo_backward``'s own default (see
``synapse_policy.bounded_beats_plain`` below for why). No floor, no clip,
same NaN/Inf guard convention as the scale policies above: a non-finite
``g``/``contrib`` (e.g. from an upstream weight that's already diverging)
must not be allowed to corrupt ``ci`` -- ``ci`` has no other bound in this
policy, so a stray NaN here is permanent. Skip the update (return the OLD
``ci`` unchanged) rather than writing NaN -- this closes a coverage gap an
earlier commit (``ba4af42``) left: that commit guarded ``value_scale``/
``output_scale``'s own update but never extended the same guard to the
per-synapse ``ci``/``cw`` path.

``update_cw`` returns a DELTA (caller does ``cw += update_cw(...)``), so
"skip the update" means return 0 (a true no-op delta), matching
``update_ci``'s "keep old value" semantics.

.. _synapse_policy.scale_invariant_quadratic_bug:

``scale_invariant``: fixing a quadratic-in-S scaling bug
-------------------------------------------------------------

*ID:* ``synapse_policy.scale_invariant_quadratic_bug``

Default ``false`` (bit-identical to every existing result). ``ci`` is
calibrated to the RAW gradient ``g^2``, unaffected by ``S`` -- but the
historical formula folds ``S`` into the numerator anyway, so ``ci`` isn't
self-normalized w.r.t. ``S``, and ``Delta(true_weight) = S*Delta(cw)`` ends
up scaling QUADRATICALLY with ``S`` once ``S`` deviates from ~1.0.

**Real bug found this way**: fp32 accuracy went ``1.0 -> 0.18`` once a
mechanism deliberately moved ``S`` away from 1.0, despite ``true_weight =
cw*S`` being algebraically unchanged by that move.

Fix: ``scale_invariant=true`` computes ``raw`` from the RAW ``g`` (properly
self-normalized by ``ci``) and divides by ``S`` once at the very end
instead, giving ``Delta(true_weight) = eff_lr*raw``, independent of ``S``.
``BoundedRMSpropSynapsePolicy::update_cw`` carries the identical fix, with
the added interaction of applying its ``max_abs_delta`` clip to the
already-S-normalized ``raw`` (see
``synapse_policy.clip_order_and_lr_ceiling`` below).

.. code-block:: cpp

   // as of PR #45, delta_csr_types.hpp -- PlainRMSpropSynapsePolicy::update_cw:
   if (scale_invariant) {
       const VALUE_TYPE raw = damp_by_importance ? (-g) / (std::sqrt(ci) + eps) : (-g);
       delta = std::isfinite(raw) ? (eff_lr * raw / S) : VALUE_TYPE(0);
   } else {
       // WRONG shape when S drifts far from 1.0 -- S folded into the
       // numerator makes Delta(true_weight) scale as S^2, not S:
       delta = damp_by_importance ? (-eff_lr * g * S) / (std::sqrt(ci) + eps)
                                   : (-eff_lr * g * S);
   }

.. _synapse_policy.bounded_beats_plain:

``BoundedRMSpropSynapsePolicy``: min_decay_frac and max_ci
------------------------------------------------------------

*ID:* ``synapse_policy.bounded_beats_plain``

``min_decay_frac``: never lets ``ci`` decay below
``min_decay_frac * ci_old`` in a single step, regardless of how small the
current ``(g,contrib)`` is. Plain EMA's own WORST-CASE per-step retention
(``g=contrib=0``) is already exactly ``beta2`` (``ci_new =
beta2*ci_old``), so ``min_decay_frac`` only has any effect when it's
strictly greater than ``beta2``; ``min_decay_frac <= beta2`` is a silent
no-op. ``min_decay_frac=1.0`` would freeze ``ci`` forever (AMSGrad-style,
rejected: would permanently suppress step size for any synapse that ever
saw one large gradient, the opposite of what a lifelong learner needs, and
would likely compound this project's own stuck-weights findings rather
than fix them).

**Tuning result** (``tests/unit/sweep_synapse_policy_min_decay_frac.cpp``):
CHOSEN PRODUCTION DEFAULT leaves ``min_decay_frac`` at its own true-no-op
value (``<=beta2``), not a value strictly between ``beta2`` and 1. Tested
up to ``min_decay_frac=0.99995`` with the delta clip disabled entirely and
found ZERO measurable protective effect against the late-training
resonance this policy exists to fix -- ``max_abs_delta``'s hard clip is
doing the ENTIRE protective job on its own, under deterministic rounding.
Under STOCHASTIC rounding (the actual production mode --
``tests/unit/sweep_synapse_policy_stochastic.cpp``), ``min_decay_frac`` is
NOT provably inert: it showed a real, single-seed/unconfirmed benefit at
the riskier end of the ``max_abs_delta`` range (e.g. ``max_abs_delta=16``:
0.9995 measurably beat 0.999), but that benefit vanishes by ``lr~0.2`` and
is bit-identical between tested values by ``lr=1.0`` -- it does NOT extend
the safe lr range, so it doesn't change the production default (production
operates at ``lr<=0.05``, where the clip alone is already deep-safe
regardless).

``max_ci``: hard ceiling on ``ci`` itself -- ``min_decay_frac``'s floor
only slows how fast ``ci`` can DECAY, it does nothing to stop ``ci`` from
GROWING. In the unsafe-pocket failure mode
(``probe_unstable_pocket_growth.cpp``), ``ci`` was directly measured
climbing continuously and unboundedly the entire 30000-step run (0.0005 ->
163+, still setting a new max at literally every checkpoint) -- ``ci``
chases ``g^2`` upward with no cap as long as the underlying divergence
keeps the gradient growing. Healthy production-default operation plateaus
at ``ci~0.5``, so a ceiling with generous margin above that costs nothing
in the normal regime. Default ``max_ci=1e30`` is a true no-op; the CHOSEN
PRODUCTION DEFAULT is ``max_ci=100.0`` (set in ``cpu_backend.cpp``'s 3 real
call sites), verified in ``tests/unit/test_ci_ceiling.cpp``: ``ci`` never
exceeds 100.0 across the full 30000-step unsafe-pocket run (15.36M
``update_ci`` calls checked), and is a mathematically guaranteed no-op at
the healthy ~0.5 plateau.

**Important correction**: capping ``ci`` does NOT rescue an
out-of-safe-zone ``max_abs_delta``/lr pocket's own SSE/weight-level
divergence -- with ``max_ci=100.0`` applied, the same unsafe pocket's SSE
still climbs continuously (up to ~313544 by step 29950, matching the
uncapped run almost exactly) and still collapses to an all-zero output in
the final ~50 steps. ``ci`` overflowing to non-finite was NOT the (sole)
root cause of that collapse-to-zero masking -- the weight/``cw``
accumulator has its own SEPARATE unbounded-growth mechanism, untouched by
this ceiling. ``max_ci`` is still worth defaulting on as a genuine, free
structural safety property for ``ci`` itself, but it is NOT a substitute
for staying inside the validated safe ``max_abs_delta``/lr range.

Same NaN/Inf guard as ``PlainRMSpropSynapsePolicy::update_ci``, checked
BEFORE the floor/``max_ci`` clamps since ``std::min``/``std::max``'s
behavior on NaN is comparison-order-dependent, not a reliable NaN filter
on its own.

.. _synapse_policy.clip_order_and_lr_ceiling:

Clip order bug: clip the RAW update, not the lr-scaled one
------------------------------------------------------------

*ID:* ``synapse_policy.clip_order_and_lr_ceiling``

``BoundedRMSpropSynapsePolicy::update_cw`` clips the LR-INDEPENDENT raw
update, THEN multiplies by ``eff_lr`` -- NOT the other way around.

**Real bug, confirmed directly**: an earlier version clipped the
already-lr-scaled delta, making ``max_abs_delta`` an ABSOLUTE cap
regardless of lr -- fine at the ``lr=0.05`` this policy was tuned against,
but silently crushed every real step for callers using a much larger lr
(``FoldedColumnLayer``'s ``lr=1.0`` couldn't converge in its own test's
step budget anymore, since any step that would have exceeded the flat cap
got clamped down to it no matter how much bigger lr was set). Clipping
BEFORE the lr multiply (matching standard gradient-clipping convention,
e.g. ``clip_grad_norm_`` in this project's own Python code) makes the
EFFECTIVE final-delta clip naturally proportional to ``eff_lr`` (==
``eff_lr * max_abs_delta``) without needing a separate multiply -- the same
``max_abs_delta`` value now generalizes across callers using different lr
instead of only being valid at the one lr it was tuned against. See
``tests/unit/sweep_synapse_policy_min_decay_frac.cpp``'s own header
comment for the raw-space value that reproduces the already-validated
``lr=0.05`` tuning exactly (0.1 final-space / 0.05 = 2.0 raw-space).

.. code-block:: cpp

   // as of PR #45, delta_csr_types.hpp -- BoundedRMSpropSynapsePolicy::update_cw:
   VALUE_TYPE raw = /* ... computed from g, ci, S, damp_by_importance ... */;
   if (raw > max_abs_delta) raw = max_abs_delta;
   if (raw < -max_abs_delta) raw = -max_abs_delta;
   const VALUE_TYPE delta = scale_invariant ? (eff_lr * raw / S) : (eff_lr * raw);
   // WRONG (the bug): clipping `delta` (already multiplied by eff_lr)
   // instead of `raw` makes the cap ignore lr entirely.

**Warning -- this does NOT make one fixed ``max_abs_delta`` safe for
UNLIMITED lr** (``tests/unit/sweep_synapse_policy_stochastic.cpp`` Round 2):
the effective final-space clip is ``eff_lr*max_abs_delta``, so a large
enough lr eventually pushes it back into the same large-step territory
that's always been risky -- the underlying resonance risk is about the
ABSOLUTE step size, not the raw/lr split; this redesign fixes the split,
not the ceiling. Measured at the production default
(``max_abs_delta=2.0``): excellent at ``lr<=0.05``, visibly degraded by
``lr=0.2``, diverging in absolute terms by ``lr=0.5``, genuinely unsafe by
``lr=1.0``. ``cpu_backend.cpp``'s ``backward_dense``/``backward`` wrappers
print a one-time stderr warning if called with ``learning_rate > 0.2`` for
exactly this reason.

.. _synapse_policy.block4vec_specializations:

``Block4Vec`` SIMD specializations of the synapse policies
----------------------------------------------------------------

*ID:* ``synapse_policy.block4vec_specializations``

Full explicit specializations of ``PlainRMSpropSynapsePolicy<Block4Vec>``/
``BoundedRMSpropSynapsePolicy<Block4Vec>``, not just a ``VALUE_TYPE
=Block4Vec`` instantiation of the generic scalar template, because the
generic template uses ``std::max``, which does not compile for
``Block4Vec`` (GCC vector-extension ``<`` produces a vector of comparison
results, not a single bool -- same reason ``block4_vec_sqrt``/
``block4_vec_max``/``block4_vec_clip_abs`` are hand-written per-lane loops
rather than calling ``std::`` equivalents). This mirrors the file's own
existing scalar-vs-SIMD math duplication precedent (``disldo_backward``'s
scattered scalar sites vs. its ``Block4Vec`` SIMD sites already hand-code
the same formula twice) -- this policy abstraction unifies the 8 call
sites onto one template parameter, it doesn't eliminate the scalar/SIMD
math split itself. Every guard/clip-order fix documented above for the
scalar versions (NaN/Inf via ``block4_vec_select_finite``, clip-before-
lr-multiply, ``scale_invariant``) is mirrored exactly, per-lane, since a
SIMD full-tile result must match the scalar boundary result for the same
synapse.

.. _sparse_linear_weights_delta.importance_scale_rationale:

Per-row / per-column importance scale: why FP4 needs it
-------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.importance_scale_rationale``

``importance_scale``/``output_importance_scale`` apply a per-row/per-column
scale to STORED importance to get TRUE units before any importance
arithmetic (the activity-correlation ``1+|imp|`` denominator, the per-step
decay). Motivation: FP4's smallest representable nonzero magnitude is 0.5,
but well-conditioned weight init scales as roughly ``1/sqrt(fan_in)`` --
for ``fan_in=1000`` that's ~0.03, far below FP4's floor. Without a scale,
importance would either underflow to zero immediately (losing all
regularization signal) or need artificially inflated raw values that don't
correspond to anything meaningful.

Per-row, not per-layer: different rows can have very different natural
importance-trace magnitude within the SAME layer (different fan-in/
connection counts, especially once synaptogenesis has been running a
while and row nnz has diverged across rows) -- a single layer-wide scale
can't serve a sparsely-connected row and a densely-connected row equally
well at the same time. The column counterpart (``output_importance_scale``)
follows the same reasoning ``output_scale`` does for weight (a synapse's
stored importance and stored weight live at the same ``(row,col)``
position, so if the weight side needs a column term, importance does too)
-- default 1.0, unused by ``from_descriptor`` today (a pure no-op for
every existing path until a caller opts in).

Both are lazily-sized vectors, not pre-sized at construction (this struct
doesn't know ``L.rows`` until ``.connections`` is populated) -- the
getters default any not-yet-touched row/column to 1.0 (exact match to
having no scale at all), so this is fully backward compatible with every
existing caller. Convention: ``true_imp = stored_imp * scale``,
``stored_imp = true_imp / scale``. See
``sparse_linear_weights_delta.rescale_row`` below for changing this
mid-training without losing accumulated data.

.. _sparse_linear_weights_delta.scale_rank_scratch_task295:

``ScaleRankScratch``: heap scratch replaces a hardcoded rank cap
-----------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.scale_rank_scratch_task295``

task #295 (real user request): block4's SIMD backward path
(``linear_disldo.hpp``) used to size its per-rank-component accumulators
as FIXED stack arrays capped at a compile-time ``SCALE_RANK_MAX=4`` --
that constant is now gone. ``scale_rank_scratch`` is instead a set of
persistent, per-instance HEAP buffers that grow (never shrink
automatically) to fit whatever ``scale_rank`` the layer actually uses,
reused across every ``disldo_backward`` call rather than reallocated per
call or per tile -- a naive "heap-allocate a ``std::vector`` fresh on
every tile visited" approach would be a real regression, since that loop
runs once per ``(block-row, block-col, li)`` triple, extremely often.
``ensure()`` is the grow-only path ``disldo_backward`` calls automatically
every call (cheap no-op once already large enough); ``resize_to()`` is the
explicit caller-driven path (``reserve_scale_rank_scratch``) that CAN
shrink, for freeing capacity a layer grew into early on and no longer
needs -- the caller is responsible for not shrinking below what's
currently actually in use, which would corrupt buffers ``disldo_backward``
is actively reading/writing.

Deliberately holds ONLY ``value_type``/``double`` vectors, no
``Block4Vec`` -- this header doesn't include ``block4.hpp`` (``Block4Vec``'s
own home), so ``linear_disldo.hpp``'s block4 SIMD code loads/stores
``Block4Vec`` values from/to these plain buffers via
``block4_vec_load``/``block4_vec_store`` (an already-established pattern,
see ``out_scale_k4``/``mcol4_rank``'s pre-existing use of that same
convention) rather than this struct storing SIMD-typed data directly.

.. _sparse_linear_weights_delta.scale_rank_rationale:

Rank-N scale: why rank-1 sometimes can't represent a row's needs
-------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.scale_rank_rationale``

``scale_rank`` is the RANK of the ``value_scale``/``output_scale``
factorization: ``true_w = quant * S[row,col]``, where ``S`` used to be a
plain rank-1 outer product ``value_scale[row]*output_scale[col]``, now
generalized to ``S[row,col] = sum_{k<scale_rank}
value_scale_k(row,k) * output_scale_k(col,k)``. Runtime-parameterized (not
a C++ template per rank) so no new class/pybind binding is needed per rank
value tried -- rank=1 is the default and reproduces the exact original
behavior; storage for ``k>=1`` defaults to 0.0, so an unconfigured extra
component contributes nothing until trained, and every existing call site
(single-component ``get_value_scale``/``get_output_scale``, meaning
component 0) keeps working unmodified.

**Why rank>1 at all**: a single shared row-scalar (rank-1) can't serve a
row whose columns have genuinely conflicting persistent gradient demand
(column A wants the scale positive, column B wants it negative) -- the
row-aggregate gradient driving ``value_scale`` sums ``g*quant`` across
every column in the row, and opposite-signed real signal cancels there
exactly like noise would, even though neither column's own signal is
actually noisy. A second ``(u2,v2)`` component gives a second,
independently-signed channel to absorb the opposite-signed column instead
of cancelling against the first.

**Scope note**: ``disldo_forward``, and ``disldo_backward``'s scattered-CSR
path AND BOTH block4 paths (FP4 and FP8, ``process_tile`` in
``linear_disldo.hpp``) are all rank-aware, using ``get_scale(row,col)``.
The SIMD fast path (both FP4 and FP8) keeps its 4-wide column
vectorization regardless of rank by looping the rank dimension outside the
lane dimension with real ``Block4Vec`` accumulators per component, rather
than falling back to scalar for rank>1. Scattered and block4 must both be
correct here, not just one: real training layers hold a MIX of both
storages simultaneously via synaptogenesis promotion/demotion, sharing the
same ``value_scale``/``output_scale`` arrays.

**Known limitation**: still rank-1-only (component 0 of a rank>1 layer)
for every ``DeferredScaleWrite`` class (e.g. ``SparseLinearLayerResync``)
on both scattered and block4 paths -- eager multiply-by-not-yet-finalized-
scale would reintroduce the staleness ``DeferredScaleWrite`` exists to
avoid (see ``disldo_backward.deferred_vs_direct_quant`` in
``docs/research/linear_disldo.rst``). If a rank>1 layer ever uses a
``DeferredScaleWrite`` class, its effective scale silently drops every
component beyond 0 there -- tracked as a follow-up, not fixed here.

.. _sparse_linear_weights_delta.rank_mutation_cooldown:

Cooldown counters and grace periods for dynamic rank control
-------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.rank_mutation_cooldown``

``scale_rank_calls_since_mutation``/``additive_rank_calls_since_mutation``
(task #292 fix, direct user instruction: "age-gate is good for now, expose
the option") count calls since the LAST rank mutation of EITHER kind
(apoptosis or neurogenesis) on that branch, not just a per-channel
apoptosis-only age check.

**Real bug this fixes**: AQRS_DESIGN.md's Theorem 10 text says the
``tau_death``/``tau_active`` hysteresis gap "stops a channel from
immediately regrowing the instant it's pruned" -- but that only holds if
gamma's own per-step movement is small relative to the gap. A real 60k-step
MQAR run showed gamma's raw gradient can be large enough to jump the whole
gap in one step, defeating it: 1464 mutations observed in 3000 steps.
Gating BOTH apoptosis and neurogenesis behind "at least
``grow_grace_period_steps``/``shrink_grace_period_steps`` calls since the
last mutation" (symmetric, not just apoptosis-only as before) gives every
mutation a real minimum window to matter, regardless of gamma's gradient
size. Initial value ``UINT32_MAX`` (not 0) so a freshly constructed
layer's first-ever qualifying mutation isn't blocked by a phantom
cooldown; guarded against wraparound since a long idle run would otherwise
increment past ``UINT32_MAX``.

A SEPARATE, per-channel gate closes a different bug: a freshly-grown
channel's gamma starts at exactly 0 (Theorem 9's own "zero contribution"
property for a new channel), so its EMA also starts at ~0 -- which
trivially satisfies apoptosis's own condition (``|gamma|_ema<tau_death``
AND ``C_ema<tau_death``) before the channel has had ANY chance to train.
**Confirmed directly**: without this gate, growth and apoptosis fired on
ALTERNATING steps forever, never letting a new channel survive long enough
to learn anything. Fix: a channel is only ELIGIBLE for apoptosis once its
own age (``scale_gamma_step``, which already increments once per backward
call as an Adam-style bias-correction counter, reused here as a free age
signal) exceeds ``grace_period_steps``. Default ~``1/(1-0.98)``, matching
the EMA's own natural warm-up window at the default ``decay=0.98``.

Both gates are needed together: the branch-level cooldown stops rapid
oscillation of the SAME branch's rank; the per-channel age gate stops a
freshly-BORN channel specifically from being immediately re-killed.

.. _sparse_linear_weights_delta.grow_shrink_asymmetry_biology:

Grow/shrink grace-period asymmetry: within-branch vs. cross-branch
---------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.grow_shrink_asymmetry_biology``

``grow_grace_period_steps``/``shrink_grace_period_steps`` are kept as two
independent knobs (originally split to fix a real 60k-step-run regression
where shrink fired on essentially the same cadence as growth, destroying
accumulated low-rank structure -- e.g. ``additive_rank`` 32->1 -- far
faster than it took to build).

Per direct instruction and a biology-literature check, the well-evidenced
asymmetry is actually CROSS-branch (the multiplicative/``scale_rank``
branch vs. the additive branch should run on very different overall
cadences), not WITHIN a branch (formation vs. elimination rates for a
single mechanism are roughly comparable in the literature). So each branch
defaults its own two grace periods to the SAME value (symmetric within the
branch), while the two branches' defaults differ by ~100x. The parameters
stay independently settable for callers who want within-branch asymmetry
too.

**Multiplicative branch, 50/50** (direct analog of a dendritic spine at
the base of a synapse): in vivo two-photon imaging of adult cortical
spines shows formation and elimination happening on COMPARABLE timescales
at steady state -- e.g. adult barrel cortex shows ~1.5% spines formed vs.
~2.1% eliminated over 3 days (Holtmaat et al., "Transient and Persistent
Dendritic Spines in the Neocortex In Vivo", Neuron 2005; see also
Grutzendler et al., "Long-term dendritic spine stability in the adult
cortex", Nature 2002) -- not the large asymmetry a bigger shrink-only
grace period would imply.

**Additive branch, 5000/5000** (~100x the scale branch's own 50/50,
replacing an earlier un-derived 4x guess): this branch is a whole-layer
low-rank correction driven by the same gamma-EMA machinery integrated
across the entire layer, structurally closer to a neuron sensing its own
aggregate ("global") state than to any one synapse -- matching the
biological distinction between fast local Hebbian-like synaptic change and
slow whole-cell homeostatic plasticity. Hebbian/STDP synaptic changes are
induced on the timescale of seconds to minutes; genuine homeostatic
synaptic scaling (the compensatory, whole-cell response to a sustained
activity change) needs on the order of 24-48 hours of sustained change
before it manifests (Turrigiano and colleagues' classic activity-blockade
experiments; see also Zenke & Gerstner, "Hebbian plasticity requires
compensatory processes on multiple timescales", Phil. Trans. R. Soc. B,
2017, for the general "temporal paradox" framing of fast Hebbian vs. slow
homeostatic timescales as a real, load-bearing separation, not an
implementation detail). Minutes vs. 24-48h is roughly a 100x-3000x
separation depending which endpoints are compared -- 100x is the
conservative end of that range, chosen so the additive branch can still
mutate at all within realistic training budgets rather than being
effectively frozen; the ratio is a real, independently tunable parameter
specifically so it can be swept and tested against real training outcomes
rather than treated as settled here.

.. _sparse_linear_weights_delta.value_scale_step_bias_correction:

``value_scale_step``: per-row/rank step counter for bias correction
----------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.value_scale_step_bias_correction``

Step counter for ``value_scale_importance``'s Adam-style bias correction
(``scale_policy.adam_bias_correction`` above) -- see that section for the
full rationale (~31.6x oversized first step without it, the real
sign-flip regression it fixes). ``uint32_t``, one per row*rank slot --
cheap (unlike a per-synapse counter, which would double memory for an
FP4/FP8 format where every byte counts; per-synapse ``ci`` does NOT get
this treatment for that reason). ``output_scale_step`` is the exact
column-axis counterpart, same mechanism one level over from row to
column.

.. _sparse_linear_weights_delta.value_scale_momentum_dead_row:

``value_scale_momentum``: first moment for the dead-row bootstrap only
---------------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.value_scale_momentum_dead_row``

Adam-style FIRST moment (signed EMA of ``g_agg``) for ``value_scale``,
companion to ``value_scale_importance``'s SECOND moment (EMA of
``g_agg^2``, unsigned). Used only by the dead-row (``nnz_row==0``) path in
``disldo_backward`` -- the existing live-synapse ``value_scale`` update
uses the instantaneous ``g_agg`` directly (RMSprop-style, unchanged). A
dead row has no per-synapse importance (no synapse exists to hold one), so
it needs its own signed accumulator to know which direction to nudge
``value_scale``; ``value_scale_importance`` doubles as its second moment
too (provably untouched by anything else while ``nnz_row==0``, since the
live-synapse loop skips the row entirely). Linear in ``g_agg`` (not
``g_agg^2``) so ``E[update]=0`` under zero-mean noise regardless of
variance -- squaring the pretend/reactive direction (an earlier, rejected
design) would have let an occasional large-magnitude gradient dominate the
accumulated direction even under otherwise-cancelling noise. See
``disldo_backward.value_scale_reduction_dead_row`` in
``docs/research/linear_disldo.rst`` for the full dead-row bootstrap
mechanism this feeds.

.. _sparse_linear_weights_delta.output_scale_trainable_gate:

``output_scale_is_trainable``: opt-in gate, not ``.empty()``
------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.output_scale_trainable_gate``

``output_scale`` is gradient-updated by ``disldo_backward`` like
``value_scale`` is, but ONLY once a caller has explicitly called
``set_output_scale_raw{,_k}`` at least once. This is tracked by an
explicit ``output_scale_is_trainable`` flag rather than checking
``output_scale.empty()``, because ``disldo_backward``'s own internal
pre-sizing resize would otherwise flip an empty-check to "trainable" after
the very first call regardless of caller intent -- the explicit flag is
set only by the setter, so it genuinely tracks caller intent rather than
internal storage state.

.. _sparse_linear_weights_delta.bulk_raw_vector_accessors:

Bulk raw-vector accessors: "virtual neurons" for scale channels
------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.bulk_raw_vector_accessors``

task #295 follow-up: expose AQRS scale channels (``value_scale``/
``output_scale``, later ``additive_u``/``additive_v``) as flat vectors a
caller can read/correct in one pass instead of ``n*scale_rank`` individual
``get_*_k``/``set_*_k`` calls. Motivated by a real incident: raised rank
caps let ``get_scale()``'s unclamped sum overflow in a real fp8 MQAR run,
NaN-collapsing it -- a caller needed to apply a bulk correction
efficiently. Deliberately raw/flat (row-major ``[n*scale_rank]``, same
layout as the member itself) -- the caller already knows ``scale_rank``
and reshapes.

Size-preserving: the setter refuses a size mismatch -- this is a
correction pass over EXISTING trained values, not a resize/grow path
(that's ``set_scale_rank``'s job). Each element is independent (no
cross-element math here), so a caller applying a per-channel, context-free
correction (e.g. clip + auto-correcting shrink) via one vectorized pass
over the returned array is equivalent to invoking that correction once per
channel -- avoids the GIL/threading cost of a literal per-element callback
inside the hot OpenMP forward/backward loops.

.. _sparse_linear_weights_delta.scale_gamma_backward_compat_bug:

``scale_gamma``: the k>=1 lazy-default backward-compat break
--------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.scale_gamma_backward_compat_bug``

AQRS per-channel gamma (task #273/#282-283, see ``sili_peridot/
AQRS_DESIGN.md``'s gamma section): decouples channel MAGNITUDE from
channel DIRECTION. ``value_scale_k``/``output_scale_k`` hold pure
direction; ``gamma_s_k`` (ONE scalar per channel ``k``, not per row/col)
holds the magnitude: ``S[row,col] = sum_k gamma_s_k * value_scale_k(row,k)
* output_scale_k(col,k)``.

**Real backward-compat break, found via the full regression suite after
landing gamma**: the lazy default for ``get_scale_gamma_k`` must be 1.0
for EVERY ``k``, not just ``k==0``. An earlier version defaulted ``k>=1``
to 0.0 to match Theorem 9's "new channel starts at zero contribution" --
but that silently broke every EXISTING rank>1 layer that sets
``value_scale_k``/``output_scale_k`` directly (the established, still-valid
construction pattern -- e.g. direct ``w.scale_rank = N`` assignment,
bypassing ``set_scale_rank`` entirely) without ever touching gamma: their
``k>=1`` components went from contributing normally to silently zeroed,
since ``scale_gamma`` stays empty and the lazy default used to kick in
unconditionally. Confirmed by a real regression:
``test_scale_handling.cpp``'s ``magnitude_rescale_output`` rank-2 test
failed after the ``k>=1->0.0`` default landed.

.. code-block:: cpp

   // as of PR #45, delta_csr_types.hpp:
   inline value_type get_scale_gamma_k(std::size_t k) const {
       if (k < scale_gamma.size())
           return scale_gamma[k];
       return value_type(1);  // transparent default for EVERY k
   }
   // WRONG (the bug): `return k == 0 ? value_type(1) : value_type(0);`
   // silently zeroed every k>=1 component of any rank>1 layer that had
   // never touched gamma at all.

Fix: the "new channel = zero contribution" property (still needed for
task #273's real dynamic growth) now lives ONLY in ``set_scale_rank``'s
reshuffle, which writes an explicit 0.0 into a genuinely NEW ``k`` slot --
but ONLY fires when ``scale_gamma`` is already non-empty (i.e. gamma is
ALREADY in active use, matching #273's actual use case: a layer under live
dynamic rank control). For any layer that never touches gamma, the
reshuffle no-ops on an empty array and the lazy default applies uniformly
-- gamma stays fully transparent, bit-identical to pre-gamma behavior.

``scale_gamma_is_trainable`` is the same opt-in gate as
``output_scale_is_trainable``, for the same reason: ``disldo_backward``'s
gamma UPDATE (not the accumulation, which is harmless and cheap even when
unused) must be skipped entirely unless a caller has explicitly engaged
gamma via ``set_scale_gamma_raw_k``. **Confirmed as a real regression**:
without this gate, ``test_aqrs_additive_branch.cpp`` and
``test_aqrs_rank_growth_shrink.cpp`` (neither of which ever touches gamma)
both failed, because ``gamma_s_k(0)`` was drifting away from 1.0 on every
step -- every existing rank>=1 layer, including ones that had never heard
of gamma, would otherwise get an unsolicited gradient-driven perturbation
to its effective magnitude.

.. _sparse_linear_weights_delta.gamma_ema_tracker_shared:

``GammaEMATracker``: shared machinery between the two gamma branches
---------------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.gamma_ema_tracker_shared``

Shared EMA-tracking + Theorem 10 trigger-condition machinery for AQRS's
per-rank-channel gamma, used by BOTH the multiplicative branch
(``scale_gamma``) and the additive branch (``additive_gamma``) -- the math
(EMA update formula, apoptosis/neurogenesis trigger conditions, channel
swap) is IDENTICAL between the two branches (AQRS_DESIGN.md's Theorem 10 is
stated "per branch, per rank channel"); only the raw gamma VALUE's own
storage/lazy-default semantics differ between branches (both default
transparently to 1.0, but for slightly different backward-compat reasons
-- see ``sparse_linear_weights_delta.scale_gamma_backward_compat_bug``
above and ``sparse_linear_weights_delta.additive_gamma_backward_compat_bug``
below), so only that piece stays branch-specific, living directly on
``SparseLinearWeightsDelta`` rather than in this shared struct. Extracted
per direct instruction not to duplicate ``scale_gamma``'s already-proven
EMA/trigger logic when adding the equivalent for ``additive_gamma``.

Three tracked signals per channel: ``abs_ema`` (EMA of the channel's own
``|gamma_k|`` magnitude), ``share_ema`` (EMA of ``C_k = |gamma_k| /
sum_j|gamma_j|``, this channel's share of the group's total L1 mass), and
``grad_ema`` (EMA of ``|dL/d(gamma_k)|`` -- gamma is a scalar per channel,
so this reduces to a plain magnitude, not a Frobenius norm; there's no
row/col structure at the gamma level to norm over). ``decay=0.98`` matches
the same EMA pattern already used for ``loss_ema``/``acc_ema`` in
``sili_peridot``'s MQAR curriculum (``train_mqar_curriculum.py``) -- not a
new convention, reused deliberately. Updated EVERY step, not every N steps
-- ``AQRS_DESIGN.md`` explicitly rejects periodic checking as a "luck
filter," not a real noise filter -- and only AFTER gamma's own value
update for every channel is finalized for that step, since ``share_k``
needs every channel's current ``|gamma|`` to compute the group's L1 norm
first.

``should_apoptose``/``should_neurogenesis`` implement Theorem 10's exact
triggers verbatim: apoptosis is ``(|gamma_i|_ema < tau_death) AND
(C_i_ema < tau_death)``; neurogenesis is ``(min_j |gamma_j|_ema >
tau_active) AND (max_j grad_j_ema > theta)``, evaluated against the EMA
values (the noise filter), not raw instantaneous gamma.
``should_neurogenesis`` takes the current ``rank`` explicitly (not stored
in the tracker) since "every existing channel" means every ``k<rank``, not
every ``k`` the EMA arrays happen to have grown to -- a channel could have
been apoptosed/shrunk away, leaving stale EMA history at a now-unused
index.

``swap_k``: EMA state travels WITH a relocated channel -- omitting this
would make a relocated channel look freshly-born to the trigger logic,
defeating the whole point of EMA smoothing as a real noise filter (see
``sparse_linear_weights_delta.rank_swap_and_resize`` below for why
channels get relocated at all).

.. _sparse_linear_weights_delta.get_scale_formula:

``get_scale``: the combined rank-N + gamma formula
------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.get_scale_formula``

``S[row,col] = sum_{k<scale_rank} gamma_s_k * value_scale_k(row,k) *
output_scale_k(col,k)`` -- THE quantity Hadamard-multiplied against
``quant`` in both ``disldo_forward`` and ``disldo_backward``'s quant-update.
Replaces the old ``get_value_scale(row)*get_output_scale(col)`` two-call
pattern wherever the caller wants full rank-N behavior (scattered-CSR
forward/backward; block4's paths still use the two-call rank-1-only form,
see ``sparse_linear_weights_delta.scale_rank_rationale`` above).

.. _sparse_linear_weights_delta.additive_branch_overview:

AQRS additive branch: why an additive term is structurally necessary
---------------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.additive_branch_overview``

``A[row,col] = sum_{k<additive_rank} additive_u_k(row,k) *
additive_v_k(col,k)`` -- ADDED (not Hadamard-multiplied against ``quant``
like ``value_scale``/``output_scale`` above) to the effective weight. See
``sili_peridot/AQRS_DESIGN.md`` for the full derivation: proven
structurally necessary (not just useful) because the multiplicative
branch's gradient is exactly zero at any ``(row,col)`` where the quantized
weight is the zero code, at any rank -- only an additive term can write a
value there. Same two-plain-vector convention as ``value_scale``/
``output_scale`` (no separate ``diag(gamma)`` scale term until gamma was
added, at task #289 -- see below), confirmed by reading that
implementation first rather than inventing a different convention.
``additive_rank`` default 0 means the branch is a pure no-op (matches
``value_scale``/``output_scale``'s own "unconfigured component contributes
nothing" convention, just at rank 0 instead of per-component).

Optimizer state (importance/step/momentum for an Adam-style update) was
deliberately NOT added when the branch itself landed -- that was task
#277's scope, tied to the specific policy chosen. Adding it earlier
without knowing that design would have risked exactly the kind of
guessed-then-duplicated state this project's own "don't duplicate code"
instruction warns against. See
``sparse_linear_weights_delta.additive_optimizer_state`` below for what
landed.

``get_additive`` materializes a single ``(row,col)`` entry directly --
useful for tests/small-scale callers only. Real forward/backward paths
(``linear_disldo.hpp``, task #276/#277) MUST use the fused Theorem-11 form
(project ``X`` down to rank ``additive_rank`` via ``additive_u``, scale,
project back up via ``additive_v``) instead of calling this per-entry
across a whole matrix, which would defeat the entire point of the
low-rank representation.

.. _sparse_linear_weights_delta.additive_gamma_backward_compat_bug:

``additive_gamma``: same backward-compat break, caught before landing
--------------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.additive_gamma_backward_compat_bug``

AQRS per-channel gamma for the additive branch (task #273/#282-283, wired
into forward/backward for real at task #289), same role as ``scale_gamma``
but for ``additive_u``/``additive_v``: ``A[row,col] = sum_k
additive_gamma_k * additive_u_k(row,k) * additive_v_k(col,k)``, matching
``AQRS_DESIGN.md``'s ``A(theta_o) = sum_k gamma_o_k*u_k*v_k^T`` exactly.

**Same class of bug as ``scale_gamma``'s own** (see
``sparse_linear_weights_delta.scale_gamma_backward_compat_bug`` above),
but caught here BEFORE landing rather than via a regression: an earlier
version of this code defaulted ``get_additive_gamma_k`` to 0.0 on the
reasoning "the additive branch has no legacy always-on component to
preserve." That's true of ``additive_rank`` itself (0 = fully off) but NOT
of gamma once it's actually multiplied into the branch's forward/backward
math -- every EXISTING caller that sets ``additive_rank>0`` and populates
``additive_u``/``additive_v`` directly (task #278's pybind bindings, the
fp8/fp4 MQAR curriculum runs already on record) never touches gamma at
all, so a 0.0 lazy default would silently zero out their entire additive
contribution the moment gamma got wired in. Lazy default is 1.0
(transparent), exactly mirroring ``scale_gamma``'s own
``get_scale_gamma_k`` -- Theorem 9's "new channel = zero contribution"
property lives ONLY in ``set_additive_rank``'s reshuffle (``zero_default``),
same split as ``scale_gamma``'s.

.. _sparse_linear_weights_delta.additive_gamma_adam_overshoot:

``additive_gamma`` uses RMSprop, not Adam -- a real test failure
-----------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.additive_gamma_adam_overshoot``

``additive_gamma_state``/``additive_gamma_step`` match ``scale_gamma``'s
OWN update policy exactly (``disldo_backward``'s additive-gamma update
block uses the function's generic ``ScalePolicy`` template param, same as
``scale_gamma``), NOT ``AdamScalePolicy``, even though ``additive_u``/
``additive_v`` themselves use ``AdamScalePolicy``.

**Found via a real, direct test failure**: an earlier version used
``AdamScalePolicy`` here to "match ``additive_u``/``v``'s own optimizer
choice." Adam's momentum term overshoots a hard L1-created zero fixed
point (Theorem 8), since Adam keeps pushing in its accumulated momentum
direction for a step or two AFTER the raw gradient has already crossed
zero, driving gamma persistently negative instead of settling exactly at 0
the way ``scale_gamma``'s own (momentum-free) L1 test does. gamma is the
SAME kind of shared magnitude-decoupling parameter in both branches with
the SAME Theorem 8 exact-zero-fixed-point requirement, so it uses the SAME
(RMSprop) policy in both -- only the direction vectors (``value_scale``/
``output_scale`` vs. ``additive_u``/``v``) get to pick their own optimizer
independently.

.. _sparse_linear_weights_delta.additive_optimizer_state:

``additive_u``/``additive_v``: dedicated Adam state (task #277)
---------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.additive_optimizer_state``

``additive_u_momentum``/``additive_u_state``/``additive_v_momentum``/
``additive_v_state`` plus their step counters are ``AdamScalePolicy``'s
own state for ``additive_u``/``additive_v`` (task #277) -- same
lazy-growth, row-major-per-component convention as everything else in this
struct. Two independent EMAs per Adam's own definition (first moment =
momentum, second moment = state), plus one step counter for bias
correction (see ``scale_policy.adam_additive_branch`` above for why this
branch defaults to Adam rather than RMSprop).

.. _sparse_linear_weights_delta.rank_swap_and_resize:

Safe rank resize: the flat-index reinterpretation bug
--------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.rank_swap_and_resize``

**Confirmed bug this section fixes**: naively assigning ``scale_rank =
new_rank`` does NOT reshuffle the existing flat ``row*old_rank+k``
storage. If any row beyond row 0 already has live data, changing
``scale_rank`` silently REINTERPRETS existing entries at the wrong flat
index -- e.g. rank 1->2: what was row 1's rank-0 value at flat index 1
becomes row 0's rank-1 value under the new indexing. No existing caller
ever hit this because ``scale_rank`` was always set exactly once at
construction, before any row was touched -- but task #273's dynamic rank
control needs to resize a LIVE, already-trained layer, so this had to be
fixed for real, not left as a footgun.

``set_scale_rank``/``set_additive_rank`` fix this via
``reshuffle_rank_array`` (a free function, not a member, specifically so
``GammaEMATracker`` can reuse the identical logic for its own rank-length
arrays rather than hand-duplicating it): preserves every existing
``(entity,k)`` pair with ``k<min(old_rank,new_rank)`` and fills any new
slots with a caller-supplied default. ``set_scale_rank`` handles both the
multiplicative arrays (``value_scale``/``value_scale_importance``/
``value_scale_step``/``value_scale_momentum``, ``output_scale``/
``output_scale_importance``/``output_scale_step``, ``scale_gamma`` and its
own state) and (in ``set_additive_rank``) the additive arrays
(``additive_u``, ``additive_v``, and their own Adam state) with one shared
reshuffle helper rather than hand-duplicating the same logic eight-plus
times. ``scale_gamma``'s own reshuffle deliberately uses ``zero_default``,
NOT the ``k==0``-special ``scale_default`` that ``value_scale``/
``output_scale`` use: a genuinely NEW gamma channel should start at 0
regardless of ``k`` (Theorem 9), and this reshuffle only ever fires once
gamma is already live (``scale_gamma`` non-empty) -- gamma's own
"transparent by default" behavior for an UNTOUCHED layer comes entirely
from ``get_scale_gamma_k``'s lazy fallback (1.0), not from this reshuffle.

``additive_rank`` legitimately floors at 0 (branch fully off) --
``set_additive_rank`` has no ``new_rank==0`` guard, unlike
``set_scale_rank``, which throws below rank 1 (component 0 IS the
original rank-1 behavior every existing caller depends on).

``swap_scale_channels``/``swap_additive_channels``: ``set_scale_rank``'s
own reshuffle can only SHRINK by truncating the highest-index channel
(``k>=new_rank`` is simply dropped), but Theorem 10's apoptosis trigger can
fire on ANY channel, not just the last one. These swap the dying channel
to the end first, then the caller calls ``set_scale_rank(rank-1)``/
``set_additive_rank(rank-1)`` to truncate it -- the general "remove an
arbitrary channel" primitive. They use the existing ``get_*``/``set_*_raw_k``
accessors (not raw vector indexing) specifically so lazy-unpopulated
rows/cols are read via their correct defaults and force-written, rather
than silently skipped -- a plain vector swap would corrupt any row/col
that hadn't been touched yet at one of the two indices. Gamma's own EMA
state and step counter (which doubles as the channel's AGE, see
``sparse_linear_weights_delta.rank_mutation_cooldown`` above) travel WITH
the swapped channel for the same reason
``gamma_ema_tracker_shared.swap_k`` does -- a relocated channel keeping
its OLD position's age/EMA would corrupt the grace period and trigger
logic exactly like a missed swap would.

.. _sparse_linear_weights_delta.dynamic_rank_control_generic:

``apply_dynamic_rank_control_generic``: shared control flow, one mutation per call
-----------------------------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.dynamic_rank_control_generic``

Shared control-flow for Theorem 10's apoptosis/neurogenesis dynamic rank
control (task #289), used by BOTH ``apply_dynamic_rank_control``
(multiplicative branch) and ``apply_additive_dynamic_rank_control``
(additive branch) -- the DECISION logic (grace-period-gated
apoptose-else-neurogenesis, at-most-one-mutation-per-call) is IDENTICAL
between branches; only the actual mutation operations differ (different
swap/resize/seed calls per branch, and a different ``min_rank`` floor --
``scale_rank`` can never legally drop below 1, ``additive_rank``
legitimately floors at 0), so those are passed in as callbacks rather than
duplicating this control flow a second time -- extracted per direct
instruction not to copy ``apply_dynamic_rank_control``'s own logic when
adding the additive-branch equivalent.

Evaluates Theorem 10's triggers against the CURRENT EMA state (updated
automatically every ``disldo_backward`` call, task #284) and performs at
most ONE real mutation per call: apoptose the first dying channel found
(swap-to-end then shrink), or grow one new channel if neurogenesis fires
and no channel is currently dying. ONE mutation per call, not "handle
everything in one pass" -- apoptosis and neurogenesis firing on the SAME
call would mean the signal that triggered growth was measured against a
rank about to change anyway; simpler and safer to let the next call
re-evaluate against the post-mutation state.

``new_channel_seed``/``new_channel_seed_u``/``new_channel_seed_v``:
Theorem 9 says a new channel's direction should align with the residual's
top singular vector; ``AQRS_DESIGN.md`` marks the practical proxy for this
(``neuron_grad_accum``/importance) as UNRESOLVED, not yet verified. These
functions deliberately do NOT hardcode that unverified proxy -- they take
the new channel's per-row/per-col direction as caller-supplied callbacks
instead, so a caller can pass real residual-aligned values once Theorem
9's proxy is validated, or (as every existing test in this codebase
already does for growth) a simple deterministic nonzero seed just to break
the symmetric zero-init deadlock in the meantime. The multiplicative
branch's ``output_scale`` side is seeded uniformly (1.0) -- no col-side
residual signal is available at this layer of the API. The additive
branch needs two real seed callbacks (not one + a uniform default) because
BOTH ``additive_u`` and ``additive_v`` need a real nonzero direction for a
new channel to generate any gradient at all -- see ``_seed_additive_rank``'s
own docstring (``sili/sparse_rnn.py``) for the identical reasoning applied
to construction-time seeding.

.. _sparse_linear_weights_delta.importance_value_stats_thread_safety:

Running Hoyer-sparsity stats: incremental tracking and a real race
--------------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.importance_value_stats_thread_safety``

``importance_l1``/``importance_l2_sq``/``importance_max_abs`` and their
``value_*`` counterparts are running L1 / L2^2 / max\|.\| for STORED
(quantized) importance and weight values, maintained incrementally (O(1)
per synapse touched, not a full-layer rescan) by
``update_importance_stats``/``update_value_stats``. These track the STORED
distribution specifically (not true units), since the question they
answer is "is the FP4 representable range being used well," which is
about the quantized values as they actually sit in the buffer. ``double``,
not ``value_type`` (float): these are long-running sums across potentially
millions of training steps, and float32 accumulation drift is a real risk
there even though individual synapse values stay float.
``hoyer_importance()``/``hoyer_value()`` compute Hoyer's sparsity measure
from these in O(1); call ``recompute_stats()`` once after constructing a
layer via ``delta_csr_from_absolute()`` (or any other path that writes
values without going through ``update_*_stats``), since these start at
zero otherwise.

``max_abs`` is a MONOTONIC upper bound, not a live exact current max -- if
the element currently holding the max shrinks, ``max_abs`` cannot decrease
without a rescan (unlike L1/L2^2, which update exactly via O(1)
arithmetic). Useful as "has this layer ever touched the ceiling," not "what
is the max right now"; call ``recompute_stats()`` for an exact value when
that distinction matters. ``max_abs_decay`` (default 1.0, no decay -- exact
backward compat, a pure monotonic bound as before) is applied to
``max_abs`` on every update BEFORE comparing against the new value
(``new_max = max(old_max * decay, |new_val|)``); a decay slightly below
1.0 (e.g. 0.9999) lets ``max_abs`` drift downward over time when the
element that set it has since shrunk, an approximate self-correcting live
max rather than an exact one.

**Real bug, found and fixed**: ``update_importance_stats``/
``update_value_stats`` mutate shared state with no locking -- safe to call
from single-threaded code, or serially after a parallel region, but NOT
safe to call concurrently from multiple OpenMP threads. This was
originally called directly inside ``#pragma omp parallel`` loops in all
four kernels, racing on these exact fields, undetected because every test
used ``num_cpus=1``. Fix: for parallel kernels, each thread accumulates
locally (sum of ``|new|``, sum of ``|old|``, sum of ``new^2``, sum of
``old^2``, local max) and calls
``update_importance_stats_aggregate``/``update_value_stats_aggregate``
ONCE per thread after the parallel region instead -- equivalent to calling
the per-synapse update for every synapse that thread touched, batched into
4 sums instead of per-synapse calls.

.. _sparse_linear_weights_delta.rescale_row:

Per-row rescale: re-encode, don't reinterpret
---------------------------------------------------

*ID:* ``sparse_linear_weights_delta.rescale_row``

``rescale_importance_row``/``rescale_value_row`` change one row's
``importance_scale``/``value_scale`` mid-training without losing that
row's accumulated data: re-reads its stored value at whatever scale it
currently has into true units, re-encodes at the new scale. Without this,
just assigning a new scale directly would silently reinterpret existing
stored values as if they'd always been at the new scale -- corrupting
every synapse's importance/weight in one step, not just changing how
future arithmetic treats it. Uses plain ``set()``, not a ``_live`` variant
-- this re-encodes whatever value the row ALREADY had under a new scale
(reparametrization, same as ``block4_maybe_promote``), not a training
update. A row can contain a freshly-grown, never-yet-trained synapse whose
weight/importance is deliberately 0 (``insert_col``'s own convention);
redirecting that to a nonzero live code here would be the same corruption
class the ``block4_maybe_promote`` regression already caught (see its
comment in ``delta_csr_memory.hpp`` for the full incident).
``rescale_importance``/``rescale_value`` are the bulk "set every row to
the same new_scale" convenience wrapper, backward-compatible with the
original per-layer-scalar design.

.. _sparse_linear_weights_delta.magnitude_rescale_output:

``magnitude_rescale_output``: moving magnitude from output_scale into the code
--------------------------------------------------------------------------------------

*ID:* ``sparse_linear_weights_delta.magnitude_rescale_output``

Gradient-free reparametrization: ``true_weight = stored_w *
value_scale[row] * output_scale[col]`` is algebraically UNCHANGED by this
-- only WHERE the magnitude lives moves, from ``output_scale`` into the
stored per-synapse weight code. Drives each column's stored-weight RMS
(across all ``n_in`` rows, including rows with no synapse in that column --
matches a dense parameter's zero-padded mean) toward ``target`` via a
DAMPED (``correction_rate``) multiplicative step per call rather than
jumping there in one shot. Ported from ``sili_peridot``'s torch-validated
prototype (``toy_tile_recurrence_rmt_torch.py``'s ``_magnitude_rescale``)
-- see that module for the full derivation and the empirical finding that
column-only (not also row/``value_scale`` -- "both axes" was tested and
found to consistently HURT) is the winning configuration.

``scale_invariant``: when true, per-synapse ``ci`` already tracks the RAW
gradient ``g`` (decoupled from ``S=value_scale*output_scale`` via
``update_cw``'s own ``scale_invariant`` flag, see
``synapse_policy.scale_invariant_quadratic_bug`` above) so it does NOT
need rescaling here. When false, ``ci`` is calibrated to ``(g*S)^2`` --
shrinking ``output_scale`` by ``k`` without correspondingly rescaling
``ci`` silently changes every touched synapse's effective RMSprop step
size.

Column RMS is measured over ``n_in`` (the row COUNT), not
``nnz_in_col`` -- a column with zero active synapses is skipped entirely
(``k=1``, no-op) rather than treated as a real all-zero column, since "no
synapse" (sparse) and "synapse present but currently zero" (torch's dense
``w_stored``) are genuinely different things the sparse engine has no
reason to conflate; a torch all-zero-but-present column would otherwise
also degenerate toward the ``eps`` floor.

Covers BOTH storages -- scattered CSR (``connections``) AND block4
(``block4``) -- not scattered-only. A real training layer promotes
synapses between the two continuously (synaptogenesis/pruning), so a
column's live weight can live in either storage, or split across both, at
any given moment; rescaling only one side would silently leave the other
side's synapses un-rescaled while still dividing the SHARED
``output_scale[col]`` they both read, corrupting their true weight. Both
FP4 (``Block4Store``, nibble-packed weight\|imp<<4) and FP8
(``Block4Store8``, separate weight/importance byte planes) are handled via
the same ``if constexpr`` dispatch ``process_tile``-style code elsewhere
in this codebase already uses -- see ``delta_csr_memory.hpp``'s own
scattered+block4 combined-export loop for the read-side precedent this
mirrors. Re-quantization here is DETERMINISTIC (``fp4_quantize``/
``fp8_quantize``), matching ``rescale_value_row``'s own convention for this
class of scale-bookkeeping rewrite, not the gradient-driven stochastic
``set_stochastic()``.

Final step divides EVERY rank component's ``output_scale_k(col,ki)`` by
the SAME column-level ``k[c]`` -- since ``S/k[c] = sum_ki(vs_ki*(os_ki/
k[c])) = (sum_ki vs_ki*os_ki)/k[c]`` (distributive law), this generalizes
cleanly to any ``scale_rank``; at ``scale_rank==1`` it's identical to the
original single-component form.
