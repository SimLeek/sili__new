``sparse_rnn.py`` research notes
===================================

Companion doc to ``sili/sparse_rnn.py``. Source comments point back here by
anchor ID (``*ID:* `` marker under each heading below); this doc links back
to source by class/function name. See ``docs/research/linear_disldo.rst``
for the pattern this follows (semantic dotted anchor IDs, visible ID
markers, frozen code snippets on real-bug/non-obvious-derivation sections),
and ``docs/research/energy.rst`` for the ``EnergyDynamics``/branching-ratio
background this module builds on directly.

.. _sparse_rnn.module_overview:

Module architecture: inline weight updates, BPTT modes, continuous synaptogenesis
---------------------------------------------------------------------------------------

*ID:* ``sparse_rnn.module_overview``

Forward flow in ``SparseRNNCell``::

    obs   (Tensor) --[DISLDOLayer]--------------------------> Tensor
    state (Tensor) --[CSR.from_dense]--[SISLDOLayer]---------> Tensor
                                                  sum --[EnergyDynamics]--> h_out

``h_out.data`` -> CSR cached inside the cell for the next step's recurrent pass.

Weight VALUE updates are inline, not a separate ``optimizer.step()``:
``DISLDOLayer``/``SISLDOLayer.forward(x, learning_rate)`` store
``learning_rate`` and their ``_backward`` closure calls
``SparseLinearLayer.backward_dense``/``backward_sparse`` with it directly --
gradient computation AND the weight update happen together, during
``aux_loss.backward()``/``loss.backward()`` itself. This is required to
keep memory bounded (no separate accumulate-then-apply buffer held between
calls) -- the old two-phase "compute gradient, then separately
``.step(lr)``" convention this module used to assume was actually wrong for
this system, not merely unimplemented.

BPTT=1 (default)::

    agent.train_step(obs)   # detaches state, forward(learning_rate=self.lr), aux_loss.backward()

Multi-step BPTT::

    for obs in episode:
        action = agent(obs, learning_rate=0.0)   # state stays in graph, no update during rollout
        ...
    loss.backward()                              # inline weight updates fire here
    agent.state = agent.state.detach()

Only structural growth (synaptogenesis: ``build_probes`` + ``synap_step`` +
``equalizer_step``) is a genuinely separate call, since it changes which
synapses exist rather than updating a value. It's cheap and O(1)-ish by
design -- ``SparseRNNAgent.step()`` calls it every online step, not on an
"every N steps" cadence (a periodic throttle would reintroduce the lag
spikes this stepwise design exists to avoid). There is no importance-decay
call in this API generation -- importance already settles via the
ADSP-style activity-correlation tracking inside forward/backward, and a
separate periodic decay interacting correctly with FP4-quantized stored
values would need more care than a simple multiply (values only resolve to
FP4 granularity, and large-error entries already get squeezed out through
ongoing training) -- a deliberate difference from the old design, not an
oversight.

.. _sparse_layer_base.synaptogenesis_importance_eps_ghost_floor:

``synaptogenesis``'s ``importance_eps``: a read-time-only "ghost" floor
------------------------------------------------------------------------------

*ID:* ``sparse_layer_base.synaptogenesis_importance_eps_ghost_floor``

``importance_eps`` floors an EXISTING synapse's stored importance ONLY for
``synap_step``'s ``importance_cutoff`` comparison -- it is NEVER written to
storage anywhere. A synapse whose real, ongoing-training importance has
decayed to exactly the FP4 zero code (a real, discrete quantization bucket
many independently-decaying synapses can land on simultaneously -- FP4's
smallest nonzero magnitude is 0.5, so 0 is a wide, common landing bucket)
isn't automatically "below cutoff" the instant it gets there.

Does NOT affect removal-priority sort order or capacity-driven
(``keep > max_row_weights``) pruning, and does NOT affect ``build_probes``
at all -- a freshly-grown synapse is stored with whatever its REAL probe
score is (``input_accum*grad_accum``, often exactly 0 for a row with no
activity yet); this floor just stops it being evicted purely for reading as
0 on its first subsequent visit, giving real backprop time to move it.

``max_prune_per_step`` caps how many connections THIS row's call may remove
at once, regardless of why they tied for lowest importance -- a safety
ceiling (default rarely binds), not a throttle on ordinary capacity
trimming.

.. _preseed_random_sparse.bootstrap_rationale_and_capacity_order:

``_preseed_random_sparse``: why the bootstrap exists, and a load/equalize ordering bug
----------------------------------------------------------------------------------------------

*ID:* ``preseed_random_sparse.bootstrap_rationale_and_capacity_order``

A freshly-constructed ``SparseLinearLayer`` has zero connections and
produces literal all-zero output until synaptogenesis grows some. NOT
strictly required when ``EnergyDynamics`` is in the loop (as it is for
``SparseRNNCell``): its forced-firing (energy accumulates via ``drive``
alone, independent of input/weights, until a neuron crosses the fire
threshold and outputs a forced constant) already gives an all-zero-weight
layer a real, if slower, way to bootstrap activity and a gradient signal to
grow from -- closer to RL-style forced exploration than a dead end. This
preseed is purely an optimization for a faster/more immediate initial
bootstrap (useful in tests, or whenever waiting several steps for
forced-firing to kick in isn't wanted) via a standard random-sparse-init
pattern -- not a requirement.

CSR rows are INPUTS, columns are OUTPUTS (``disldo_forward`` iterates
``for r in range(n_inputs)``, each row's nonzero entries its output
connections) -- see ``linear_disldo.hpp``.

Ordering bug avoided: ``load_weights`` is a tight/exact-fit encode
regardless of any prior allocation -- ``equalize_to_capacity`` must be
called AFTER it, not before, or the headroom gets compacted away
immediately (verified empirically). This also calls
``equalize_to_capacity`` so later ``synap_step`` calls have per-row
headroom to grow into -- ``synap_step`` raises if a row's pre-allocated
space is already full, and ``load_weights``' own allocation has no spare
room. Not ``expand_headroom_to`` -- despite its docstring promising per-row
headroom after a full ``equalizer_step`` pass, that didn't hold up
empirically here; ``equalize_to_capacity``, called AFTER ``load_weights``,
is what actually works.

``rng`` defaults to ``np.random.default_rng()`` (fresh OS entropy,
genuinely non-reproducible by design -- NOT affected by
``np.random.seed()``, which only controls the legacy global
``RandomState``, not this Generator API). Pass an explicit
``np.random.default_rng(seed)`` for reproducible wiring in tests -- direct
request: keep true randomness as the default, only make it OVERRIDABLE,
not seeded-by-default.

.. _overflow_guard_array.two_part_correction_design:

``_overflow_guard_array``: the real AQRS unbounded-envelope bug, and why a hard clip alone wasn't enough
------------------------------------------------------------------------------------------------------------------

*ID:* ``overflow_guard_array.two_part_correction_design``

Elementwise, context-free correction for one AQRS scale/additive channel
array (``value_scale``, ``output_scale``, ``additive_u``, ``additive_v`` --
task #295 follow-up / task #286 "virtual neuron" exposure).

Real bug this fixes: ``get_scale()``'s combined envelope
``S(row,col) = sum_k gamma_k*value_scale_k*output_scale_k`` is an
UNCLAMPED sum (``delta_csr_types.hpp``) -- raising ``scale_rank_max``/
``additive_rank_max`` past the old hardcoded 4 let a real fp8 MQAR
curriculum run's per-channel scale values grow unbounded, overflowing
``S`` in the forward pass and NaN-collapsing training (confirmed: the NaN
onset lined up exactly with a rank mutation on ``q_proj``).

Two parts, deliberately NOT a plain hard clip alone (direct instruction: a
hard clip's backward either gives zero gradient or gradient computed from
the wrong post-clip value once a channel is pinned at the boundary --
neither tells upstream training "shrink this"):

1. Auto-correcting shrink: once ``|value|`` exceeds ``near``, subtract a
   nudge proportional to the excess, always pointing back toward zero --
   exactly the gradient of a ``0.5*coef*relu(|x|-near)^2`` hinge-squared
   penalty, applied directly as a value-space correction (these channels
   are C++-internal RMSprop-optimized state, not Python Tensor autograd
   leaves, so there's no backward pass to inject an extra gradient INTO --
   this achieves the same corrective effect one call later, sufficient
   since ``S`` is fully rebuilt from these stored values every forward
   call).
2. Hard clip to ``clip`` as the final numerical-safety net (and
   ``np.nan_to_num`` first, since a value that's ALREADY NaN/Inf wouldn't
   be fixed by ``np.clip`` alone -- ``np.clip(nan,...)==nan``).

``near`` should sit comfortably below ``clip`` so the corrective nudge has
room to act before the hard clip ever engages. Per-channel, not
aggregate-``S`` (direct instruction: chose this over clipping ``S``
itself) -- each element is corrected independently with no knowledge of
other channels, so one vectorized numpy pass over the whole array is
exactly equivalent to "run the same correction once per channel" without
the GIL/threading cost of a literal per-element callback inside the hot
C++ OpenMP forward/backward loops.

.. _orthogonality_penalty_array.normalized_space_bug:

``_orthogonality_penalty_array``: real bug -- raw-magnitude correction was CUBIC, not linear
------------------------------------------------------------------------------------------------------

*ID:* ``orthogonality_penalty_array.normalized_space_bug``

Ongoing (every-step) diversity penalty for one AQRS channel array
(``value_scale``, ``output_scale``, ``additive_u``, ``additive_v`` --
direct instruction: preferred over residual-targeted growth, since an
init-time-only fix doesn't stop channels drifting back toward redundancy
as training continues, whereas this is a per-step force).

Real problem this addresses: nothing else in the AQRS design prevents two
rank channels from converging to duplicate directions -- neurogenesis's own
health check (``abs_gamma_k``/``grad_ema``) is purely magnitude-based, so a
channel that's redundant with another reads as "healthy" even though it
adds no new capacity. ``l1_sparsity_coef`` doesn't help either -- it
penalizes the SUMMED output after every channel's already combined, with no
visibility into the per-channel decomposition.

Standard soft-orthogonality regularizer: gradient-descent step on
``sum_{k1!=k2}(cos(M_k1,M_k2))^2`` -- correlation of DIRECTION only,
magnitude excluded.

Real bug found running this against live training (real MQAR curriculum):
the FIRST version used raw (non-normalized) columns, computing
``G=M^T@M`` and stepping ``M -= coef*M@G_offdiag`` directly. For real
channel magnitudes in the 10s-100s (exactly the range
``apply_scale_overflow_guard``'s own ``near``/``clip`` thresholds exist to
handle), ``G`` entries scale as ``O(n*mag^2)`` and the correction scales as
``O(rank*n*mag^3)`` -- CUBIC in magnitude. Every synthetic test used
near-unit-magnitude vectors, where this is negligible; against real
training it exploded, and because this penalty was called AFTER the
overflow guard in the training loop, nothing sanitized the result before it
was stored -- NaN-collapsed a real fp8 run at step 12650, even earlier than
the original unguarded-envelope bug this whole mechanism exists downstream
of (task #295's own NaN, at step 38166). Confirmed by direct calculation
before touching the fix: n=128, clip=200, rank=32 gives a correction on the
order of 1e10, nowhere near "a small per-step nudge."

.. code-block:: python

   # WRONG (first version): raw-magnitude columns
   G = M.T @ M                     # entries ~ O(n * mag^2)
   M -= coef * M @ offdiag(G)      # correction ~ O(rank * n * mag^3) -- CUBIC

   # Fix: normalized (unit-direction) space
   norms = np.linalg.norm(M, axis=0, keepdims=True)
   U = M / np.maximum(norms, 1e-6)          # unit columns, provably finite
   gram = U.T @ U                            # cosine similarities, bounded [-1, 1]
   np.fill_diagonal(gram, 0.0)
   correction_dir = U @ gram                 # bounded by ~rank regardless of M's scale
   M -= coef * norms * correction_dir        # step LINEAR in each channel's own magnitude

Fix: do the whole computation in NORMALIZED (unit-direction) space, which
is also the conceptually correct fix, not just a numerical patch -- since
the intent was always "penalize direction, not magnitude," operating on
unit vectors makes that literal instead of incidental. ``gram`` has entries
bounded in [-1,1] (cosine similarities); the correction direction is then
bounded by ``O(rank)`` regardless of ``M``'s raw scale; the actual step
re-scales that bounded direction by each channel's OWN norm, so the update
stays LINEAR in magnitude, not cubic, while still moving faster for a
channel that's already larger. A near-zero-norm channel is guarded by
``max(norm, 1e-6)`` in the denominator -- this floor makes the whole
computation provably finite (Cauchy-Schwarz bounds every normalized
component by 1) rather than relying on a defensive ``nan_to_num``.

Gram's diagonal is zeroed (a channel's own norm isn't penalized, only its
correlation with OTHERS) -- the diagonal identity term (which would
additionally push every channel toward UNIT norm) is deliberately excluded,
since AQRS's own ``gamma_k`` already exists specifically to control channel
magnitude separately from direction -- including it here would fight
``gamma`` for the same job.

Cheap (``O(rank^2*n)``) since it only depends on the CURRENT parameter
values, not on any batch's ``dy``/``x`` -- unlike residual-targeted growth,
needs no new C++ state or hot-loop hook.

Known, accepted limitation: if two channels are ever EXACTLY identical (a
genuine float tie), the correction each receives is identical too
(scale-invariant), so this can only shrink an exact tie uniformly, never
truly separate it. Not expected to matter in practice: growth already seeds
new channels with independent random noise, so real channels essentially
never reach an exact float tie; this penalty's actual job is stopping the
ONGOING drift of already-distinct channels toward redundancy during
training, which it does regardless.

.. _default_rank_cap.byte_budget_derivation:

``_default_rank_cap``: deriving the rank ceiling from real storage cost
------------------------------------------------------------------------------

*ID:* ``default_rank_cap.byte_budget_derivation``

Derived from actual storage cost (direct instruction, superseding task
#295's cruder ``min(n_in,n_out)//4`` heuristic -- that formula wasn't
checked against real byte counts and, confirmed via a real 60k-step run,
let ranks grow well past the point where the low-rank envelope uses MORE
memory than the dense fp32 matrix it exists to avoid).

``value_scale``/``output_scale``/``additive_u``/``additive_v``/
``scale_gamma``/``additive_gamma`` are all stored as plain fp32 (no
quantization on the rank-N components themselves, confirmed in
``cpu_backend.cpp``/``delta_csr_types.hpp``) -- each rank-1 channel of the
scale branch costs ``4*(n_inputs+n_outputs+1)`` bytes (row vector + col
vector + 1 gamma scalar), same formula for the additive branch. The base
fp4-quantized weight matrix (weight+importance packed 2-per-byte) costs
``n_inputs*n_outputs*1`` byte regardless of rank.

The cap is the largest per-branch rank K such that BOTH branches growing to
K simultaneously still costs no more, combined with the base weights, than
a plain dense fp32 matrix of the same shape (``n_inputs*n_outputs*4``
bytes) -- i.e. AQRS is only ever memory-neutral-or-cheaper than "why not
just use dense fp32," never a net loss. Splits the remaining budget (after
the fp4 base) evenly between the scale and additive branches since both
draw from the same underlying justification and neither is structurally
more important than the other.

.. _seed_scale_rank.chicken_egg_deadlock:

``_seed_scale_rank`` / ``_seed_additive_rank``: breaking a zero-gradient deadlock, and why loop order matters
------------------------------------------------------------------------------------------------------------------------

*ID:* ``seed_scale_rank.chicken_egg_deadlock``

Both functions seed new AQRS rank channels with small random values because
components default to 0.0 on construction, and 0.0 on BOTH sides of a
product/sum is a genuine chicken-and-egg deadlock: neither factor can ever
get a nonzero gradient on its own unless at least one side starts nonzero.

For ``_seed_scale_rank``: per ``scale_rank``'s own docstring
(``delta_csr_types.hpp``), ``k>=1`` defaults to 0.0 on BOTH
``value_scale_k`` and ``output_scale_k``. ``rank=1`` is a no-op (nothing to
seed, matches the original single-component behavior exactly). Loop order
matters here, not just cosmetically: ``set_value_scale_raw_k``/
``set_output_scale_raw_k`` lazily resize+fill new slots with 1.0 (the
correct default for component 0), so as long as every row's ``k=0`` slot
gets touched by that fill BEFORE this function's own ``k>=1`` writes land
(true here: row-major, ``k`` ascending from 1, one index at a time -- each
row's ``k=0`` slot is always filled-by-resize immediately before this row's
own ``k=1`` write), every component-0 slot ends up correctly at 1.0 without
ever being written explicitly. Do not reorder these loops (e.g.
column-major, or ``k`` descending) without re-verifying that still holds.

For ``_seed_additive_rank``: same deadlock, additive instead of
multiplicative -- ``disldo_backward``'s ``dU_rk`` depends on ``dP``, which
is zero whenever every ``additive_v_k`` is zero (``dP[b,k] = sum_c
dy*V[c,k]``); symmetrically ``dV_ck`` depends on ``P``, which is zero
whenever every ``additive_u_k`` is zero. Unlike ``scale_rank``, there's no
``k==0`` "always on baseline channel" to preserve -- ``additive_rank==0``
is a true, fully transparent no-op (see ``disldo_forward``'s own
``if (weights.additive_rank > 0)`` guard), so every channel from ``k=0``
gets seeded, not just ``k>=1``.

.. _activate_gamma_tracking.opt_in_design:

``_activate_gamma_tracking``: why gamma tracking must be explicitly turned on
--------------------------------------------------------------------------------

*ID:* ``activate_gamma_tracking.opt_in_design``

Turns on EMA tracking for BOTH branches' gamma (task #292), which is what
makes ``apply_dynamic_rank_control``/``apply_additive_dynamic_rank_control``'s
Theorem 10 triggers actually evaluate against real, updating signal instead
of permanently-zero EMA state.

Gamma's own EMA only updates inside ``disldo_backward``'s gamma-update
block, itself gated on ``scale_gamma_is_trainable``/
``additive_gamma_is_trainable`` (opt-in flags) -- set true only once
``set_scale_gamma_raw_k``/``set_additive_gamma_raw_k`` has been called at
least once. Neither ``_seed_scale_rank`` nor ``_seed_additive_rank`` ever
touches gamma, so a layer constructed via those alone has gamma tracking
permanently OFF regardless of ``scale_rank``/``additive_rank`` -- this is
the explicit "opt in" step.

Writes ``gamma_k(0)=1.0`` for BOTH branches -- the exact same value
``get_scale_gamma_k``'s own lazy default already returns (transparent, zero
behavior change) -- so this call activates tracking WITHOUT perturbing
anything the layer was already computing. Additive branch only gets
activated if ``additive_rank>0`` (a rank-0 additive branch has no channel 0
to touch, and the neurogenesis trigger structurally can't fire from rank 0
anyway) -- a caller wanting the additive branch to ever grow under dynamic
control must seed it at rank>=1 up front, same as ``scale_rank``.

.. _preseed_dense.two_real_bugs:

``_preseed_dense``: two real bugs found verifying the dense-init preseed
------------------------------------------------------------------------------

*ID:* ``preseed_dense.two_real_bugs``

Fully dense counterpart to ``_preseed_random_sparse`` -- every
(input, output) pair connected, loaded straight into block4 via
``load_dense_codes``. Added per direct request to test whether this
project's usual random-SPARSE "echo network" preseed is itself a
significant source of seed-to-seed accuracy variance in toy comparisons
(reservoir-computing-style connectivity-draw sensitivity), independent of
whatever else is being compared.

**Bug 1**: init scale can NOT just be ``_preseed_random_sparse``'s own
``1/sqrt(k)`` fan-in scaling carried over naively into the RAW quantized
value. FP4 (E2M1) has a FIXED absolute zero-rounding floor (~0.25,
independent of layer width) -- at typical toy widths
(``state_width=128``, ``1/sqrt(128)~=0.09``), nearly EVERY drawn value
would quantize to code 0 ("not live"), silently collapsing "dense init"
back down to mostly-empty. Confirmed empirically: raw ``scale=0.3`` gave
only 23/64 live codes on an 8x8 test matrix; ``scale=1.5-2.0`` gave ~90%+
live.

**Bug 2**: using a FIXED raw scale (1.5) WITHOUT any compensating fan-in
correction elsewhere is ALSO wrong, just in the opposite direction --
confirmed empirically on the real curriculum: every one of ``base=4/6/12/24``
collapsed to pure CHANCE (``mean_acc~0.094``, ``std~0.003`` across 3 seeds),
identically regardless of ``base``. Root cause:
``output[c] = sum_r input[r] * weight[r,c]`` -- its variance scales with the
number of ROWS feeding column ``c`` (that column's fan-in), not the row's
own width. Sparse's ``k=5`` connections/row at scale ``1/sqrt(5)`` keep that
sum's variance ~1 (properly normalized); dense's ~128 connections/column at
raw scale 1.5 give variance ``~128*1.5^2~=288``, almost certainly
saturating whatever clip/RMSNorm sits downstream -- exactly explaining why
every dense arm gave identical numbers.

**Correction to the first fix** (per direct review): the first attempt
applied the correction via per-ROW ``value_scale`` scaled by
``1/sqrt(n_outputs)`` -- the WRONG axis. It only happened to produce a
working number in testing because q/k/v/o_proj are square
(``n_inputs==n_outputs``), making the wrong-axis correction numerically
coincide with the right one there; it would have been wrong for
``lm_head`` (16x10, not square). Also computed from the ASSUMED
``n_outputs``/``n_inputs``, not the REAL post-quantization live count --
wrong whenever bug 1's zero-code collisions leave a row/column short of
full density (confirmed ~87.5% actual density at scale=1.5, not 100%).

.. code-block:: python

   # Fix: real per-column live count from weight_codes (code 0 = not live),
   # fan-in-correcting output_scale PER COLUMN as metadata:
   #   true_w = stored_w * value_scale[row] * output_scale[col]
   live = weight_codes.reshape(n_inputs, n_outputs) != 0
   col_nnz = live.sum(axis=0)
   output_scale = 1.0 / (scale * np.sqrt(np.maximum(col_nnz, 1)))
   # value_scale (per-row) stays at its default 1.0 -- the correction
   # lives entirely on the column axis, matching the variance math above.

``output_scale[c] = 1/(raw_scale * sqrt(col_nnz[c]))`` reproduces the same
effectively-normalized weight distribution ``_preseed_random_sparse``
already produces for a k-wide row, generalized to each column's OWN real
fan-in rather than an assumed uniform width.

``_preseed_dense_scattered`` is the same idea for storage types with no
block4 support (``DeltaCSRBiValues<float>``/``DISLDOLayerV``) -- no
quantization floor to correct for, so it's just standard fan-in-normalized
(``1/sqrt(n_inputs)``) Gaussian init, matching ordinary Xavier/Kaiming
convention. Its caller MUST construct the layer with
``max_weights >= n_inputs*n_outputs`` -- unlike block4's dense path (a
separate allocation, unbounded by the scattered-CSR ``max_weights``
budget), this genuinely needs that much scattered-CSR storage.

.. _preseed_empty.design_vs_zero_grid:

``_preseed_empty``: the genuinely-designed zero-weight-init
------------------------------------------------------------------

*ID:* ``preseed_empty.design_vs_zero_grid``

NO connections at all (``nnz=0``), not a dense grid pre-loaded with
weight=0 (see ``_preseed_random_sparse`` -- "a freshly-constructed
``SparseLinearLayer`` has zero connections... this is purely an
optimization for a faster bootstrap, not a requirement"). Real synapses are
meant to be created by ``synaptogenesis()`` (``build_probes``/
``synap_step``/``equalizer_step``), each starting with a real nonzero value
the first time it's grown in -- not by gradient-nudging a pre-existing
zero.

Per direct correction: pre-loading weight=0/importance=1 into every slot
(this project's earlier ``all_zero_init`` mechanism) is a different,
synthetic experimental arm, not this one -- it has its own failure mode (an
eps-shaped backprop term can decay that seeded importance back toward 0)
that doesn't apply here since nothing exists to decay.

Only reserves per-row growth headroom via ``equalize_to_capacity`` -- safe
to call directly on a fresh, never-``load_weights``'d layer (unlike after
``_preseed_random_sparse``'s ``load_weights`` call, which would immediately
compact any headroom back away).

.. _graded_top_k_csr.python_to_cpp_migration:

``_graded_top_k_csr``: moved from a Python loop to a C++ kernel after a real slowdown
------------------------------------------------------------------------------------------

*ID:* ``graded_top_k_csr.python_to_cpp_migration``

GENUINELY per-row top-k selection: row r independently keeps its own top
``k_per_row[r]`` largest-magnitude entries. Lets a caller grade gradient
density by row (e.g. by how far back in time a row's content is, for
``step_cached``'s query-step credit-assignment design).

Was a pure-Python/numpy per-row ``argpartition`` loop -- moved to a real
C++ kernel (``_cpu.dense_to_graded_top_k_csr``, ``csr.hpp``'s
``top_k_csr_graded``) after a live 20k-step curriculum comparison showed
this loop running on every query/backward step dominated the step's own
cost: ``use_tile_cache=1`` (which relies on this function) measured SLOWER
(5.1 steps/sec) than the plain ``step()`` baseline (6.6 steps/sec) instead
of the expected speedup from caching. This Python wrapper is kept only so
existing callers/tests (that expect this exact name/signature) don't need
to change.

Found the hard way: this is NOT equivalent to
``_cpu.dense_to_top_k_csr(dy2d, k, cpus)`` with a uniform ``k``, even at
matching average density. That function's ``k`` is spent GLOBALLY across
the WHOLE flattened ``rows*cols`` array (via ``top_k_indices(values,
rows*cols, k, ...)`` in ``csr.hpp``), not per row -- a row can end up with
zero surviving entries there, purely by losing the global competition, no
matter what ``k`` is requested. Don't "verify" this function by comparing
its output against ``dy_sparsity_p``'s existing path; they answer different
questions.

.. _nucleus_top_k_csr.r_target_math:

``_nucleus_top_k_csr``: energy-threshold top-k, k as a consequence not a constant
----------------------------------------------------------------------------------------

*ID:* ``nucleus_top_k_csr.r_target_math``

Row r independently keeps the SMALLEST set of its own top-``|v|`` entries
whose captured squared-magnitude ratio
``R(v,k) = sum(v_topk^2)/sum(v^2)`` is ``>= r_target[r]``. ``k`` is a
CONSEQUENCE of ``r_target`` and the row's own data, not a fixed constant --
same math as truncated-SVD captured-variance / LLM nucleus (top-p) sampling
applied to squared magnitude instead of softmax probability. See
sili_peridot's JOURNAL.md "nucleus/energy-threshold top-k math" design note
for the full derivation.

``k_min``/``k_max``: hardware-driven density floor/ceiling applied to the
``R_target``-derived ``k`` AFTER the fact (direct instruction --
``R_target`` alone can degenerate to ``k=0``, a fully-dead row, or to
near-100% density on hardware that wants a bounded ceiling). ``k_min``
padding pulls from the same magnitude-sorted order the ``R_target``
selection already computed, so a clamped row degrades to plain top-k rather
than picking arbitrarily; an all-zero row can't manufacture ``k_min``
entries out of nothing and stays at ``k=0`` regardless.

.. _disldo_layer_forward.design_notes:

``DISLDOLayer.forward``: five small design decisions bundled into one call site
---------------------------------------------------------------------------------------

*ID:* ``disldo_layer_forward.design_notes``

- ``min_decay_frac``/``max_abs_delta``/``max_ci``: ``None`` (default) means
  "use the C++ side's own tuned production defaults" -- kept ``Optional``
  here rather than hardcoding the production floats a second time in
  Python, so there is still exactly one source of truth
  (``cpu_backend.cpp``'s ``kSynapsePolicy*`` constants) even though this
  wrapper exposes them for quick per-call experiments (e.g. testing
  ``max_abs_delta=huge`` to effectively disable the clip) without a C++
  rebuild.
- ``forward_dense``/``backward_dense`` always return ``[batch, cols]`` --
  even for a bare 1-D ``[cols]`` input, batch is implicitly 1, but the
  OUTPUT shape stays 2-D regardless. Squeezed back to 1-D when the input
  was 1-D so a single online sample round-trips to 1-D, matching this
  class's own no-batch-dimension docstring; genuinely 2-D (batched) input
  is left alone.
- ``lr_per_row_nnz`` default ``True`` preserves prior behavior (was
  hardcoded) -- exists at all because it was previously unreachable from
  this wrapper: ``learning_rate`` silently meant
  "``learning_rate / row_degree``," not the literal rate passed in. That
  normalization exists to keep aggregate row updates comparable when
  synaptogenesis makes degree vary WITHIN a layer -- at uniform density (no
  synaptogenesis, or a deliberately dense/parameter-matched layer) it does
  nothing but silently shrink the effective rate by the row's degree. Pass
  ``False`` for a literal, degree-independent learning rate.
- ``forward_dense`` no longer takes ``learning_rate`` at all -- it used to
  run its own gradient-free ADSP-style (Activity-Dependent Structural
  Plasticity) importance update on every call, independent of whether a
  ``backward()`` would ever follow. Real weight/importance updates now
  happen ONLY in ``backward_dense``/``backward_sparse``.
- CSR-typed input (task #334/Phase 5): NOT a new ``sparse_input`` bool --
  the caller controls forward-input density entirely by whether ``x.data``
  IS a CSR (``Tensor.is_csr``), matching the existing precedent at
  ``SparseRNNCell.forward()`` in the opposite direction. A dense
  ``np.ndarray``/``Tensor`` here is a completely unchanged code path --
  zero behavior change for every existing caller who never builds a CSR.

.. _disldo_layer_forward.dy_sparsity_schedule_and_nucleus_grad:

Gradient sparsification variants: ``dy_sparsity_schedule`` and ``dy_r_target``
--------------------------------------------------------------------------------

*ID:* ``disldo_layer_forward.dy_sparsity_schedule_and_nucleus_grad``

``dy_sparsity_p`` (task #334/Phase 5) is a genuinely independent axis from
``x``'s own type (``disldo_backward_sparse_grad``'s own signature takes a
dense ``x`` + sparse ``dy``). ``None`` (default) keeps the exact dense-``dy``
behavior via ``backward_dense``.

``dy_sparsity_schedule``: per-row graded density (task: query-step credit
assignment for ``step_cached``) -- overrides the scalar ``dy_sparsity_p``
when both are given. ``len(dy_sparsity_schedule)`` must match the batch's
row count; each entry is that row's own density fraction (same convention
as ``dy_sparsity_p``, one value per row instead of one for the whole call).

``dy_r_target``: nucleus/energy-threshold grad sparsification (task #367,
priority 1 per direct instruction -- "grad is the one that's definitely
required"). ``k`` is a CONSEQUENCE of ``dy_r_target`` and this step's actual
gradient energy, not a fixed fraction. See sili_peridot/JOURNAL.md's
nucleus design note, and ``nucleus_top_k_csr.r_target_math`` above for the
underlying math.

.. _disldo_layer.apply_dynamic_rank_control_theorem10:

``apply_dynamic_rank_control``: Theorem 10, and the biological basis for per-branch grace periods
------------------------------------------------------------------------------------------------------------

*ID:* ``disldo_layer.apply_dynamic_rank_control_theorem10``

AQRS Theorem 10 dynamic rank control (task #292) -- evaluates BOTH
branches' apoptosis/neurogenesis triggers against their own EMA state
(updated automatically inside ``backward_dense``/``backward``, see
``disldo_backward``'s own gamma update block) and performs at most one
mutation PER BRANCH per call (so up to two total: one multiplicative, one
additive). Call once per training step, after backward -- matches
``test_aqrs_dynamic_rank_control_integration.cpp``'s own "breathing"
integration test convention exactly, just from Python.

``theta``'s default (1e-4, task #294 fix) is tuned against the gradient
normalized by layer size (``n_in*n_out``) -- NOT the old pre-normalization
raw scale (which used to require ``theta~0.02`` and made the trigger
meaningless across differently-shaped layers, since a 128x128 layer's raw
gradient could be 9 orders of magnitude larger than a 16x128 layer's for
the same real signal).

``scale_grace_period_steps``/``additive_grace_period_steps``: separate
PER-BRANCH cooldowns (direct instruction, replacing an earlier
within-branch grow-vs-shrink asymmetry after a biology literature check --
see ``apply_dynamic_rank_control_generic``'s own comment in
``delta_csr_types.hpp`` for the full citations). Each branch uses the SAME
value for its own grow/shrink internally (symmetric within a branch,
matching the roughly comparable formation/elimination rates seen in real
dendritic spine turnover -- Holtmaat et al., Neuron 2005; Grutzendler et
al., Nature 2002).

The real asymmetry is CROSS-branch: the scale (multiplicative) branch is
the per-synapse analog and runs fast (default 50), while the additive
branch is a whole-layer correction structurally closer to a neuron
integrating its own aggregate state -- the biological analog of
homeostatic/intrinsic plasticity, which needs ~24-48h of sustained change
to manifest vs. seconds-to-minutes for Hebbian/STDP synaptic change
(Turrigiano and colleagues' classic activity-blockade experiments; see also
Zenke & Gerstner, "Hebbian plasticity requires compensatory processes on
multiple timescales," Phil. Trans. R. Soc. B, 2017) -- roughly a
100x-3000x separation depending which endpoints are compared.
``additive_grace_period_steps`` defaults to 5000 (100x scale's 50), the
conservative end of that range so the additive branch can still mutate
within realistic training budgets; both values are real, independently
tunable parameters meant to be swept against real training outcomes, not
treated as settled by this docstring.

.. _disldo_layer_variants.diagnostic_history:

The ``DISLDOLayer*`` variant zoo: each class isolates one variable
------------------------------------------------------------------------

*ID:* ``disldo_layer_variants.diagnostic_history``

Several ``DISLDOLayer`` subclasses/siblings exist purely to isolate one
variable at a time while chasing a real FP4/FP8 training gap found in
sili_peridot's toy quantization sweeps -- none of these are meant as the
"production" choice by default:

- ``DISLDOLayerResync`` / ``DISLDOLayer8Resync``: apply the
  DeferredScaleWrite fix (touched entries' stored codes are written out
  only after ``value_scale``/``output_scale`` are both finalized for the
  ``backward()`` call, instead of immediately under the stale pre-update
  scale). Plain ``DISLDOLayer``/``SparseLinearLayer`` used the DEFAULT
  ``ScalePolicy``/``DeferredScaleWrite=false`` template args (the same
  stale-code path ``SparseLinearLayer8Resync`` was built to fix for FP8),
  never updated when that fix landed for FP8 -- see
  sili_peridot/JOURNAL.md's ``TrueMultiDigitLayer`` entry: real FP4
  per-digit training collapsed to chance while an architecturally identical
  fp32-shadow control succeeded by a wide margin, which is what prompted
  checking whether FP4 had this same staleness bug too.
- ``DISLDOLayerNoScale``: ``value_scale``/``output_scale`` permanently
  forced to 1.0 -- never trained, nothing to go stale, a direct hardware
  test of the "zero trained scale" design instead of just fixing the
  staleness bug.
- ``DISLDOLayerDeterministic``: same RMSprop scale handling as plain
  ``DISLDOLayer``, but weight/importance storage uses deterministic
  nearest-neighbour rounding (``fp4_quantize``) instead of stochastic
  dithered rounding (``fp4_quantize_stochastic``) -- built after
  ``DISLDOLayerResync``/``NoScale`` both still collapsed to chance on
  sili_peridot's out-of-context curriculum, to test whether real FP4's
  per-step stochastic rounding (not ``value_scale`` staleness) is what the
  deterministic-rounding fp32-shadow controls were actually avoiding.
- ``DISLDOLayerResyncDeterministic`` / ``DISLDOLayerNoScaleDeterministic``:
  combinations of the above, isolating each fix independently.
- ``DISLDOLayer32``: same ``disldo_forward``/``disldo_backward`` kernels,
  generic over ``VALUES_TYPE``, instantiated with the 32-bit float fallback
  (``DeltaCSRBiValues<float>``) instead of 4-bit ``FP4BiPacked`` -- isolates
  what FP4's coarseness specifically contributes vs. the update-rule math
  itself. Not a production layer: no ``equalize_to_capacity``, no block4
  promotion. Its ``dense=True`` uses ``_preseed_dense_scattered`` because
  this ``VALUES_TYPE`` has no block4 support, so it genuinely needs
  ``max_weights`` expanded to cover every (input, output) pair -- done
  automatically. Added because the DEFAULT (``_preseed_random_sparse``,
  ~``k=max_weights//(2*n_inputs)`` connections/row) silently gave this
  class far fewer trainable parameters than a ``dense=True`` FP4/FP8 arm or
  a fully-connected torch reference at the same ``max_weights`` budget -- a
  genuine capacity gap, not a precision or optimizer difference, confirmed
  as a real contributor to fp32's poor MQAR results before this existed.
- ``DISLDOLayer8``: real 8-bit storage (``SparseLinearLayer8``, OCP MX
  E4M3 per-value codec, ``fp8quant.hpp``). Combined with the existing
  rank-1 ``value_scale``/``output_scale`` mechanism, this is the concrete
  "8-bit + rank-1 scale, weight AND importance both quantized" scheme
  validated in sili_peridot's toy-model quantization sweep (three task
  families, ten configs, never lost to native FP4). Its ``dense=True`` uses
  the block4 dense-tile SIMD path (``Block4Tile8``/``Store8``, task
  #94/#96) -- this class's docstring previously said "no block4 dense-tile
  SIMD promotion yet" because the PYTHON wrapper never called it, even
  though the C++/pybind side (``SparseLinearLayer8.load_dense_codes``) has
  been ready since #123. ``scale_rank`` was never threaded into its
  constructor before -- confirmed missing (unlike ``DISLDOLayer``'s own
  ``_seed_scale_rank`` call) while wiring up a real-engine rank1/rank2
  sweep across fp4/fp4_dual/fp8 (sili_peridot task #247); a rank2 FP8 arm
  would otherwise ``TypeError`` immediately. ``additive_rank`` is the
  branch task #280 re-validates against the fp8 MQAR
  input-independent-collapse ("mumbling") case -- see AQRS_DESIGN.md
  Theorem 3/4.
- ``DISLDOLayer8AdaMax``: same as ``DISLDOLayer8Resync`` (DeferredScaleWrite
  also on), but ``value_scale``/``output_scale`` use an AdaMax-style
  decayed running-max update instead of RMSprop.

``apply_dynamic_rank_control``/``apply_scale_overflow_guard``/
``apply_channel_orthogonality_penalty`` are duplicated on ``DISLDOLayer8``
rather than shared via inheritance, since ``DISLDOLayer8`` doesn't subclass
``DISLDOLayer`` (separate ``VALUES_TYPE`` entirely).

.. _sisldo_layer.dense_x_reconstruction_not_last_input:

``SISLDOLayer``: why backward reconstructs dense ``x`` instead of reading ``last_input``
-------------------------------------------------------------------------------------------------

*ID:* ``sisldo_layer.dense_x_reconstruction_not_last_input``

Forward exploits sparse ACTIVATIONS (``forward_sparse``, skips inactive
input rows). Backward exploits a top-k'd sparse GRADIENT
(``backward_sparse``) -- these are independent axes on ``SparseLinearLayer``,
not a matched forward/backward pair (see its class comment in
``cpu_backend.cpp``): ``dx`` doesn't depend on the input's own sparsity,
only on weights and ``dy``, so ``backward_sparse``'s required dense ``x``
argument is reconstructed from the same CSR ``forward_sparse`` was called
with (``csr.to_dense()`` -- zeros in the dropped positions), NOT read from
``self._c.last_input``.

Real bug this avoids: ``forward_sparse`` never populates ``last_input``
(only ``forward_dense`` does, see ``cpu_backend.cpp``) -- an instance that
never called ``forward_dense`` would otherwise silently compute its weight
update against stale/empty data. Task #333 fixed this by threading the real
dense ``x`` explicitly through the closure instead (same fix applied to
``DISLDOLayer.forward``).

``learning_rate`` is kept in this wrapper's own signature (matches every
other layer's ``forward()`` call convention throughout this project) but no
longer forwarded to ``forward_sparse`` -- it used to run its own
gradient-free ADSP-style importance update on every call, independent of
whether ``backward()`` would ever follow. Real weight/importance updates
now happen ONLY in ``backward_sparse``.

.. _fit_rank1_scale_envelope.coo_no_densify:

``fit_rank1_scale_envelope``: COO triplets, never a dense intermediate
------------------------------------------------------------------------------

*ID:* ``fit_rank1_scale_envelope.coo_no_densify``

Alternating max-fit producing a rank-1 (outer-product) envelope:
``row_scale[r] * col_scale[c] >= |M[r, c]|`` for every entry, using
``O(rows+cols)`` parameters instead of ``O(rows*cols)``. Each update
recomputes the exact bound needed against the other side's current
estimate (Sinkhorn-style, but max- rather than sum-based).

Takes ``M``'s nonzero entries as COO-style triplets
(``row_idx, col_idx, abs_vals``), not a dense matrix -- a folded/stacked
layer's matrix is too large to densify safely (e.g. ``mlp.gate_proj``
folds to ``[110592, 1536]``; converting that to dense once already risked
OOM on a real conversion run). Entries not listed are implicitly 0 and
never bind the max, so this needs no dense intermediate at all.

Returns float32, deliberately not float64 (a max/divide fit has no
accumulating-sum precision need, and float64 here would double the
``O(nnz)`` working set for no benefit).

.. _folded_layer.csr_layout_conversion_no_densify:

``FoldedLayer.from_descriptor``: CSC-to-CSR relabeling, and why ``.to_dense()`` is never called
------------------------------------------------------------------------------------------------------

*ID:* ``folded_layer.csr_layout_conversion_no_densify``

``csr.t()`` is a metadata-only relabelling into CSC (no densify, no
nnz-proportional copy) -- but CSC's own ``ccol_indices``/``row_indices``
are grouped by COLUMN, not the per-ROW grouping ``load_weights`` needs.
``.to_sparse_csr()`` does the real (but nnz-proportional, not
densify-proportional) reorganization into row-major order.

NEVER call ``.to_dense()`` on a stacked/folded layer's matrix -- it can be
too large to safely materialize (e.g. ``mlp.gate_proj`` folds to
``[110592, 1536]``; densifying that alone risked OOM converting a real
checkpoint). This is why the conversion path routes through
``t().to_sparse_csr()`` instead.

Budget for the delta-CSR pool is sized for the fully-connected maximum
(``n_in * n_out``), not for current nnz. This is the fixed total the
staggered ``equalizer_step()`` will redistribute within -- equalization
only moves bytes between rows, never grows the pool. Sizing for
``n_in*n_out`` guarantees every row can hold ``n_out`` connections after a
full equalization pass, the absolute ceiling for any ``max_row_weights``
value.

``value_scale_mode="rank1"`` fits one scale per input row AND one per
output column directly from the nonzero triplets (no dense intermediate --
see ``fit_rank1_scale_envelope.coo_no_densify``). ``"per_row"`` (default,
exact prior behavior) maps each row's max-abs to FP4_MAX for full
quantizer resolution, leaving ``output_scale`` at its default 1.0.

Per-row ``importance_scale`` addresses the same FP4 representability
problem as ``value_scale`` but for importance: importance is updated via
activity correlation in ``forward_dense`` (magnitude ``~ |x| * |h| * lr ~
lr`` after value scaling), and FP4's minimum nonzero is 0.5, so a raw
update of ``lr=0.01`` rounds to 0. Setting
``importance_scale = lr / FP4_MAX`` maps FP4 range to ``[-6*lr, +6*lr]``,
making importance updates of order ``lr`` representable from the first
step. Weight VALUES are NOT changed in ``forward_dense``; they are updated
only by ``backward_dense()`` via the task gradient.

``compact_after_build=True`` (default) strips ``equalize_to_capacity``'s
per-row growth headroom right after loading -- measured on a real
checkpoint: dropped 7 suffixes' combined RSS from ~1.9x to ~1.08x the
2-bytes/param theoretical minimum. Call ``expand_headroom_to()`` on the
built layer before synaptogenesis needs room to grow again.

.. _sparse_rnn_cell.overview:

``SparseRNNCell``: unifying two sparsification passes, and recurrent-only branching measurement
--------------------------------------------------------------------------------------------------------

*ID:* ``sparse_rnn_cell.overview``

Returns ``(h_out: Tensor, aux_loss: Tensor, actual_p: float)`` -- ``h_out``
(dense) is returned unchanged as the new state, so argmax/save/inspection
on it keep working without modification.

**Unifying the sparsification passes** (see
``sili.energy._apply_energy_dynamics``'s ``kept_indices`` docstring, and
``apply_energy_dynamics.kept_indices`` in ``docs/research/energy.rst``):
the CSR fed into ``recurrent()`` at the top of the NEXT call is built from
THIS call's own energy-gating decision (``kept_indices`` + the PRE-gating
``h`` values at those indices) rather than an independent top-k
re-derivation that could disagree with it. That decision is cached on the
cell (``_prev_kept_indices``/``_prev_h_dense``) rather than smuggled
through the state Tensor itself, since ``state.data`` must stay dense for
argmax/save to keep working, and CSR must not be built from
``state.data``'s own values anyway -- ``state.data`` (``h_out``) has
fire/shutoff positions flattened to energy-derived constants (``2.0``,
``e+2``), not the real activation magnitude the gate decided to keep. The
cache is invalidated on ``reset()``/whenever the caller hands in a state
the cell didn't itself just produce (e.g. after ``SparseRNNAgent.load()``),
falling back to ``CSR.from_dense`` (the true step-0 path) exactly once
until the cell has run again.

**Branching-ratio measurement** (see
``branching_ratio_tracker.model`` in ``docs/research/energy.rst``):
recurrent-only activity is measured on ``recurrent_out`` BEFORE it's summed
with ``input_proj(obs)`` -- measuring on the combined ``h`` cannot
distinguish a genuinely self-propagating recurrent pathway from fresh input
alone carrying activity while the recurrent branching factor is silently 0.
``branching_tracker`` selects which estimator backs
``self.branching_recurrent``: ``"window"`` (a hard sliding window, also the
only one that supports ``avalanche_sizes()`` for a SOC power-law-tail
check) or ``"ema"`` (default, O(1) memory, exponentially-discounted --
prefer this for a continuously-updated read with a tunable fast/long-term
tradeoff via ``branching_ema_alpha``, e.g. for
``dynamic_density_from_branching_ratio`` reacting promptly to a regime
change). Want both a fast EMA read and the avalanche-size check at once?
Construct a second tracker yourself and feed it the same ``recurrent_out``
activity this cell already computes each step -- not built into this
class, since which additional trackers (if any) matter is a caller
decision.

.. _sparse_rnn_cell.density_p_inversion_bug:

Real bug: ``density`` and ``p`` were swapped in ``SparseRNNCell.__init__``
--------------------------------------------------------------------------------

*ID:* ``sparse_rnn_cell.density_p_inversion_bug``

``density`` is the target active fraction; ``p`` is a hard compute-limit
ceiling that must sit clearly above it (~5x here), not the thing that
shapes learned sparsity -- see ``EnergyDynamics``'s own
``density <= p * 0.8`` assertion (``apply_energy_dynamics.p_vs_density`` in
``docs/research/energy.rst``) for why.

.. code-block:: python

   # WRONG (the actual bug this replaced): density derived FROM
   # percent_active*0.9 while p was set TO percent_active directly --
   # density could exceed p*0.8, violating EnergyDynamics's own invariant.
   density = percent_active * 0.9
   p = percent_active

   # Fix: density IS the target active fraction; p is the ceiling, set
   # clearly above it.
   density = min(0.9, percent_active)
   p = min(1.0, percent_active * 5.0)

``activation_cost=0.08*r`` grows unbounded with ``percent_active`` (``r``
scales linearly with it) -- fine for the small ``percent_active`` this
formula was tuned around, but a small state legitimately wants a much
higher ``percent_active`` to get more than 0-1 active neurons (e.g.
``state_size=12``), which pushes ``activation_cost`` past
``EnergyDynamics``'s own asserted ``[0.01, 0.5]`` range. Clamped rather
than letting construction fail for that (valid) regime.

.. _sparse_rnn_cell.dynamic_density_nudge:

``dynamic_density_from_branching_ratio``: a first-cut proportional nudge, not a derivation
------------------------------------------------------------------------------------------------------

*ID:* ``sparse_rnn_cell.dynamic_density_nudge``

Optional (default off) use of the measured recurrent branching ratio to
nudge the KL density target -- a first-cut proportional adjustment, not a
first-principles derivation; see energy-params.md. Centered on the intended
near-critical band ``[0.97, 0.99]`` (``m_target = 0.98``). Bounded to
``+/-2x`` the base density so a noisy early estimate can't send the target
somewhere degenerate. "window" tracker is the default elsewhere in the
class so existing behavior/callers are unaffected by this feature's
addition; "ema" trades the ``avalanche_sizes()`` check away for O(1) memory
and a tunable fast/long-term response.

.. _sparse_rnn_agent.build_probes_k_scaling:

``synaptogenesis_k``: why the default is small (``build_probes`` is O(k^2))
--------------------------------------------------------------------------------

*ID:* ``sparse_rnn_agent.build_probes_k_scaling``

``build_probes(k)`` generates ``min(k, n_in) * min(k, n_out)`` candidate
pairs (top-k input rows outer-producted against top-k output columns) --
O(k^2), not O(k). Measured directly: on a 1000x1000 layer, ``k=64`` can
grow ``nnz`` from ~0 to 1000 (full density) in a SINGLE ``synap_step``
call, vs. ``k=4`` growing ``nnz`` by only ~16 -- since this runs every
online step (see ``sparse_rnn.module_overview``), a large ``k`` saturates
connectivity almost immediately rather than growing gradually. ``k=4``
(default) or smaller is far more sane for continuous per-step growth; only
raise this if you've checked the resulting per-step ``nnz`` growth against
your actual ``max_weights`` budget and step rate.
