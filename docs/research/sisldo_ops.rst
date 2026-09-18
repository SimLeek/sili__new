``sisldo_ops.hpp`` research notes
====================================

Companion doc to ``sili/lib/headers/sisldo_ops.hpp``. Source comments point
back here by anchor ID (``*ID:*`` marker under each heading below); this doc
links back to source by file/symbol name. See ``docs/research/linear_disldo.rst``
and ``docs/research/delta_csr_types.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers, frozen code snippets on
real-bug/non-obvious-derivation sections) -- ``sisldo_ops.hpp`` is the
sparse-input-CSR counterpart to ``linear_disldo.hpp``'s dense-input path, so
many sections below are direct ports; where that's the case, this doc says so
and points at the shared mechanism's own writeup rather than duplicating it.

.. _sisldo_ops.file_split_context:

File overview: split out of ``sparse_struct.hpp``
------------------------------------------------------

*ID:* ``sisldo_ops.file_split_context``

Whole-structure memory operations (``compact``/``expand_headroom`` --
opposite operations, see their own sections below) and the actual
forward/backward computation: ``sisldo_forward`` (SISLDO -- sparse input)
and ``disldo_backward_sparse_grad`` (dense input, sparse gradient --
deliberately NOT sparse input; see
``disldo_backward_sparse_grad.dense_input_rationale`` below for why a
sparse-input backward permanently loses the ability to correct "didn't fire
& should have").

.. _compact.headroom_removal:

``compact()``: repack to zero inter-row blank space
----------------------------------------------------------

*ID:* ``compact.headroom_removal``

Repacks a ``DeltaCSRWeights`` so every row occupies exactly its active
bytes/elements, zero inter-row blank space -- both the index buffer
(``byte_start``/``byte_end``) AND the values buffer (``elem_start``/
``elem_end``) are separate growth-headroom axes and both get compacted.

``delta_csr_from_absolute()``'s reserved headroom is correct and necessary
for a LIVE, training model -- rows need O(1) append room for
synaptogenesis. For a freshly converted or long-since-pruned model being
saved/measured for deployment, that headroom is pure unused padding that
``nnz()``/``total_alloc_bytes()`` otherwise count as consumed. Use
``compact()`` before saving/measuring; call ``reserve_indices()``/
``reserve_values()`` again after loading if the model is about to resume
training rather than just be measured or deployed.

Generic over ``VALUES_TYPE`` via ``ValueAccessor`` -- one implementation for
both ``FP4BiPacked`` and ``DeltaCSRBiValues<float>``, matching the rest of
this file's pattern (``sisldo_forward``/``disldo_backward_sparse_grad``).

**Test requirement**: must be lossless -- decode every synapse from the
input and the output (column indices via ``row_cursor``, weight/importance
via ``ValueAccessor::get_w``/``get_imp``), compare row by row, must match
exactly. Also verify ``total_alloc_bytes()``/``total_alloc_elems()``
strictly decrease (or stay equal) after compacting a
``delta_csr_from_absolute()``-constructed layer, and that a second
``compact()`` call is idempotent (sizes unchanged).

.. _expand_headroom.budget_propagation_bug:

``expand_headroom()``: restoring headroom without losing the caller's budget cap
--------------------------------------------------------------------------------------

*ID:* ``expand_headroom.budget_propagation_bug``

Opposite of ``compact()``: restores growth headroom to a ``DeltaCSRWeights``
that has none (or not enough) -- typically because ``compact()`` removed
it. Reuses ``delta_csr_from_absolute()``'s already-tested
headroom-reservation logic (extract to absolute CSR via
``delta_csr_to_absolute``, then rebuild) rather than duplicating it.
``blank_fraction`` is the SAME parameter ``delta_csr_from_absolute`` takes
-- 0.2 (20%) restores the same headroom a freshly-converted layer gets by
default; pass a larger value before a synaptogenesis-heavy phase, smaller
if memory is tight and only modest growth is expected.

**Behavior note**: ``expand()`` NORMALIZES headroom to exactly
``blank_fraction`` of current content size -- it does not add
``blank_fraction`` on top of whatever headroom the input already had
(``delta_csr_to_absolute`` extracts only the actual synapses, not existing
slack, so there's nothing to add to). Calling ``expand()`` on an
already-roomy layer with a smaller ``blank_fraction`` than it currently has
will shrink its headroom, same as ``compact()`` would, just not all the way
to zero -- consistent with ``compact()`` normalizing to exactly 0% and
``expand()`` normalizing to exactly ``blank_fraction``, not "at least
``blank_fraction``."

**Real bug, fixed**: the input ``dc``'s own hard limits
(``max_indices_bytes``/``max_values_bytes``) must be propagated through --
without this, the freshly-constructed result starts with DEFAULT
(unbounded) limits, so it can grow arbitrarily past whatever budget the
caller originally set via ``set_limits()``, regardless of how small
``n*(1+blank_fraction)`` is relative to it. Measured: ``nnz`` reached 127x
the intended ``max_weights`` budget in a synaptogenesis stress test before
this fix, since repeated ``expand_headroom()`` calls each silently
re-based the cap on current content instead of the original budget.

**Test requirement**: after ``compact()`` then ``expand()``,
``row_rebuild``/``synap_row_step`` must succeed on rows that failed
immediately post-compact -- this is the actual bug this function exists to
let callers work around ("silent failure is the worst case"). Also verify
``expand()`` is lossless (same content as ``compact()`` already checks).

.. _expand_headroom_to.per_row_budget:

``expand_headroom_to()``: sizing for a minimum per-row connection count
------------------------------------------------------------------------------

*ID:* ``expand_headroom_to.per_row_budget``

Like ``expand_headroom()`` but sizes the total budget for at least
``min_nnz_per_row`` connections per row. Use before synaptogenesis on a
freshly loaded layer, then call ``equalizer_step()`` for each row to
redistribute the budget evenly -- after a full equalization pass each row
has ``total_budget/rows = min_nnz_per_row`` elements of reserved headroom.

Carries the identical hard-limit-propagation fix as
``expand_headroom.budget_propagation_bug`` above: if
``min_nnz_per_row*rows`` genuinely exceeds the layer's original
``max_weights`` budget, this now correctly throws ``std::bad_alloc``
instead of silently granting more than was ever configured.

.. _sisldo_forward.no_learning_rate_param:

``sisldo_forward``: no ``learning_rate`` parameter, deliberately
-------------------------------------------------------------------

*ID:* ``sisldo_forward.no_learning_rate_param``

Matches ``disldo_forward``'s own fix (see
``docs/research/linear_disldo.rst`` for the full rationale): this used to
run a gradient-free ADSP-style (Activity-Dependent Structural Plasticity)
importance update whenever a nonzero ``learning_rate`` was passed,
unconditionally on whether a matching backward call would ever follow.
Real footgun, confirmed via direct tracing on the DISLDO sibling --
REMOVED here too, not just disabled. Importance updates only ever happen
in a backward pass now, coupled to a real gradient.

.. _sisldo_forward.dc_empty_block4_bug:

``sisldo_forward``: ``dc.empty()`` does not mean "nothing to do"
-------------------------------------------------------------------

*ID:* ``sisldo_forward.dc_empty_block4_bug``

``dc.empty()`` (zero scattered nnz) does NOT mean "nothing to do" -- a
layer can be entirely block4-resident (``dc.empty() == true``) and still
have real work in the block4 phase further down. Historically this
function returned here unconditionally, which silently skipped the block4
phase too for any all-block4 layer.

**Found via**: a benchmark reporting an implausible, density-independent
~0.0001ms for the sparse path -- it was hitting this return before ever
reaching block4 code. ``L``'s rows/cols come from ``dc.layout``'s shape,
which stays valid even when nnz is 0 (set at construction, e.g.
``delta_csr_from_absolute``), so it's safe to keep using ``L`` regardless
of ``dc.empty()``. The identical fix applies to
``disldo_backward_sparse_grad`` (see below).

.. _sisldo_forward.output_scale_read_bug:

``sisldo_forward``: rank-1 output scale silently dropped (real bug)
--------------------------------------------------------------------------

*ID:* ``sisldo_forward.output_scale_read_bug``

**Real bug, found and fixed**: ``out_scale``/``output_importance_scale``
(per-column, e.g. from ``FoldedLayer.from_descriptor``'s
``value_scale_mode="rank1"``) were never read here, unlike
``disldo_forward``'s identical row*col combination -- a
rank-1-quantized layer run through ``forward_sparse`` silently dropped its
column scale entirely, reconstructing only ``stored_w * val_scale`` instead
of the true value.

Scale lookups are per-synapse here, not hoisted: ``in_idx`` (the row)
varies within this loop (work-offset iteration, not a simple per-row loop)
-- unlike ``disldo_forward``/``disldo_backward``, this can't be fixed once
per outer iteration. The replacement uses rank-N scale (see
``sparse_linear_weights_delta.scale_rank_rationale`` in
``docs/research/delta_csr_types.rst``), which reduces to the exact original
``val_scale*out_scale`` at ``scale_rank==1``, matching ``disldo_forward``'s
identical replacement.

.. code-block:: cpp

   // as of PR #45, sisldo_ops.hpp -- sisldo_forward's scattered inner loop:
   const value_type wval = wval_stored * weights.get_scale(in_idx, out_idx);
   // WRONG (the bug, now fixed): reconstructing via wval_stored * val_scale
   // alone silently drops out_scale/output_importance_scale entirely for
   // any rank1-quantized layer.

.. _sisldo_forward.block4_gather_design:

``sisldo_forward``: block4 contribution -- Design A window gather
------------------------------------------------------------------------

*ID:* ``sisldo_forward.block4_gather_design``

**Real, previously-silent bug this section closes**: everything above only
ever touches ``weights.connections`` (the scattered CSR side) -- block4-
resident synapses (created automatically by ordinary synaptogenesis, see
``block4_maybe_promote``) were NEVER read here at all, so a layer with any
block4 tiles gave silently wrong output through ``forward_sparse()``.
block4 is FP4-specific (see ``block4.hpp``), hence the ``if constexpr``
guard -- this whole section compiles to nothing for any other
``VALUES_TYPE``.

Read-only, same as ``disldo_forward``'s own block4 loop
(``linear_disldo.hpp``): forward does NOT update per-synapse block4
weight/importance inline (a documented, pre-existing gap -- see
``disldo_forward``'s "KNOWN GAP" comment, not something this change is
expected to newly fix), so no ``learning_rate`` handling is needed here,
only decode + multiply + accumulate.

**Design A** (see ``TODO_DUAL_BLOCK4.md``): reuses the exact
``work_offsets``/``chunk``/``w_start``/``w_end`` shape the scattered pass
above already uses, one level up -- over ACTIVE windows (block-rows with
>=1 nonzero input AND >=1 live block4 tile) instead of over individual
scattered synapses. A window's real "work" is its block4 tile count
(``weights.block4.block_layout.row_nnz(br)``), mirroring how the scattered
pre-pass sizes work by ``L.row_nnz(in_idx)``. Explicitly a first,
measured-not-assumed choice: if the serial gather pre-pass turns out to
dominate at realistic densities, Design B (direct per-active-window binary
search, no pre-pass) is the documented fallback, not a hypothetical.

**PRECONDITION**: ``input_tensor``'s indices, within each batch row, must
be ascending (standard CSR convention) -- required for the gather to find
a window's up-to-4 entries via one contiguous scan instead of a search.
``top_k()``'s own output is sorted by magnitude, not index -- callers must
run ``sort_indices()`` (``parallel.hpp``) first if their input came from
there. Not re-checked/enforced here (same convention the scattered pass
above already silently assumes).

Per-batch scratch (``win_br``/``win_vals``/``win_work_offsets``) is reused
across batches, not reallocated per batch -- same reasoning as
``block4.hpp``'s own persistent scratch buffers: batch=1 real-time calls
can't amortize repeated heap allocation.

.. _sisldo_forward.block4_fp4_only_silent_zero_for_dense_fp32_fp8:

``sisldo_forward``/``disldo_backward_sparse_grad``: FP4-only block4 gate
silently zeroes dense=True fp32/fp8 layers
------------------------------------------------------------------------

*ID:* ``sisldo_forward.block4_fp4_only_silent_zero_for_dense_fp32_fp8``

The ``if constexpr (std::is_same_v<VALUES_TYPE, FP4BiPacked>)`` guard on
both the forward (this file, ~line 252) and backward (~line 739) block4
branches was a deliberate, documented scoping choice when written --
block4 itself was FP4-only at the time. Since then, block4 storage grew
FP8 (``Block4Tile8``/``Store8``, tasks #90-96) and float32
(``Block4Tile32``/``Store32``, tasks #391-398) variants, and neither of
those got the same ``if constexpr`` branch. Nobody circled back to widen
this gate or add a guard against the combination -- it just silently
compiles to a no-op for those types, exactly as designed for the FP4-only
case it was written for.

**The consequence that wasn't previously connected**: a layer constructed
with ``dense=True`` is, by design, 100% block4-resident (the initial load
goes straight into block4 via ``load_dense_codes``, bypassing
``max_weights``/scattered storage entirely -- see ``sparse_rnn.py``'s
``_preseed_dense``/``_preseed_dense_fp32``). For such a layer, at
``VALUES_TYPE != FP4BiPacked``, this function's scattered-side loop finds
nothing (0 synapses live there) AND the block4 branch compiles to
nothing -- so ``forward_sparse``/``backward_sparse`` on a dense=True
float32 or FP8 layer returns EXACTLY ZERO regardless of input, with no
error, warning, or NaN to signal it. Found via
``sili_peridot``: an adaptive-sparsity MQAR curriculum run
(``dense=True``, ``precision="fp32"``, ``x_r_target=0.9``) showed loss
frozen bit-for-bit at ``ln(128)=4.852030277252197`` (the exact
cross-entropy of a uniform 128-way distribution) for 45,000+ steps --
traced directly to ``DISLDOLayer32.forward``'s CSR-input branch
returning an all-zero output tensor for a real, well-formed, nonzero
CSR input (``csr.nnz=9``, ``values.sum=-0.465``).

This means activation-level sparsity (``x_r_target``/``dy_r_target``,
and the older fixed-fraction ``input_sparsity_p``/``dy_sparsity_p`` --
both route through this same forward_sparse/sisldo_forward function) has
never actually worked for ``dense=True`` fp32 or FP8 layers, only for
FP4. Fix direction: replace the hardcoded ``FP4_TABLE[byte & 0xFu]``
decode (this file's block4 gather/scatter loops) with the
type-generic ``ValueAccessor<VALUES_TYPE>`` pattern already used
throughout the scattered-side code in this same file, mirroring how
``linear_disldo.hpp``'s DENSE-path block4 kernel was made type-generic
across FP4/FP8/float32 (tasks #197, #391-398) -- not a new design, a
generalization of an existing one. Tracked as sili_peridot task #413.

**Update -- fix landed, plus two independent pre-existing bugs found and
fixed along the way.** ``sisldo_forward`` and
``disldo_backward_sparse_grad``'s block4 branches are now type-generic
(``decode_weight``/``decode_importance``/``encode_and_store`` lambdas,
mirroring ``linear_disldo.hpp``'s ``if constexpr`` dispatch pattern), with
a new TDD parity gate (``tests/unit/test_sisldo_disldo_parity_fp32.cpp``'s
``run_config_block4``, ``tests/unit/test_sisldo_disldo_parity_fp8.cpp``,
new file) asserting the sparse (CSR) path produces bit-close forward
output, dx, post-update true weights, and full AQRS state matching the
dense reference path exactly. Getting that gate green surfaced two
further, genuinely pre-existing bugs unrelated to the FP4-only gate above:

1. ``disldo_backward_sparse_grad``'s row-local workspace byte cursor
   (``local_pos``) advanced using the bare FP4-only
   ``block4_stored_tile_len`` free function for every ``VALUES_TYPE``,
   instead of the type-correct ``..._len8``/``..._len32`` variant
   (``block4.hpp`` defines three separate, non-interchangeable versions --
   different tile byte widths and sparse-pack formats per storage type).
   For any row with more than one live tile, this misaligned every FP8/
   float32 tile after the first, corrupting decode of subsequent tiles in
   that row. Fixed with a small type-dispatched ``tile_len_of`` lambda,
   matching the file's existing ``decode_weight``-style dispatch
   convention. This one was entirely new code (introduced by this same
   generalization effort), not a latent bug in previously-shipped code.

2. **Independently pre-existing, unrelated to this file entirely**:
   ``linear_disldo.hpp``'s FP8 block4 backward write-back (both the SIMD
   path and its scalar fallback) called ``fp8_quantize_stochastic``/
   ``fp8_quantize_stochastic_live`` unconditionally, never checking the
   ``StochasticRounding`` template parameter -- unlike the FP4 and
   float32 branches in the same function, which already gate correctly.
   This meant the "dense reference" arm of the new parity test (which
   passes ``StochasticRounding=false``, expecting deterministic rounding)
   was silently dithering every FP8 block4 weight write regardless, while
   the sparse arm (``sisldo_ops.hpp``, correctly gated) rounded
   deterministically -- producing per-cell divergences on the order of one
   FP8 quantization step that superficially looked like column-index
   swaps but were actually independent stochastic-vs-deterministic
   rounding outcomes on the same underlying value. Fixed by adding the
   same ``if constexpr (StochasticRounding)`` gate FP4/float32 already
   have, using ``fp8_quantize``/``fp8_quantize_live`` for the deterministic
   branch. This bug predates the FP4-only gate fix above entirely and
   would have affected ANY FP8 block4 caller requesting deterministic
   rounding, not just the sparse path -- e.g. any test or production
   caller passing ``StochasticRounding=false`` to ``disldo_backward``
   directly on a dense=True FP8 layer.

Both fixes are covered by the new parity tests -- ``test_sisldo_disldo_
parity_fp32``/``test_sisldo_disldo_parity_fp8`` are true bit-close TDD
gates now (per sili_peridot's explicit directive: "sisldo with csr should
match disldo with 0 value inputs so we can have bit exact tests"), not
just loose-property checks. Full suite: 153/153 passing.

.. _sisldo_forward.block4_incremental_walk_perf:

``sisldo_forward``: incremental tile walk avoids a redundant rescan
--------------------------------------------------------------------------

*ID:* ``sisldo_forward.block4_incremental_walk_perf``

The block4 gather loop walks from a row's start, tracking
``elem_pos``/``byte_pos`` as it goes (mirrors ``disldo_forward``'s own
collection loop) -- this avoids ``find()``'s redundant O(row_nnz) rescan:
the earlier version did the SAME walk twice per tile (once via the cursor,
once again inside ``find()``'s ``raw_find``), measured as the dominant real
overhead vs. dense at high density -- **2.7x slower than dense for
identical tile-work at density=0.9**, with the serial pre-pass itself
under 1% of total time.

``at_index()`` here is a cheap lookup: the walk above already knows this
tile's exact storage position, so it skips ``find()``'s redundant rescan.
Read-only, does not mark the handle dirty, same as ``disldo_forward``'s own
block4 loop.

.. _sisldo_forward.block4_zero_skip:

``sisldo_forward``: block4 zero-skip (top-k sparsity was a silent no-op)
---------------------------------------------------------------------------

*ID:* ``sisldo_forward.block4_zero_skip``

**Real bug, found and fixed** (top_k4 sparsity work): ``local[]`` holds a
window's 4 gathered input values -- until this fix, every ``li`` was
decoded and multiplied regardless of whether ``local[li]==0``, so
sparsifying the input (top-k or otherwise) was a silent no-op on a
block4-resident layer; only the OUTER window-admission check
(``row_nnz_b4>0``) ever skipped work.

**Window-level early-out**: if every gathered value in a window is exactly
zero (either because the input was already zero there, or a
sparsification step zeroed all 4), skip the ``lj``/``li`` decode+
accumulate loops entirely -- the walk bookkeeping above
(``elem_pos``/``byte_pos``/``bc_cursor``) has already run and must NOT be
skipped (later windows' incremental walk depends on it), only the real
per-slot decode work is saved here.

**Per-``li`` skip**: a zeroed gathered input contributes nothing
regardless of the weight -- skip its decode too, not just the
multiply-accumulate.

.. _sisldo_forward.additive_branch_port:

``sisldo_forward``: AQRS additive branch, ported to a CSR walk
-------------------------------------------------------------------

*ID:* ``sisldo_forward.additive_branch_port``

Direct port of ``disldo_forward``'s identical section
(``linear_disldo.hpp``), adapted for sparse input: the ``P[k]`` projection
walks the CSR's nonzero entries instead of a dense row scan --
functionally identical, since the dense version already skips ``iv==0``
entries. The second pass (``P -> output``) is unchanged (dense over
``out_cols``, no sparsity to exploit there -- every output column can
receive a nonzero additive contribution regardless of which synapses are
live). No-op at the default ``additive_rank==0``.

.. _sisldo_ops.delta_csr_backward_removed:

Removed: ``delta_csr_backward`` (sparse input + sparse gradient)
-------------------------------------------------------------------

*ID:* ``sisldo_ops.delta_csr_backward_removed``

Confirmed wrong design: sparse input in backward permanently loses the
ability to correct "didn't fire & should have" (a row not in the sparse
input representation has no computational path to receive gradient at all,
regardless of how strong the signal is). Only "fired & shouldn't have"
could ever be fixed. Replaced by ``disldo_backward_sparse_grad`` below
(dense input, sparse gradient) -- the only sparse-gradient backward variant
that should exist. Confirmed zero real callers before removal (only this
file's own definition matched a search for ``delta_csr_backward(``).

.. _disldo_backward_sparse_grad.dense_input_rationale:

``disldo_backward_sparse_grad``: why input is always dense here
------------------------------------------------------------------

*ID:* ``disldo_backward_sparse_grad.dense_input_rationale``

Per ``sisldo_ops.delta_csr_backward_removed`` above: this is the ONLY
sparse-gradient backward variant -- there is deliberately no sparse-INPUT
backward. Input is always dense here (available regardless of which
forward path was used, since sparsification never destroys the underlying
dense array). Only the GRADIENT toggles sparse/dense, matching the actual
performance bottleneck (backward's cost is dominated by the gradient side,
forward's by the activation side -- these are independent axes, not mirror
images of each other).

**Why dense input specifically** (not just "simpler to implement"):
``dx[r] = sum_c W[r,c]*dy[c]`` depends only on weights and the gradient,
not on ``input[r]`` itself -- so a row whose OWN activation was zero/
near-zero this pass still gets a correct ``dx``, correctly telling
whatever produced this input "you should have fired more here." A
sparse-input design would skip that row entirely (it's not in the sparse
representation at all), permanently losing the ability to correct this.
The weight update DOES scale with ``input[r]`` (via
``grad = dy_val * in_val``), so it naturally stays small for rows that
didn't fire -- appropriately conservative, without needing to skip the
row. Net effect: dense input covers both "fired & shouldn't have" (weight
update, scales with the real input value) and "didn't fire & should have"
(``dx``, weight-only, reaches the row regardless of its own value) --
sparse input would only ever cover the first.

.. _disldo_backward_sparse_grad.template_parity_and_scale_space_bug:

Template-parameter parity with ``disldo_backward``, and a scale-space bug
--------------------------------------------------------------------------------

*ID:* ``disldo_backward_sparse_grad.template_parity_and_scale_space_bug``

Template-parameter parity with ``linear_disldo.hpp::disldo_backward``
(task #100, "Apply same template params to sisldo_ops.hpp backward
functions"): ``ScalePolicy``/``StochasticRounding``/``SynapsePolicyT``, in
the same order ``disldo_backward`` uses (``DeferredScaleWrite``
deliberately scoped out -- orthogonal to this function's actual callers).

This closes two real gaps at once:

1. ``synapse_kwargs`` (``max_abs_delta``/``max_ci``/``min_decay_frac``/
   ``scale_invariant``) previously could not reach ``backward_sparse`` at
   all, no matter what a caller passed.
2. The scattered AND block4 phases below were both keeping their
   weight-update math in TRUE-WEIGHT space while storing back into CODE
   space via ``new_w / combined_scale`` -- the same ~1/S^2-scale-direction
   bug class found and fixed in FP8's block4 backward earlier (see
   ``docs/research/fp8quant.rst`` / the FP8 block4 scale bug fix). Porting
   to ``disldo_backward``'s own code-space convention (``quant +=
   update_cw(...)``, ``S`` properly threaded as a real per-synapse scale
   rather than implicitly inverted) fixes this identically here.

.. _disldo_backward_sparse_grad.dy_density_normalization_fix:

AQRS neurogenesis-trigger normalization: scaling by actual dy density
------------------------------------------------------------------------------

*ID:* ``disldo_backward_sparse_grad.dy_density_normalization_fix``

Direct instruction, ``sili_peridot`` JOURNAL.md 2026-08-31: ``scale_gamma``'s
own merge-walk (``og_ptr`` against ``out_grad_sparse``) and
``additive_gamma``'s ``dP``/``dgamma`` computation both only accumulate
contributions from ``out_grad_sparse``'s SURVIVING columns -- this function
is only ever called when ``dy`` is genuinely sparse (``dy_sparsity_p``
set; see ``disldo_backward_sparse_grad.dense_input_rationale`` above), so
those sums are structurally smaller than the dense case by roughly the
surviving fraction.

``grad_norm_divisor`` (both gamma branches) was ``n_inputs*out_cols``
unconditionally, with nothing to compensate.

**Real bug, confirmed via a 20k-step run**: ``input_sparsity_p``/
``dy_sparsity_p=0.5`` pinned every layer's AQRS rank at 1 the entire
curriculum (a 933-mutation isolation run with ``dy`` dense again grew
ranks normally, same seed/everything else). Fix: scale
``grad_norm_divisor`` by the ACTUAL observed ``dy`` density for this call,
not the nominal ``p`` -- self-corrects for graded per-row schedules too,
not just a uniform ``p``.

.. _disldo_backward_sparse_grad.rank_backfill_pattern:

AQRS rank-N scaffolding: shared accumulators and the k>=1 backfill
-----------------------------------------------------------------------

*ID:* ``disldo_backward_sparse_grad.rank_backfill_pattern``

Direct port of ``disldo_backward``'s identical scaffolding
(``linear_disldo.hpp``) -- see its own comments for the full rationale.
Shared by both the scattered phase and the block4 phase further down,
hence computed once here rather than inside either guarded block. AQRS
gamma's own gradient (task #273/#283 parity) is layer-wide, not
per-row/col, so sized ``num_cpus*rank`` (not ``num_cpus*out_cols*rank``
like ``t_col_grad``).

Pre-sizing ``value_scale``/``output_scale`` to ``n_inputs*rank``/
``out_cols*rank`` uses the same backfill pattern as
``disldo_backward``'s identical pre-sizing (``linear_disldo.hpp``) -- see
that for the real bug this exact pattern fixes: a uniform 1.0 fill would
put every new rank component in permanent lockstep with ``k==0``; only
``k==0`` gets the transparent-default 1.0, ``k>=1`` starts at the neutral
0.0.

.. _disldo_backward_sparse_grad.batch_outer_row_inner_layout:

Why the value-scale-gradient accumulator differs from ``disldo_backward``'s
-----------------------------------------------------------------------------------

*ID:* ``disldo_backward_sparse_grad.batch_outer_row_inner_layout``

Importance stats accumulators (``total_sum_abs_new_i`` etc.) accumulate
across batches. The ``value_scale`` gradient accumulator
(``scale_grad_sums_rank``) is a serial per-``(row,k)`` vector: within the
parallel-for, each ``r`` is unique per thread, so ``+=`` into
``scale_grad_sums_rank[r*rank+k]`` is race-free. Applied once after all
batches ("sum first, then apply lr" per conversation).

**Historical note**: this function used to nest batch OUTER / row INNER
(the opposite of ``disldo_backward``'s row-outer/batch-inner layout),
which is why this accumulator was sized ``n_inputs*rank`` (persisting
across the whole ``for (batch)`` loop) rather than a small per-row-local
vector. That layout turned out to hide a real bug -- see
``disldo_backward_sparse_grad.batch_aggregated_update`` below, which
fixed the loop nesting itself. This accumulator's shape (indexed by
``(row,k)``, sized for the whole call) was kept as-is since it's correct
either way and reshaping it wasn't needed to fix the bug -- it's just no
longer required BY the (now row-outer) loop structure, only by mirroring
``disldo_backward``'s naming convention here.

.. _disldo_backward_sparse_grad.batch_aggregated_update:

Fixed: sequential per-sample ci/cw updates instead of one batch-aggregated update
-------------------------------------------------------------------------------------

*ID:* ``disldo_backward_sparse_grad.batch_aggregated_update``

``disldo_backward`` (the dense path, ``linear_disldo.hpp``) reads each
synapse's weight/importance ONCE before its batch loop, accumulates
grad/contrib across the WHOLE batch, and applies exactly ONE ci/cw
(RMSprop state + weight) update per synapse per call -- a real batch-
gradient-descent step. ``disldo_backward_sparse_grad`` did NOT: both its
scattered path (``for (b) { parallel for (r) {...} }``) and its block4
path (same nesting) re-read AND re-encoded every live synapse on EVERY
batch sample, sequentially -- each sample's update was applied against
the state left behind by the previous sample in the same call. That's a
different optimizer (sequential SGD across the batch dimension) whenever
the same synapse is touched by more than one sample in a call, not a
batch-gradient step. ``test_sisldo_disldo_parity.cpp`` never caught this
because it only ever exercises batch=1, where "aggregate then apply once"
and "apply once per (single) sample" are the identical operation.

Found by explicitly asking "does this function actually apply a real
batch-gradient step like disldo_backward, or something else, under
batch>1?", then proving it with a new test
(``test_sisldo_disldo_multibatch_parity.cpp``, batch=2 with input/dy rows
deliberately overlapping on the same synapses) which failed hard before
the fix (e.g. one true-weight divergence of ``dense=1.50`` vs
``sparse=1.00`` -- not numerical noise).

**Fix**: swapped the loop nesting in both the scattered and block4 write
paths from batch-outer/row-inner to row-outer(parallel)/batch-inner,
matching ``disldo_backward``:

- **Scattered path**: snapshot every synapse's ``cw``/``S``/``ci`` ONCE
  per row (before any batch sample), accumulate ``grad_sum``/
  ``contrib_sum`` per synapse across the whole batch, then apply exactly
  one ``update_ci``/``update_cw`` + ``set_live`` per synapse afterward.
- **Block4 path**: same idea, structured as three passes per row --
  (1) read-only: decode every live tile's weight/importance ONCE
  (byte-cursor positions are stable until pass 3 re-encodes, since
  decode-only unpacking never changes a tile's stored length);
  (2) accumulate-only per batch sample (dx uses the pre-update weight
  for every sample, matching ``disldo_backward``); (3) apply-once: one
  ci/cw update + one encode/commit per tile, using grad/contrib
  aggregated across the whole batch.
- The read-only (``learning_rate == 0``) block4 branch got a smaller,
  free bonus fix along the way: tile coordinates (``bc``/``elem_pos``/
  ``byte_pos``) don't depend on the batch sample, so they're now
  precomputed once per row instead of re-walking the row cursor on every
  batch sample.
- The rank-N ``value_scale``/``output_scale``/``scale_gamma`` axis was
  ALREADY correctly aggregated across the whole batch in both paths
  (feeding ``scale_grad_sums_rank`` et al.) -- unaffected by this bug,
  no changes needed there.

Verified via the new multi-batch parity test (both scattered and block4
configs now pass) plus the full regression suite (164/164 C++ incl. both
bit-exact dequant-equality gates, 215/215 Python). Shared template code
(one ``disldo_backward_sparse_grad`` instantiated per ``VALUES_TYPE``), so
the fix applies to fp8/fp4/fp32 uniformly.

.. _disldo_backward_sparse_grad.read_only_decode_cache:

Fixed: read-only (``learning_rate == 0``) block4 path re-decoded per batch sample
-------------------------------------------------------------------------------------

*ID:* ``disldo_backward_sparse_grad.read_only_decode_cache``

Follow-up to ``batch_aggregated_update`` above, found while benchmarking it: the
read-only block4 branch (no weight update, just dx) already had its tile
*coordinates* precomputed once per row, but still called
``weights.block4.at_index()`` and ``decode_weight()`` fresh on EVERY batch
sample that touched a given tile -- the coordinate fix removed the cursor
re-walk but not the actual bit-unpack + scale lookup, which is the more
expensive part. Fixed by decoding ``w_decoded * S`` once per (tile, li, lj)
into a per-row buffer before the batch loop, alongside the coordinate
precompute; the batch loop now does a single array read instead of a
tile lookup + decode per sample.

Measured on a fully-dense (``load_dense_codes``) 288x288 fp4 layer,
batch=256: this read-only path went from 34.8ms to 10.3ms (~3.4x), on top
of the earlier coordinate-precompute fix (38.8ms originally) -- ~3.8x
total. The write path (``learning_rate != 0``) already had its own decode
cache from ``batch_aggregated_update`` and is unaffected by this change.

``sisldo_forward`` (the separate all-read forward function, sisldo_ops.hpp
top half) has a similar-flavored redundancy in its own block4 branch --
each work item independently calls ``at_index()`` + decode, with no cache
across batch samples touching the same tile -- but its work is distributed
by flattening `(batch, window)` pairs across ALL samples to load-balance
genuinely variable per-sample input sparsity, rebuilt fresh every batch
iteration. Fixing it the same way would mean inverting that work
distribution (group by tile first, then find which batches touch it)
rather than a small mechanical addition, and risks hurting load balance
for genuinely uneven-sparsity batches -- deliberately NOT done in this
pass; flagged as a separate, larger-scoped follow-up if needed.

.. _disldo_backward_sparse_grad.merge_scan_design:

Merge-scan cost and the additive contrib combination
---------------------------------------------------------

*ID:* ``disldo_backward_sparse_grad.merge_scan_design``

The scattered backward loop merge-advances both this row's columns and the
gradient's columns (both sorted ascending) -- O(``nnz_this_row`` +
``grad_nnz``) per row, not a search per synapse. ``dx_accum`` is
weight-only, so it reaches this row regardless of ``in_val`` (see
``disldo_backward_sparse_grad.dense_input_rationale`` above).

The additive ``contrib`` combination (``ci = SynapsePolicy::update_ci(ci,
grad, contrib, ...)``) mirrors ``disldo_backward``'s own ``ci`` update
(see ``scale_policy.contrib_agg_combination`` in
``docs/research/delta_csr_types.rst``) -- square-then-sum, not
sum-then-square: a large-magnitude disagreement between ``grad`` and
``contrib`` must still damp the step, not collapse the denominator toward
zero and explode it.

``dL/d(value_scale_k(r,k)) = grad * quant_floor * output_scale_k(col,k) *
gamma_k`` for each component ``k`` -- direct port of ``disldo_backward``'s
identical loop (``linear_disldo.hpp``), including ``quant_floor``'s
zero-escape gating (only ``cw_orig==0`` substitutes a small positive
epsilon; every other value uses itself directly, exact and signed).

.. _disldo_backward_sparse_grad.block4_backward_design:

block4 backward: parallelized by block-row, not by tile
------------------------------------------------------------

*ID:* ``disldo_backward_sparse_grad.block4_backward_design``

**Real, previously-silent bug this section closes**: same as
``sisldo_forward.block4_gather_design`` above -- block4-resident synapses
were never touched by this function at all. FP4-specific, hence the
``if constexpr`` guard.

**Gather design**: mirrors the scattered loop's own merge-scan
(``row_cursor`` + ``og_ptr`` walking forward through sorted
``out_grad_sparse``), applied one level up -- per TILE (4 output columns)
instead of per synapse. PRECONDITION: ``out_grad_sparse``'s indices,
within each batch row, must be ascending (same convention
``sisldo_forward``'s input requires).

**Parallelized by BLOCK-ROW (``br``), not by tile** -- unlike
``disldo_backward`` (which partitions by flat tile index and therefore
needs cross-thread accumulator buffers, since two different tiles can
share a block-row when they differ only in block-column), here each
``br`` owns exactly 4 unique input rows (``br*4..br*4+3``) that NO OTHER
``br`` ever touches, and a thread processes one ``br``'s tiles serially
within itself -- so ``value_scale``/``dx``/importance-stat writes for
those 4 rows can go straight into shared (non-per-thread) accumulators
with no race, simpler than ``disldo_backward``'s scheme. This also gives
the same row-exclusive-ownership safety a block4 tile resize needs (see
``block4_resize_tile_in_row``'s comment) for free.

Correctness-first scalar port (matches ``sisldo_forward``'s own choice not
to bring over ``disldo_forward``/``disldo_backward``'s ``Block4Vec`` SIMD
machinery immediately) -- revisit with real profiling if a benchmark shows
this is a bottleneck.

**KNOWN SIMPLIFICATION** (documented, not a bug, mirrors
``disldo_backward``'s identical one): a row with both scattered and block4
synapses gets two sequential ``value_scale`` gradient steps (this
section's own, after the scattered section's own above) rather than one
combined step.

Total live slots across ALL of a row's tiles this call: every tile
contributes exactly ``BLOCK4_TILE`` slots per row it covers (dense,
weight=0.0 included, see ``block4.hpp``), mirrors ``disldo_backward``'s
``row_live_count``.

.. _disldo_backward_sparse_grad.block4_workspace_concurrency:

block4 backward: read-only vs. writing paths, and the ``was_live`` gate
----------------------------------------------------------------------------

*ID:* ``disldo_backward_sparse_grad.block4_workspace_concurrency``

**Read-only path** (``learning_rate == 0``): no writes anywhere in this
row, so no resize can ever happen -- the plain shared-store ``at_index()``
path has no concurrency hazard at all here (see
``Block4Store::RowWorkspace``'s comment on why that hazard is specifically
about growth).

**Writing path**: uses a row-local workspace (``snapshot_row``) so growth
never touches shared ``tile_data``/``tbyte_start``/``tbyte_end`` until a
single row-exclusive merge-back at the end -- the plain shared-store path
(``find()``-free or not) is NOT safe here under concurrent
per-row-owning threads, **confirmed via ASan**.

``was_live`` gate: ``byte`` here is the PRE-update packed byte, still
untouched at this point -- a cell that was never a real synapse
(``byte==0``) must stay allowed to round back to 0, not be forced
permanently live just because this column happened to have gradient
signal this step. See ``disldo_backward.was_live_gating`` in
``docs/research/linear_disldo.rst`` for the full rationale (same mechanism,
applied here to the CSR path).

**Merge back**: evicts lowest-|true-importance| synapses only if this row
genuinely grew past its own current headroom (see
``merge_row_workspace``'s comment). True importance uses the SAME
per-row/per-col scale lookups as the rest of this function, per
conversation (raw 4-bit codes alone aren't enough resolution to rank
meaningfully).

.. _disldo_backward_sparse_grad.output_scale_and_gamma_reduction:

``output_scale`` and ``scale_gamma``: reduced once after both phases
--------------------------------------------------------------------------

*ID:* ``disldo_backward_sparse_grad.output_scale_and_gamma_reduction``

``output_scale``'s own gradient reduction is a direct port of
``disldo_backward``'s identical block (``linear_disldo.hpp``) -- reduces
``t_col_grad``/``t_col_grad_contrib`` across every thread, once per
``(col,k)``, then applies via the same ``ScalePolicy``. Placed AFTER both
the scattered phase AND the block4 phase (both phases' ``mcol_at``
contributions land in the SAME shared ``t_col_grad`` buffer, matching
``disldo_backward``'s own placement) -- reducing any earlier would
silently drop block4's own contribution.

``scale_gamma``'s own update is likewise a direct port of
``disldo_backward``'s identical block, including the EMA/
dynamic-rank-control tracking call (``update_scale_gamma_ema_k``) -- that
method itself already lives on ``weights`` (shared with the dense path),
this just needs to feed it the same per-``k`` inputs ``disldo_backward``
does.

.. _disldo_backward_sparse_grad.additive_branch_backward:

AQRS additive branch backward: CSR walk replaces two dense scans
-----------------------------------------------------------------------

*ID:* ``disldo_backward_sparse_grad.additive_branch_backward``

Direct port of ``disldo_backward``'s identical section
(``linear_disldo.hpp``) -- genuinely independent of the sparse/block4
structure above (same reasoning as forward's additive branch), computed
as its own self-contained pass. No-op at the default ``additive_rank==0``.
``input`` is already dense (this function's own design); ``out_grad_sparse``
is the one place this needs real adaptation from ``disldo_backward``'s
dense ``output_grad`` scans -- both the ``dP`` projection AND ``dV``'s own
gradient walk it once via its CSR instead of a dense scan over ``n_out``
(the dense version already skips zero entries either way, so this is the
same computation, just walking the nonzero set directly):

- ``dP[b,k] = sum_c V(c,k)*dy[b,c]`` -- CSR walk over ``out_grad_sparse``
  instead of a dense ``for c in n_out`` scan.
- ``dV_accum[c,k] = sum_b dy[b,c]*P[b,k]`` -- built in the SAME CSR walk
  (both need the same ``(b,c,dy)`` triples), avoiding a second O(``n_out``)
  dense scan ``disldo_backward``'s own ``dV`` loop does.
- ``dX = sum_k U(r,k) * gamma_k * dP_raw[b,k]`` -- direct port,
  ``disldo_backward``'s identical formula.
- ``dU[r,k] = gamma_k * sum_b dP_raw[b,k]*X[b,r]`` -- direct port.
- ``dV[c,k] = gamma_k * dV_accum[c,k]`` -- already reduced above via the
  CSR walk, no dense scan needed here (unlike ``disldo_backward``'s own
  ``dV`` loop, which re-scans ``output_grad`` densely per ``(c,k)`` since
  it never had a sparse ``dy`` to walk).

``additive_gamma``'s own update is a direct port including its own EMA/
dynamic-rank-control tracking call (``update_additive_gamma_ema_k``), same
shared-method pattern as ``scale_gamma``'s above.
