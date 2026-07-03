# Refactoring status and plan

High-level "where are we and what's left" for the cpu_sparse_io +
optim_merge/master + sili_old consolidation into `sili_new`. This document
is the active work queue for the refactoring itself. `TODO.md` is for
backburner items -- things that wait until *after* this refactoring is
done, not things blocking it.

## Goal / definition of done

- `sili_new` contains the union of everything worth keeping from all three
  sources, actually working (tests passing), not just present.
- The three source clone directories end up empty (or `.MIGRATED` markers
  only) -- an empty directory is the "done" signal.
- gen_toy_mistral + rnn_fold working end-to-end against the current
  sili_new API, with the sparsity-metric-driven dense/sparse dispatch (see
  active queue below) -- this is the concrete target that "done" cashes
  out to right now, not an abstract completeness check.
- File-size limit (~1k lines) applies to *new* files going forward, not
  retroactively to old, already-tested files -- that cleanup is explicitly
  the *last* step (see active queue), not ongoing.

## Priority ordering

1. **This session's own SILi work** (built before the repo consolidation
   started) -- highest priority of all sources. Python-level tests
   (`test_sili.py`, `multimodal_sparse_rnn.py`, `sparse_tcnn_audio.py`,
   preserved in `test/python/`) are highest priority within that.
2. **optim_merge/master** (the actual GitHub repo, what `sili_new` was
   branched from) -- second priority.
3. **cpu_sparse_io** -- explicitly the old direction (2026-03-10, predates
   master). Relevant only where it has something genuinely unique.
4. **sili_old** -- oldest (Jan 2025), GPU/vision-shader-focused, mostly
   superseded.

Where two sources describe the same concept, higher priority wins by
default.

## Active priority queue (in order)

1. ~~Write this document~~ / ~~write TODO.md~~ -- in progress, this pass.
2. **Update tests for forward_dense/backward_dense + forward_sparse/
   backward_sparse.** `test_sili.py` and `multimodal_sparse_rnn.py`
   currently call this session's original unified `SparseLinearLayer`
   API (`forward()`/`forward_disldo()`-style names) -- need updating to
   the current split names. `test_sili.py` is already huge (1245 lines) --
   NEW tests for this go in a fresh file, not added to it.
3. ~~Get `importance_scale`/`rescale_importance` working and tested.~~
   DONE -- field on `SparseLinearWeightsDelta`, threaded through
   `disldo_forward`/`disldo_backward`/`delta_csr_forward`/
   `delta_csr_backward_sparse_grad`, exposed on `SparseLinearLayer`,
   verified at the FP4-encoding level, the kernel level, and end-to-end
   through the real Python pybind path (5 new regression tests, 26 total
   test cases / 94 assertions passing).

   EXTENDED per follow-up discussion: `importance_scale`/`value_scale`
   need to actually be MAINTAINED (not just settable), since the right
   scale can drift as training progresses. Not backprop (importance is a
   Hebbian, not gradient-based, quantity -- backprop doesn't fit the same
   framework) -- instead, self-correcting via an inverse-Hoyer signal on
   the STORED distribution: Hoyer's score near 0 means values are spread
   evenly across FP4's representable range (good), near 1 means
   concentrated (most values collapsing to the same point -- typically 0,
   meaning underflow, meaning rescale is needed).

   DONE (this pass): the O(1)-per-update infrastructure this needs.
   `SparseLinearWeightsDelta` now maintains running `value_l1`/
   `value_l2_sq`/`value_max_abs` and `importance_l1`/`importance_l2_sq`/
   `importance_max_abs` incrementally (updated by every kernel that writes
   a synapse, via `update_value_stats()`/`update_importance_stats()`) --
   L1 as a running sum of `|new|-|old|` (not raw deltas -- `|a+d| != |a|+d`
   in general), L2 as a running sum of squares updated via
   `l2_sq - old^2 + new^2`, both O(1), no rescan needed. `hoyer_value()`/
   `hoyer_importance()` compute Hoyer's measure from these in O(1).
   `recompute_stats()` gives an exact, from-scratch answer when needed
   (called automatically after every layer construction/load_weights).
   Exposed on `SparseLinearLayer` for Python-side use.

   Real bug found and fixed during this: FP4BiPacked's quantizer rounds to
   the nearest FP4_TABLE entry, so the value passed to `update_*_stats()`
   must be the value read back AFTER `set()` (the actual stored/quantized
   value), not the pre-quantization float that was computed -- using the
   latter caused real, measurable drift starting from the very first
   update. Caught by testing the incremental tracking against a fresh
   `recompute_stats()` after many forward+backward calls, not by
   inspection.

   CAVEAT, worth remembering when the policy below gets built: `max_abs`
   is a MONOTONIC upper bound incrementally (can't decrease without a full
   rescan, since maintaining an exact live max under arbitrary decreases
   needs more than O(1) bookkeeping) -- fine as "has this ever touched the
   ceiling," not as "what is the max right now." Also: the pure
   "keep Hoyer near 0" objective catches UNDERFLOW (collapse toward 0)
   but likely does NOT catch OVERFLOW/saturation (values piling up at
   FP4's ceiling, ±6) -- a layer where every value is exactly 6.0 has the
   same L1/L2 ratio as a fully dense, well-spread layer (Hoyer score 0
   either way), so it wouldn't be flagged by Hoyer alone. `max_abs` is the
   cheap complementary check for that case (e.g. trigger a rescale if
   `max_abs` sits at or near the ceiling for a stretch), tracked
   specifically because of this gap.

   NOT DONE, the actual next step: the adaptive POLICY itself (when to
   trigger `rescale_importance()`/an equivalent for value_scale, and what
   the new scale becomes) -- deliberately Python-side per explicit
   guidance ("newer and experimental, people might want to change it,
   python is easier to change, people can see what it's doing"). C++ owns
   the cheap running stats (near-free, since it's already touching every
   synapse it updates); Python owns the decision heuristic. Should apply
   broadly -- most layers, and specifically needed for rnn_fold (item 4).
   Not started.
4. **Get rnn_fold + gen_toy_mistral working again against the current
   API**, with the actual new piece: automatic dense/sparse dispatch using
   `hoyer_score()`, not just manually choosing forward_dense vs
   forward_sparse. Specifics:
   - Lives in **Python-level** forward/backward wrapper methods (used
     frequently, doesn't need to be in the hot C++ path the way the
     underlying kernels do).
   - Threshold: route to sparse when `hoyer_score > 0.8` (equivalently,
     estimated active fraction under ~20%) -- otherwise dense.
   - Note directly in the wrapper methods' own docstrings/comments, AND in
     `TODO.md`: this fixed threshold could eventually become adaptive
     based on measured time performance instead of a fixed constant --
     don't build that now, just don't lose the idea.
   - rnn_fold needs editing to support this (gen_toy_mistral depends on
     rnn_fold) -- expected to be a straightforward refactor, not a
     rewrite.
5. **Cleanup, last step, only after gen_toy_mistral is converted, working,
   and running**: delete the old files identified below as safe to
   delete, then produce a directory-tree listing of what remains across
   all four source locations as the final "here's where we ended up"
   artifact. The ~1k-line-file split-up (currently skipped for old,
   pre-existing, already-tested files) happens here too, if still wanted
   at that point.

## Known editing pitfalls (watch for these, don't let them ship silently)

- **str_replace anchor-consumption.** When `old_str` is a prefix of existing
  text used purely as an insertion anchor ("put new content right before
  this line"), `new_str` must repeat that anchor text at ITS OWN end, or
  the anchor gets silently deleted rather than preserved. Happened five
  times this session now (twice in `sparse_struct.hpp`/`TODO.md` during
  earlier edits, twice in `test_disldo_synaptogenesis.cpp`, once in this
  very document while writing this note about it). Every instance was
  caught by actually compiling/running tests afterward or by grepping for
  the anchor text -- never by re-reading the diff alone; the failure mode
  (an orphaned string literal + brace, or a document section landing in
  the wrong place / duplicated) can read as plausible on a quick visual
  pass. Mitigation: after any str_replace that inserts content adjacent to
  existing text rather than cleanly replacing a whole block, grep for the
  anchor text's distinctive substring afterward to confirm it's still
  present exactly once, in the right place -- in addition to (not instead
  of) compiling/running tests before considering an edit done. The
  compile/test/grep step has caught this every time so far -- the goal is
  reducing how often it's needed, not replacing it.

## Old-directory file verdicts

Full pass taken this session incorporating direct review (not just this
document's earlier categorization) -- verdicts below are as specific as
given, not guessed at.

### Confirmed safe to delete (cpu_sparse_io)
- `sparse_struct.hpp` -- superseded: `delta_csr_forward`/`delta_csr_backward`
  already do what this did.
- `linear_sisldo.hpp` -- "just the old sparse input sparse layer, which we
  already have."
- `unique_vector.hpp` -- "Premature optimization tbh. A non-copy vector.
  Makes maintaining hard. Remove it."
- `quantized_arrays.hpp` -- "Old attempt at general quantization. Fp4 is
  information theoretically optimal for reasonable compute times, so
  we're just going with that now."
- `outer_product.hpp` -- already deleted (earlier this session), confirmed.

### Confirmed safe to delete (sili_old)
- **Anything with "pyr" in the name** -- `modules/pyr_conv/`,
  `modules/depth_pyr_conv/`, `modules/horiz_pyr_conv/`,
  `modules/vert_pyr_conv/`, `modules/pyr_FAST/`,
  `modules/pyr_FAST_rejection/`, `modules/pyr_local_max_sparsity_enforce/`,
  `modules/pyr_patch_max_sparsity_enforce/`, `modules/image_pyramid/`
  (and cpu_sparse_io's `modules/image_pyramid_float32/`,
  `modules/image_pyramid_int8/`, `modules/pyramid_conv_d2d*/`). Verdict:
  "All the pyr_conv ops are basically worthless. The reason is we can just
  run a sparse layer as a kernel over the input and it works much
  better/faster... Anything with 'pyr' on it, we drop."
- `modules/radacon/`, `modules/adacon/` -- "GPU optim ops, r added random
  noise to break lock step failures. Superseded by our current optim
  stuff."
- `modules/multi_matrix_inverse/` -- "A kompute example I made for some
  forums. not really useful. Drop."
- `modules/to_spvec/` -- "to_csr but one sparse vector instead of one per
  row. to_csr does everything it does and actually works. Drop." (Also
  contains two files literally named `*_not_working.comp` -- wasn't even
  finished within sili_old's own history.)
- `TODO.md` (sili_old's own) -- "Those sound like things that were
  completed actually. Drop." No longer cross-referenced from this repo's
  TODO.md.

### Backburner (not deleted, not urgent, revisit only if a real need comes up)
- `coo.hpp` (cpu_sparse_io) -- parallel COO generation/sorting for
  synaptogenesis. "if we don't have those parallel versions then just
  move that to the backburner, because most of the time we can either
  just init a diagonal of zeroes or do maybe 100 or so synapse
  generations/deletions, and we would need to work with the new memory
  layout... I'd just say push that to backburner rather than look through
  probably 1000s of lines."
- `csf.h`/`csf.cpp` -- "Compressed sparse fiber. Backburner stuff."
- `fiber.hpp`/`old_fiber.hpp` -- already backburnered (see TODO.md).
- `modules/mse_loss/` -- "Just a simple mse_loss on gpu op." Low priority,
  trivial to recreate if ever needed, not worth porting proactively.
- `modules/reduction/` -- "Just a simple gpu reduction op." Same
  treatment.

### Needs real evaluation, not decided (don't guess)
- `csr.hpp` (cpu_sparse_io) -- "a bunch of stuff that worked with sparse
  struct. Since we're no longer using that structure I think a lot of it
  doesn't actually work, though some might so it would be good to check
  the functions to see if we have versions or not." Needs a
  function-by-function check against sili_new's current headers, not a
  blanket verdict either way.
- `rolling_linear.hpp` -- "an attempt at sliding window attention. Might
  be good to look at and compare to our version of sliding window
  attention if we have sliding window attention in the newest repo." (The
  `banded_attention_forward`/`sparse_banded_attention_forward` functions
  from earlier this session are the likely comparison point -- not yet
  actually compared.)
- `cache_info.h`/`cache_info.cpp` -- "I honestly forgot what these were.
  Idk." Genuinely unclear even to the source, low priority either way.
- `modules/csr_matmul_csr/` (sili_old) -- "Basically a linear op that
  doesn't require csr-csc. Could be useful idk." Uncertain value, not a
  drop, not a confirmed-keep.

### Genuinely valuable, worth porting (not yet started)
- `linear_sidlso.hpp` (cpu_sparse_io) -- **Sparse Input, Dense Layer,
  Sparse Output** (correcting an earlier mis-parse of the acronym as
  "sidldo"). The dense-layer-with-sparse-input core (a layer whose WEIGHTS
  are dense/unstructured but whose INPUT activations are sparse-CSR --
  architecturally distinct from SparseLinearLayer, which assumes
  connectivity-sparse weights) is genuinely useful for "odd scenarios,
  like sub-manifold conv ops" -- doesn't need delta-CSR or any of the new
  connectivity-sparsity machinery at all since the weights themselves
  aren't row-sparse, but DOES need converting to FP4 for the weight
  values (currently presumably float32). The auto-sparsification of the
  OUTPUT specifically ("I made an attempt to use output
  auto-sparsification and it didn't work well") is NOT worth preserving
  -- drop that part, keep the dense-layer-with-sparse-input computation.
  Relevant sooner than originally expected given the V-LLM vision-support
  need below (submanifold conv is a real technique for sparse spatial
  data).
- System/hardware energy-and-thermal monitoring (`util/energy.py`,
  `energy_2.py`, `system_energy.py`, `stress_2.py`, cpu_sparse_io) --
  confirmed NOT superseded by the neural energy dynamics system from
  earlier this session, genuinely complementary infrastructure. Design
  intent, stated directly: this feeds the energy system's hard-ceiling
  parameter so it can "work comfortably at temperatures that keep the
  CPU/GPU lasting a long time," "work harder when focusing on tasks," and
  "risk super high temperatures if it needs to do something critical,
  like safely managing a fall in a robot body when the CPU is in that
  robot body (no point in preserving long term temperatures when the
  thing we're preserving might break in the short term)." The different
  operating modes (conservative/long-term vs. aggressive/short-term-
  critical) are envisioned as **output neurons or tokens the AI itself has
  access to** -- i.e. the model can choose to push hardware limits when it
  judges something more important than thermal longevity is happening,
  not just a fixed external policy. Not started -- needs its own design
  pass when energy-dynamics wiring is picked up.

## Elevated by the V-LLM realization (was backburner, now "later todo")

This project is porting Mistral 24B, which is a vision-language model --
this changes the calculus on two previously-backburnered items:

- **GPU device/runner abstraction** (Kompute wrapper, `core/devices/gpu.py`
  and friends, cpu_sparse_io) -- moves from backburner to "later todo."
  Needed to handle vision input at decent speed. The CPU-side equivalent
  of this abstraction is already effectively covered by the existing
  direct-call architecture (`cpu_backend.cpp`/`SparseLinearLayer` calls
  the C++ functions directly -- no separate "device" indirection layer
  needed for CPU specifically); it's the GPU path that needs the real
  abstraction, since GPU execution requires shader dispatch and buffer
  management the CPU path doesn't.
- **A "default" conv op** (explicitly NOT the pyramid-based ones, all of
  which are dropped per above) -- needed for vision input, later todo,
  not backburner. Not yet designed.

## Corrections to prior TODO.md entries

- **"Parallel pointers" (mid-row resume mechanism) is the wrong design,
  not just "not yet ported."** Original framing (built earlier this
  session, before this repo's consolidation) was a resume-via-search
  mechanism for parallelizing across chunks of one row. Per correction:
  "This is supposed to be superseded by the work pointers iirc. The CSR
  memory had 2 pointers, one set that would go to regions that were
  almost exactly equal size, the other that would point to the row
  beginnings. That's strictly better than the mid-row mechanism because
  there's no searching required. It's O(1) at runtime instead of O(log n),
  and synaptogenesis/pruning just needs to maintain a clean work pointer
  set." Two pointer sets, not one: (1) roughly-equal-sized WORK regions
  for load-balanced parallelization (not necessarily row-aligned), (2)
  row-beginning pointers (the normal row_ptr array). Synaptogenesis/
  pruning's job is just keeping the work-pointer set clean/valid, not
  searching. TODO.md entry needs rewriting to this design, not the
  original one.
- **`make_weights`/`make_weights_v`'s bug may not need fixing at all.**
  Clarified: the V/high-precision layers should NOT use FP4BiPacked --
  they should use ULEB128 delta encoding (i.e. genuinely
  `SparseLinearWeightsDelta<S, DeltaCSRBiValues<float>, COL_TYPE>`, the
  same delta-CSR pattern DISLDOLayerV already correctly uses) with
  **BiValues, not TriValues** (weight + importance only -- no longer
  storing backprop in a third value list, since backward always updates
  in place now). This means `SISLDOLayerV` itself is currently
  architecturally wrong (still on `SparseLinearWeightsV`/TriValues/
  absolute-CSR), not just calling a buggy function -- the fix is
  rewiring it to the delta-CSR+BiValues pattern (exactly mirroring
  DISLDOLayerV's already-verified upgrade), at which point the old
  `make_weights_v` code path it currently depends on becomes unnecessary
  rather than needing its own fix. Separately: `make_weights`'s actual
  PURPOSE -- "give a csr in a few vectors with {1,2,3,...} and have the
  usable CSR for testing" -- is already served by
  `delta_csr_from_absolute`, used in every test written this session.
  Explicit permission given to just replace the old standalone
  `make_weights`/`make_weights_v` pybind bindings with something built on
  `delta_csr_from_absolute` rather than debug the old ones.
