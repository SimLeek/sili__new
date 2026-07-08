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
   preserved in `tests/unit/python/`) are highest priority within that.
2. **optim_merge/master** (the actual GitHub repo, what `sili_new` was
   branched from) -- second priority.
3. **cpu_sparse_io** -- explicitly the old direction (2026-03-10, predates
   master). Relevant only where it has something genuinely unique.
4. **sili_old** -- oldest (Jan 2025), GPU/vision-shader-focused, mostly
   superseded.

Where two sources describe the same concept, higher priority wins by
default.

## Active priority queue (in order)

1. **Update tests for forward_dense/backward_dense + forward_sparse/
   backward_sparse.** `test_sili.py` and `multimodal_sparse_rnn.py`
   currently call this session's original unified `SparseLinearLayer`
   API (`forward()`/`forward_disldo()`-style names) -- need updating to
   the current split names. `test_sili.py` is already huge (1245 lines) --
   NEW tests for this go in a fresh file, not added to it.
3. **Adaptive rescale policy for importance_scale / value_scale** (item 3,
   Python side -- C++ stats infrastructure is DONE, in refactoring_done.md).

   The adaptive POLICY for when to trigger `rescale_importance()` and what
   the new scale becomes. Deliberately Python-side. C++ owns the cheap
   running stats (near-free); Python owns the decision heuristic:
   - Trigger on `hoyer_importance() > threshold` (underflow) or
     `max_abs` near ceiling (overflow/saturation)
   - Applies broadly -- most layers, specifically needed for rnn_fold

   NOT started.

   **Memory management design (open issue from synaptogenesis work):**
   `equalize_to_capacity()` currently grows memory dynamically via
   `delta_csr_shift_row`. This is wrong -- memory cannot grow
   indefinitely at runtime. The correct design:
   - At construction (from_descriptor): size the initial budget for
     `peak_max_row_weights` connections per row. The existing
     `SparseLinearLayer(n_in, n_out, max_weights, cpus)` constructor
     already takes a `max_weights` budget. The rnn_fold path just needs
     to pass the right value (n_folds * out_dim * some_factor).
   - Drop `equalize_to_capacity` (or convert it to a no-grow assertion
     that verifies the existing budget is sufficient, not a grow-op).
   - Let staggered `equalizer_step()` handle per-cycle redistribution
     within the fixed pool. That was always the design intent.
   - The old `optim_synaptogenesis` in the pre-delta-CSR codebase used
     `reserve_connections` (a simple `.reserve()`) for the same reason:
     one upfront reservation, then in-place modification. The new
     delta-CSR version just needs the budget passed at construction time.

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
  the anchor gets silently deleted rather than preserved. Has now happened
  8 times this session; a specific pattern has emerged: it happens almost
  exclusively when the anchor is a `TEST_CASE(` macro opening (or an
  orphaned closing fragment like `"[tag][tag]") {`). Root cause: these
  strings occur many times in the files, so a vague prefix match during
  an edit that inserts content "just before this test" is especially likely
  to match and consume the one instance being targeted. Mitigation:
  NEVER use a TEST_CASE opening macro as the anchor. Instead, anchor on
  the closing `}` of the NEW content being inserted (unique, since it was
  just written) and replicate the next test's full header inside new_str.
  Or anchor on a unique internal line from the next test's body. After any
  str_replace that inserts content adjacent to existing code rather than
  cleanly replacing a whole block, grep for the anchor text's distinctive
  substring immediately to confirm it's still present exactly once, in the
  right place -- in addition to (not instead of) compiling/running tests.
  The compile/test/grep step has caught this every time so far.

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

  **GPU-specific -- defer until GPU kernel work.** The CPU path already has
  `equalizer_step` for memory redistribution and OpenMP for parallelism; the
  work-pointer set gives near-exact equal-size work (any workload is at most 1
  off from others) which matters most for GPU warp/wavefront scheduling. When
  GPU kernel work starts: implement the two pointer sets in the delta-CSR
  layout, maintain them in synaptogenesis/pruning.
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

---

## Performance tests (deferred)

Integration tests (`tests/integration/`) validate correctness before merge but
take tens of minutes to run. A separate performance test suite should:

- Re-run the same tasks (pass_through, rare, copy, transformer) at fixed step
  counts and assert MSE <= known thresholds, flagging regression.
- Include the PyTorch comparison from `test_transformer.py` and assert the
  sili/torch MSE ratio stays within 1.5x. Ratios > 3x indicate a backprop bug.
- Be run on a reference machine (or CI) with recorded baselines so that changes
  to the C++ kernels or autograd can be caught before merge.
- Keep run-time under 10 minutes on a laptop CPU for the core set.

Current status: not implemented. Add after the gen_toy_mistral -> FoldedLayer
pipeline is working end-to-end so the sparse performance tests have a real
pre-trained weight source rather than random initialization.

---

## STREAMING (layer-by-layer) CONVERSION -- TOP PRIORITY, NOT AFTER REFACTORING

The working pipeline has top priority, and the pipeline must convert models on
the same 32-96 GB machine that runs them (24B bf16 = ~48 GB on disk; whole
state-dict load is impossible at 32 GB). Initial implementation landed in
sili/conversion/streaming_prune.py (two-phase: per-tensor sparsify -> per-
suffix fold, --no-stack fallback when suffix nnz exceeds mem budget). See
docs/requirements_vlm_streaming_rtac.md section 3 for the memory math and
section 7 for the remaining edge-case checklist.

## MoE EXPERT-MERGE (later)

No non-MoE 100B+ dense models exist to test truly-larger-than-RAM conversion,
so large-model conversion requires merging mixture-of-experts experts into one
giant sparse layer: CSR sparsity subsumes MoE routing sparsity (router picks
block-columns; CSR picks individual weights). Design: concat expert weights
block-wise, fold router scores into per-block importance at conversion time.
Later work -- after the dense streaming path is verified on 24B.

## CONV-KERNEL SPARSIFICATION (later)

Pixtral's patch_conv.weight (1024,3,14,14) = ~602K params, kept fully DENSE
in streaming_prune.py for now (ndim>2 branch). Two reasons: small in every
dimension (prior sparsification attempts on vision-tower-scale tensors found
poor sparsify ratios vs the 21M-scale text matrices where CSR actually pays
off), and CSR is 2-D only so ndim>2 needs a reshape-to-(out,-1) + orig_shape
restore on the way back out -- extra machinery not worth building until dense
is shown to actually cost something. If/when this is picked up: reshape
(1024, 3*14*14)=(1024,588), prune, CSR, restore original 4-D shape at
inference load time. Test first whether it sparsifies well at all before
building the restore path -- if density stays high, it may not be worth
doing regardless of whether the machinery exists.


## RTAC ADVANCED FEATURES (later)

Two items deferred from the RTAC curiosity agent (see
docs/requirements_vlm_streaming_rtac.md section 4 for the base implementation
-- PopArt and double critic are done; these two are not).

**Critic-through-trunk.** rtac.py recomputes values from the SHARED hidden
representation so the critic's gradient reaches the trunk weights, not just
the value head. In the current online loop this needs the PREVIOUS step's
forward pass to still be a LIVE Tensor with its autograd graph intact when
the critic loss is computed one step later -- but the current loop applies
SGD and clears every Tensor's graph at the end of each step (h_prev is
stored as a detached numpy array specifically to avoid holding stale graphs
across iterations). Two ways to fix this, both real refactors, not a small
patch:
  (a) Defer the SGD apply by one step: compute forward at t, hold the Tensor
      graph, do the critic-through-trunk backward at t+1 using weights as
      they were at t (values become slightly stale by one step, which
      one-step TD already tolerates for the target net).
  (b) Recompute h_prev's forward pass fresh at the critic step (rerun
      core_net.forward on the stored raw inputs) so a live graph exists
      again -- doubles the forward-pass cost per step and needs the raw
      inputs (not just h_prev) retained.
Not attempted when the base RTAC agent was built -- wanted correctness
confidence before landing it, and (a) touches the main loop's control flow
in a way that risks regressing the working zero-init learning signal.

**Replay buffer / batched updates.** rtac.py samples batches from a large
replay memory (Memory class, size 500k-1M in the paper's configs); the
current loop is strictly online (one transition, one gradient step,
immediately discarded). Adding replay means: a ring-buffer transition store,
batched forward/backward (currently every op in the RL training loop is
single-sample, e.g. EnergyDynamics.forward expects a single vector not a
batch dimension -- needs a batch-axis audit before this is safe), and a
decision on whether curiosity reward is recomputed at sample time (correct
but requires storing enough to re-render the Mandelbrot view) or cached at
collection time (cheaper, slightly off-policy). Substantially larger than
anything else in the RTAC work; not attempted given the emphasis on the
working pipeline over expanding scope.

## VISION: PER-PATCH SEQUENCE + SPATIAL MERGE (later)

UPDATE: the worst part of the old simplification is fixed -- test_mandelbrot_rl.py
used to mean-pool ALL patches together into one vector before projection,
destroying every bit of spatial layout (confirmed: with view=32, patch=8,
that was averaging all 16 separate 8x8 patches into a single 64-dim vector --
the network had no way to know WHERE any feature was in the frame, only
"what does an average local patch look like"). Now flattens the full view
directly (raw_dim = view*view + 3, the +3 being explicit position/zoom) so
every pixel occupies a fixed position in the input vector and a linear
projection can in principle learn to treat different spatial regions
differently.

That is still NOT the real Pixtral architecture's per-patch token sequence +
attention + 2x2 spatial merge (multi_modal_projector.patch_merger,
spatial_merge_size=2 per the real config.json) -- there is still no
attention operating over spatially-distinct patch tokens, just one big flat
vector through one linear layer. Implementing the real version still needs:
patches kept as a genuine sequence (not flattened into one vector either),
the vision tower's attention operating over that sequence, and the 2x2 merge
implemented as a constant gather/matmul (group 2x2 neighboring patch tokens,
concatenate their channels, project down) matching
patch_merger.merging_layer.weight's real shape
(VIS_HIDDEN, VIS_HIDDEN*MERGE*MERGE). Still a real architecture change from
the current (much improved, but still single-vector) input, not a small
patch -- deferred alongside the RTAC items above.


## KNOWN ISSUE: tests/unit/python/test_sili.py is stale against the current SparseLinearLayer API

Pre-existing, confirmed unrelated to any change made while fixing the
`pip install -e .` build (dated via `git blame` to 2026-07-01, days before
that fix). Surfaced because that fix was verified by running the FULL
`tests/unit/run_tests.sh` end-to-end for the first time in a while, not because
anything in `sili/cpu_backend.cpp`'s `SparseLinearLayer` binding changed
today.

Current constructor (sili/cpu_backend.cpp, `py::init<int,int,int,int>()`):
    SparseLinearLayer(n_inputs, n_outputs, max_weights, num_cpus=4)

test_sili.py's `make_layer()` helper and ~8 direct `_cpu.SparseLinearLayer(...)`
call sites assume an OLDER, more elaborate signature with separate concepts
that have since been consolidated or moved to other methods:
    _cpu.SparseLinearLayer(n_in, n_out, bw, budget, cpus)        # 5 args
    _cpu.SparseLinearLayer(8, 8, 1, BUDGET, 1, 5)                 # 6 args
    _cpu.SparseLinearLayer(1000, 1000, 10, 1)                     # 4 args,
        # but semantically wrong even though the COUNT happens to match:
        # 3rd positional is `max_weights` now, not a byte `budget`

61 failed + 23 errored (out of 103 collected) when run via
`tests/unit/python/run_py_tests.sh` / `tests/unit/run_tests.sh` -- all TypeErrors at
construction, all downstream of this one mismatch (TestConstruction,
TestForward, TestBackward, TestSynaptogenesis, TestEqualizer, TestToAbsolute,
TestPytorchLike, TestNumpyViews, TestSparseAttention, TestBandedAttention,
TestParallelPointers, TestSerialisation, TestBufferAccess -- essentially the
whole file). The C++ Catch2 suite (tests/unit/*.cpp, 612 assertions) and
tests/integration/* are unaffected and reliable; this is isolated to this
one legacy Python file.

Needs real investigation before fixing, not a blind signature patch: where
did the old `bw` (bandwidth) concept go -- folded into `max_weights`
entirely, or moved to a separate method? Is byte-level `budget` gone in
favor of an element-count `max_weights`, or available elsewhere (e.g.
`equalize_to_capacity`, discussed earlier this session)? What does the
trailing 6th arg in some call sites (parallel-pointer count?) map to now --
`TestParallelPointers` existing as its own test class suggests a
`.set_parallel_ptrs(...)`-style method may already exist and just isn'''t
what these older call sites use. Whoever picks this up should read the
CURRENT SparseLinearLayer binding's full method list in cpu_backend.cpp
first, then decide per-test whether to update the call to the current API
or whether the test itself is now redundant with something in
tests/integration/.