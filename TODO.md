# TODO — backburner, after the refactoring is done

Things that wait until *after* the active consolidation work is finished.
See `refactoring_todo.md` for the active priority queue, old-directory
verdicts, and anything currently blocking gen_toy_mistral working
end-to-end -- that document is where most of what used to live here has
moved (importance_scale, the parallel-pointers correction, the
SISLDOLayerV architecture fix).

## Correctness, needs a real fix eventually

- **The standalone `m.def("make_weights", ...)` pybind registration calls
  a stale 7-arg signature.** Module-level, not inside any class -- calls
  the base `make_weights<int,float>` template with an old signature (rows,
  cols, ptrs, indices, values, grads, importance -- a separate
  backprop-adjacent grads array that no longer exists as a concept) against
  the current 5-arg definition. Per clarification (see
  refactoring_todo.md): its actual purpose -- constructing a usable CSR
  weight layer from a few plain vectors for testing -- is already served
  by `delta_csr_from_absolute`, used in every test written this session.
  Likely just needs replacing with something built on that rather than
  debugging the old 7-arg call. Not done yet.

- **`SISLDOLayerV::load_weights` calls `make_weights_v`, which has its own
  internal type-conversion bug** (`csr.hpp:307`, "could not convert 'w'
  from FP4BiPacked-typed sparse_weights to TriValues-typed"). Per
  clarification (see refactoring_todo.md), this is now understood to be
  moot rather than needing its own fix: `SISLDOLayerV` is currently
  architecturally wrong regardless of this specific bug (still on
  `SparseLinearWeightsV`/TriValues/absolute-CSR when it should be on
  `SparseLinearWeightsDelta<S, DeltaCSRBiValues<float>, COL_TYPE>` --
  BiValues, not TriValues, since backward no longer stores a separate
  backprop array). Once `SISLDOLayerV` is rewired to that pattern
  (mirroring `DISLDOLayerV`'s already-verified upgrade), `make_weights_v`
  won't be called from there at all. Tracked as part of the SISLDOLayerV
  fix, not as a standalone bug to chase.

- **`sili.sparse_rnn.DISLDOLayer`/`SISLDOLayer` (Python) call C++ methods
  that don't exist on any currently-bound class -- `SparseRNNCell`/
  `SparseRNNAgent` cannot be constructed at all.** Found while adding
  dedicated `SparseRNNCell` tests (`tests/unit/python/test_sparse_rnn_cell.py`,
  see its module docstring for the full detail). `DISLDOLayer.__init__`
  calls `_cpu.DISLDOLayer(...)` and `SISLDOLayer.__init__` calls
  `_cpu.SISLDOLayer(...)` -- neither name is bound in `cpu_backend.cpp`,
  only `DISLDOLayerV` is. `_SparseLayerBase.step()`/`.decay()`/
  `.synaptogenesis()` additionally call `self._c.optim_weights(lr)`/
  `.decay_importance(rate)`/`.optim_synaptogenesis(...)` -- none of which
  exist on `DISLDOLayerV` OR `SparseLinearLayer` either. Three genuinely
  different C++-layer API generations exist in this codebase today:
  (1) whatever `DISLDOLayer`/`SISLDOLayer` assume (single-arg `forward`/
  `backward`, separate `optim_weights`/`decay_importance`/
  `optim_synaptogenesis` calls) -- never actually implemented/bound;
  (2) `DISLDOLayerV` (`forward(x, learning_rate)`/`backward(dy,
  learning_rate)`, `synap_row_step` instead of a probes+optim pair, no
  weight-decay/separate-optim methods at all);
  (3) `SparseLinearLayer` (`forward_dense`/`backward_dense` take
  `learning_rate` directly and apply the update inline, `build_probes`+
  `synap_step`, no separate optim/decay step) -- this is the one that
  actually works today, proven by `FoldedLayer` (used by
  `tests.integration.test_toy_mistral`) and `make_grown_sparse_layer` in
  `tests/integration/test_mandelbrot_rl.py` (used by `SparseCore`/
  `MistralCore`, i.e. the actual Mandelbrot experiment's sparse core).
  **Not on the MiniCPM5 conversion critical path** -- that work goes
  through `FoldedLayer`/`SparseLinearLayer` directly, never through
  `SparseRNNCell`. Real fix: rebuild `DISLDOLayer`/`SISLDOLayer` on
  `SparseLinearLayer`'s actual (working) API, which also means reworking
  `SparseRNNCell.step()`'s separate-call convention into
  `SparseLinearLayer`'s inline-learning-rate convention. Sized as its own
  task, not a quick patch -- `tests/unit/python/test_sparse_rnn_cell.py`
  has the tests already written and marked `xfail(strict=True)`, ready to
  flip green once this lands.

  **Noted while building `FoldedColumnLayer` (Phase A4)**: it landed on
  the exact same `h = input_proj(x) + recurrent(state)` shape
  `SparseRNNCell` already has (structurally deliberate, not
  coincidental -- see its docstring), but the two are NOT good
  subclass/merge candidates right now: `SparseRNNCell` bundles
  `EnergyDynamics`+`BranchingRatioTracker`+CSR-caching directly inside
  `forward()`, while `FoldedColumnLayer` deliberately leaves energy
  gating external (the caller composes it with `column_averaging_loss`,
  which needs to see the gated state specifically); `SparseRNNCell`'s
  `input_proj`/`recurrent` are the broken `DISLDOLayer`/`SISLDOLayer`
  above, while `FoldedColumnLayer`'s are `SparseLinearLayer`-based on
  purpose, specifically to avoid that bug. Once `DISLDOLayer`/
  `SISLDOLayer` are rebuilt on `SparseLinearLayer` (the fix above), both
  classes' `input_proj`/`recurrent` would share the same actual
  underlying primitive rather than just resembling each other --
  *that's* the point where extracting a shared minimal
  `h = a(x) + b(state)` base becomes genuinely motivated instead of
  premature (two real working examples instead of one).

- **A fired-but-not-selected neuron's energy grows without bound under a
  chronically repeated identical input -- this looks like curiosity/
  novelty-seeking pressure working as intended, not a bug.** Found while
  validating the column-averaging mechanism (Phase A3/A4) end-to-end with
  `FoldedColumnLayer` + `EnergyDynamics` on a FIXED input repeated every
  step: `_apply_energy_dynamics`'s aux_loss (specifically `energy_loss`,
  the per-neuron quadratic term) grew from ~0.08 to 177+ over 270 steps
  with no sign of stopping. Mechanism: step 3's fire-threshold handling
  applies a small refractory drain (`2 * activation_cost`) to every
  neuron whose `new_energy >= 2.0`, regardless of whether the LATER
  top-p gate (step 4) actually selects it into `kept_fire` -- a neuron
  that fires but consistently loses the top-p competition nets
  `drive - 2*activation_cost` per step (positive whenever
  `drive > 2*activation_cost`, true of the config used here:
  `drive=0.15, activation_cost=0.05`), so its energy keeps climbing with
  no ceiling as long as nothing about its input changes. Confirmed NOT a
  bug in the column-averaging code itself (isolated: `FoldedColumnLayer`/
  `column_averaging_loss` both converge correctly on their own, see
  `tests/integration/test_folded_column_layer.py`) -- this is
  `_apply_energy_dynamics`'s own behavior under a genuinely static input,
  and per direct feedback that's exactly the signature the curiosity/
  compression-progress citations (`CITATIONS.md`) describe: "nothing new
  is happening, pressure should build" is the correct response to a truly
  unchanging input, not a defect to route around. Every other existing
  usage in this codebase (Mandelbrot, RL agents, real token streams) never
  exercises this regime because their inputs vary step to step -- adding
  small per-step input variation made the aux_loss settle (~1-1.5) in the
  same test, consistent with that reading. Not changed here; worth
  deliberately harnessing later (e.g. as an input to Phase E's eventual
  action pathway -- "this region has been quiet too long" as a literal,
  already-present signal) rather than treating it as something to clamp
  away.

- **Synaptogenesis after `compact()` needs automatic handling, not just a
  loud failure.** FIXED: `synap_row_step` now throws a catchable
  `std::runtime_error` (with the row index and exact bytes/elements needed
  vs available) instead of silently reporting `did_work=true` while nnz
  never changes. Also fixed a related bug this exposed: `out_degree`
  bookkeeping was being updated even when the underlying rebuild never
  actually wrote anything. Added `expand_headroom()` (the opposite of
  `compact()` -- normalizes headroom to exactly `blank_fraction` of
  current content) as the explicit fix-it-yourself path; verified
  compact → throws → expand_headroom → works again, end to end.
  REMAINING: this is all manual. The real fix is automatic -- e.g. a layer
  transparently calling `expand_headroom()` itself the first time growth
  is attempted and fails, or `compact()` taking a flag for "but keep
  enough headroom for typical synaptogenesis rates" instead of
  normalizing all the way to zero. Needs actual design thought (how much
  headroom is "enough" without defeating compact's purpose), not a quick
  patch.

  Note on scope: `compact()`/`expand_headroom()` currently handle BOTH
  axes together (the ULEB128 index-byte buffer AND the values/importance
  buffer, keyed by `byte_start`/`byte_end` and `elem_start`/`elem_end`
  respectively) in one pass. These are genuinely independent axes with
  independent headroom budgets -- bundled here because both become
  useless after compaction, not because they're the same operation.
  Splitting into `compact_indices()`/`compact_values()` (and matching
  `expand_` variants) if a use case needs independent control hasn't come
  up yet but would be straightforward given the current implementation.

## Architecture decisions made, not yet fully executed

- **`SparseLinearLayer` has no bare `forward`/`backward`, deliberately,
  for now.** Only `forward_dense`/`backward_dense` and `forward_sparse`/
  `backward_sparse` exist. Per the active queue (refactoring_todo.md),
  the auto-dispatching version IS being built next (not indefinitely
  deferred) -- the underlying ops already exist and are tested
  (`hoyer_sparsify.hpp`, exposed as `hoyer_sparsify(x)` and
  `hoyer_score(x)`), using Hoyer's Sparsity Measure:

      hoyer(x) = (sqrt(n) - ||x||_1/||x||_2) / (sqrt(n) - 1)     in [0, 1]

  Two-stage design: `hoyer_score(x)` aggregates over the whole flattened
  batch (since forward_dense/forward_sparse are each invoked once per
  batch, not once per sample -- a per-sample answer isn't actionable at
  that granularity) for the ROUTING decision; `hoyer_sparsify(x)` gives
  each row (batch sample) its own k_estimate for CONSTRUCTING the actual
  CSR once routing has decided "sparse." Threshold decided (see active
  queue): route to sparse when `hoyer_score > 0.8`. Lives in Python-level
  wrapper methods, not the C++ hot path.
  REMAINING (explicitly deferred, not forgotten): the 0.8 threshold is a
  fixed constant for now. Could eventually become adaptive based on
  measured time performance instead -- note this in the wrapper methods'
  own docstrings when they're written, not just here.

- **Work pointers, not a mid-row resume mechanism.** Corrected design
  (see refactoring_todo.md for the full correction) -- two pointer sets:
  (1) roughly-equal-sized WORK regions for load-balanced OpenMP
  parallelization (not necessarily row-aligned), (2) row-beginning
  pointers (the normal row_ptr array). O(1) at runtime, no searching --
  strictly better than the originally-planned search-based mid-row resume
  mechanism it replaces. Synaptogenesis/pruning's job is just keeping the
  work-pointer set clean/valid. Not yet designed in detail or implemented
  against this repo's actual `DeltaCSRLayout`.

## Backburner (deliberately deprioritized, not forgotten)

- **fiber.hpp / old_fiber.hpp / parallel-growth concurrency** (from
  cpu_sparse_io). Was for dynamic in-place neuron growth; manual layer
  expansion (new larger buffer, copy) covers current needs. Revisit for
  future hardware where RAM can grow without a restart. There was
  reportedly a working neurogenesis test built on this
  (test_sisldo_neurogenesis.cpp) -- worth checking against once revisited.

- **unittest_sisldo.cpp** (parked out of the active build). Tests the
  TriValues high-precision path specifically, calls the old 7-arg
  make_weights signature. Many of its 7 test cases probably test
  still-relevant concepts (outer_product, top-k probe generation) that
  should become shared functions across the 4-bit/32-bit paths rather
  than duplicated -- triage each test case individually (keep + update,
  or drop as stale) rather than blanket-fix.

- **`coo.hpp` (cpu_sparse_io) parallel COO generation/sorting** for
  synaptogenesis. Current small-scale needs (diagonal init, or ~100
  synapse gen/deletions at a time) don't need the parallel version, and
  it would need adapting to the new memory layout regardless -- see
  refactoring_todo.md for the full note.

- **`csf.h`/`csf.cpp` (Compressed Sparse Fiber, cpu_sparse_io)** --
  backburner per direct guidance, not evaluated further.

## GPU / vision (mixed priority -- see refactoring_todo.md for the elevated items)

Most GPU/vision work from `sili_old` was confirmed worthless and dropped
(pyramid-conv variants, radacon/adacon, multi_matrix_inverse, to_spvec --
see refactoring_todo.md for the full verdict list). What's left:

- **`to_csr` / `to_coo` GPU shader groups (sili_old) confirmed genuinely
  useful** (fixed-IO-size guarantee, real PCIe bandwidth win), not yet
  integrated. Used a custom Kompute fork; kompute-python isn't well
  supported, so eventually wants its own runner system with GPU ops as
  part of a GPU "device" abstraction (this abstraction itself has been
  elevated to "later todo, not backburner" given the V-LLM/vision
  requirement -- see refactoring_todo.md). For now, normal Kompute is an
  acceptable stopgap. No GPU available in the sandbox this consolidation
  is being done in -- verification limited to "compiles, fails to find a
  GPU gracefully" until run on real hardware.

- **`from_csr` / `from_coo` GPU shaders don't exist** (only the `to_`
  direction does). Could reduce PCIe bandwidth pressure if sending dense
  activations/gradients to the GPU becomes the bottleneck. Per guidance,
  CSR-CSC conversion may work better for GPU purposes than from_csr/
  from_coo specifically -- cross-reference this session's disldo_gpu.py
  CSC-construction-via-argsort work if picking this up. NEW: worth trying
  whether Claude Fable 5 can generate working from_csr/from_coo shader
  implementations directly -- per direct suggestion, genuinely plausible
  given current model capability even though not a common benchmark task,
  worth an actual attempt rather than assuming it can't. GPU shader code
  can be tested without real hardware via glsl->spir-v->c++ transpilation
  or a software implementation like llvmpipe -- relevant for verifying
  whatever gets generated.

## Training target: Mandelbrot exploration via RL curiosity

After backprop and synaptogenesis are wired up end-to-end, a good first
training target is 2D complex-plane fractal navigation:

- Model outputs: movement ops (pan left/right/up/down, zoom in/out)
- Reward signal: the energy RL system already handles complexity-seeking
  implicitly through synaptogenesis/pruning dynamics -- zlib compression
  ratio of the viewport would be redundant on top of that
- Token vocabulary size is flexible: from raw (x, y) float pairs that
  feed directly into input neurons (no encoder/decoder), to discretised
  coordinate tokens with learned embeddings
- Self-similar structure at different zoom levels exercises the RNN state
  explicitly -- the state needs to track "where we are in the scan"
- The sparse structure of the Mandelbrot set (most of the complex plane
  escapes quickly; complexity lives at the boundary) naturally exercises
  sparsity enforcement: neurons should activate mainly for boundary inputs

## Training target: Mandelbrot exploration via RL curiosity

After backprop and synaptogenesis are wired up end-to-end, a good first
training target is 2D complex-plane fractal navigation:

- Model outputs: movement ops (pan left/right/up/down, zoom in/out)
- Reward signal: the energy RL system already handles complexity-seeking
  implicitly through synaptogenesis/pruning dynamics -- zlib compression
  ratio of the viewport would be redundant on top of that
- Token vocabulary size is flexible: from raw (x, y) float pairs that
  feed directly into input neurons (no encoder/decoder), to discretised
  coordinate tokens with learned embeddings
- Self-similar structure at different zoom levels exercises the RNN state
  explicitly -- the state needs to track "where we are in the scan"
- The sparse structure of the Mandelbrot set (most of the complex plane
  escapes quickly; complexity lives at the boundary) naturally exercises
  sparsity enforcement: neurons should activate mainly for boundary inputs

## Memory / synaptogenesis

**In-place insert/delete per delta-encoded row (DONE):** `delta_csr_row_insert_col` /
`delta_csr_row_remove_col` in `delta_csr_memory.hpp` replace the old
"read all, re-encode entire row from scratch" approach. For typical layers
(n_out <= 16384, column deltas fit in 1-2 bytes), each insertion needs 1-2
bytes of blank space, and each removal FREES bytes. Equal add/remove per
synaptogenesis step requires near-zero net blank space. `equalize_to_capacity`
uses the actual ULEB128 cost for the layer's column range instead of worst-case
(5 bytes/connection).

**Blank space padding optimization (FUTURE, low priority):** Currently
`equalize_to_capacity(max_row_weights)` sizes each row for exactly
`max_row_weights` connections at typical delta cost. Giving rows slightly MORE
blank (e.g. 10-20%) reduces the frequency of "no blank space" throws, which
require the calling loop to call `equalizer_step` and retry. But this trades
memory for reduced retry rate. Measure first on real model (Mandelbrot) before
tuning this knob.

**Work pointer set (FUTURE):** Two pointer sets (1) roughly-equal-size work
regions for load-balanced parallelization, (2) row-beginning pointers. See the
Corrections section in refactoring_todo.md.

