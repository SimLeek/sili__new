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
