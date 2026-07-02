# TODO — sili_new consolidation

Tracks outstanding items from merging cpu_sparse_io + optim_merge + sili_old
into one repo. See sili_old/TODO.md (if still present) for historical GPU
design notes from that era — different scope, not merged into this file.

## Correctness / not-yet-fixed

- **SISLDOLayerV::load_weights doesn't compile.** `make_weights_v`'s call
  into `make_weights` has an internal type mismatch (`csr.hpp:307`, "could
  not convert 'w' from FP4BiPacked-typed sparse_weights to
  TriValues-typed"). Confirmed genuine and separate from the FP4/TriValues
  confusion that broke two other errors last pass (those were resolved by
  deleting the non-V SISLDOLayer/DISLDOLayer, which were the only callers
  mixing FP4BiPacked into TriValues-designed helpers). Not investigated —
  needs looking at `make_weights`/`make_weights_v`'s actual template chain
  in csr.hpp.

- **Synaptogenesis after compact() needs automatic handling, not just a
  loud failure.** FIXED this pass: synap_row_step now throws a catchable
  `std::runtime_error` (with the row index and exact bytes/elements needed
  vs available) instead of silently reporting `did_work=true` while nnz
  never changes — a silent failure here is the worst case, since training
  just stops improving with no signal why. Also fixed a related bug this
  exposed: out_degree bookkeeping was being updated even when the
  underlying rebuild never actually wrote anything. Added
  `expand_headroom()` (the opposite of `compact()` — normalizes headroom to
  exactly `blank_fraction` of current content, reusing
  `delta_csr_from_absolute`'s already-tested reservation logic rather than
  duplicating it) as the explicit fix-it-yourself path; verified compact →
  throws → expand_headroom → works again, end to end.
  REMAINING: this is all manual. The real fix is automatic — e.g. a layer
  transparently calling expand_headroom() itself the first time growth is
  attempted and fails, or compact() taking a flag for "but keep enough
  headroom for typical synaptogenesis rates" instead of normalizing all the
  way to zero. Needs actual design thought (how much headroom is "enough"
  without defeating compact's purpose), not a quick patch.

  Note on scope: `compact()`/`expand_headroom()` currently handle BOTH axes
  together (the ULEB128 index-byte buffer AND the values/importance
  buffer, keyed by byte_start/byte_end and elem_start/elem_end
  respectively) in one pass. These are genuinely independent axes with
  independent headroom budgets — a row's byte-headroom and element-headroom
  can differ, since ULEB128 cost varies with actual column gaps while
  element slots are fixed-width — bundled here because both become useless
  after compaction and reclaiming both together seemed natural, not because
  they're the same operation. Splitting into compact_indices()/
  compact_values() (and matching expand_ variants) if a use case needs
  independent control hasn't come up yet but would be straightforward given
  the current implementation.

## Architecture decisions made, not yet fully executed

- **SparseLinearLayer has no bare `forward`/`backward`, deliberately, for
  now.** Only `forward_dense`/`backward_dense` and `forward_sparse`/
  `backward_sparse` exist — an explicit choice was required so nobody
  defaults into dense out of habit and silently loses a real 10-100x
  speedup on genuinely sparse activations.

  Future auto-dispatching `forward`/`backward`: should decide dense vs.
  sparse using **Hoyer's Sparsity Measure**, not a naive "count near-zero
  elements" threshold. For a vector x of length n:

      hoyer(x) = (sqrt(n) - ||x||_1/||x||_2) / (sqrt(n) - 1)     in [0, 1]

  The relevant fact: for a vector with exactly k nonzero entries of equal
  magnitude, ||x||_1/||x||_2 = sqrt(k) exactly. For a realistic activation
  vector (a mix of large and small nonzero values, not exactly k-sparse),
  the same ratio still gives a smooth, principled estimate of the
  "effective" number of significant elements — k_estimate = (||x||_1 /
  ||x||_2)^2 — without needing an arbitrary epsilon-threshold count.

  Proposed algorithm for the auto-dispatching forward/backward:
    1. Compute ratio = ||x||_1 / ||x||_2 on the input (forward) or the
       output gradient (backward).
    2. k_estimate = ratio^2.
    3. Compare against a threshold (informed by where dense-vs-sparse
       execution cost actually crosses over for this layer's shape — not
       decided yet, needs benchmarking) to decide dense or sparse.
    4. If sparse: use k_estimate as the target k for a top-k sparsification
       pass (keep only the k_estimate largest-magnitude elements, zero the
       rest — discarding the long tail as noise, not just whatever happens
       to be exactly zero), convert the result to CSR, then call
       forward_sparse/backward_sparse.
    5. If dense: call forward_dense/backward_dense directly, no conversion
       overhead.

  This does NOT contradict "sparsification between layers is a separate
  function" (see class comment on SparseLinearLayer) — the top-k step
  stays a distinct, independently-callable operation; this auto-dispatch
  layer would just be a convenience orchestrator that decides whether to
  invoke it and with what k, rather than requiring the caller to always
  decide and call it manually.

- **SISLDOLayerV / DISLDOLayerV → one unified high-precision class.**
  SparseLinearLayer already unifies DISLDO-dense-input +
  SISLDO-sparse-input (forward_dense/backward_dense +
  forward_sparse/backward_sparse) in one class using
  DeltaCSRWeights<...,FP4BiPacked,...>. The V-suffixed
  classes should get the same treatment with
  DeltaCSRWeights<...,DeltaCSRBiValues<float>,...> — probably named
  something like SparseLinearLayerHighPrecision. DISLDOLayerV's
  dense-input half already works (verified this session, same generic
  functions as SparseLinearLayer). SISLDOLayerV's sparse-input half is
  currently broken (see above) and its synaptogenesis
  (sisldo_optim_synaptogenesis, using synap_mark_duplicates/
  synap_parallel_fill from parallel.hpp/coo.hpp) hasn't been evaluated
  against the newer delta_csr_build_probes/delta_csr_synap_row_step path.
  Per explicit guidance: parallel.hpp/coo.hpp have genuinely fast parallel
  ops (bitonic merge sort, exclusive scan) that might be worth adapting to
  the new (different) memory layout rather than automatically replacing
  with the simpler newer code — two ways to fix each broken piece, pick
  per-case, not resolved by default to "use the new code."

- **Parallel pointers not yet ported into sparse_struct.hpp.** Built and
  tested earlier this session (in a separate, now-superseded codebase) —
  a mid-row resume mechanism for parallelizing forward/backward across
  chunks of a single row without a full row scan. Confirmed via grep at
  the time: doesn't exist anywhere in this repo's history. Needed for
  DISLDO/SISLDO forward's OpenMP parallelization to scale well on rows
  with very uneven density.

- **importance_scale / rescale_importance not yet ported.** Built and
  tested earlier this session — per-layer fp32 scale so importance
  accumulates in true units before FP4 quantization, avoiding underflow
  for layers whose natural Hebbian-trace magnitude sits outside FP4's
  representable range. Same class of fix as value_scale problem this
  session's conversion pipeline needed, just for the live-training path
  instead of one-time conversion. Needs threading through
  delta_csr_forward/backward's importance update (the `1+|imp|`
  denominator), matching how it was done for disldo_forward/backward.

## Backburner (deliberately deprioritized, not forgotten)

- **fiber.hpp / old_fiber.hpp / parallel-growth concurrency** (from
  cpu_sparse_io). Was for dynamic in-place neuron growth; manual layer
  expansion (new larger buffer, copy) covers current needs. Revisit for
  future hardware where RAM can grow without a restart. There was
  reportedly a working neurogenesis test built on this
  (test_sisldo_neurogenesis.cpp) — worth checking against once revisited.

- **unittest_sisldo.cpp** (parked out of the active build). Tests the
  TriValues high-precision path specifically, calls the old 7-arg
  make_weights signature (pre-refactor: took a separate backprop
  array/flag that no longer exists — backward always updates value/
  importance now, lr=0 to disable). Per guidance: many of its 7 test cases
  probably test still-relevant concepts (outer_product, top-k probe
  generation) that should become shared functions across the 4-bit/
  32-bit paths rather than duplicated — triage each test case
  individually (keep + update, or drop as stale) rather than blanket-fix.

- **from_csr / from_coo GPU shaders don't exist** (sili_old only has the
  `to_` direction). Noted as a TODO since it could reduce PCIe bandwidth
  pressure if sending dense activations/gradients to the GPU becomes the
  bottleneck — to_csr/to_coo already solve half of that. Per guidance,
  CSR-CSC conversion may work better for GPU purposes than from_csr/
  from_coo specifically — cross-reference this session's disldo_gpu.py
  CSC-construction-via-argsort work if picking this up.

- **to_csr / to_coo GPU shader group** (sili_old, `sili/modules/to_csr/`,
  `to_coo/`) confirmed genuinely useful (fixed-IO-size guarantee, real
  PCIe bandwidth win) but not yet integrated. Used a custom Kompute
  fork; kompute-python isn't well supported, so eventually wants its own
  runner system with GPU ops as part of a GPU "device" abstraction. For
  now, normal Kompute is an acceptable stopgap. No GPU available in the
  sandbox this consolidation is being done in — verification limited to
  "compiles, fails to find a GPU gracefully" until run on real hardware.
