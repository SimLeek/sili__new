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

- **Synaptogenesis is inert after compact(), not just suboptimal.**
  Confirmed via direct test (see conversation): after compact() removes all
  headroom, build_probes still generates real candidate probes, but
  synap_row_step's underlying delta_csr_row_rebuild has nowhere to grow
  into, so every row_step call reports `did_work=true` while nnz never
  actually changes — silently inert, not crash-safe-but-slow. Root cause:
  synap_row_step doesn't check/propagate row_rebuild's actual success/
  failure. Needs: either (a) synap_row_step calls reserve_indices/
  reserve_values for a row before rebuild if it's about to grow past
  capacity (defeats compact's compactness immediately, so probably not
  ideal as a default), or (b) an explicit "wake this row up" step the
  caller invokes deliberately before resuming training on a compacted
  model, or (c) something else — needs actual design thought, not a quick
  patch. Should just work at all after this, not necessarily efficiently.

## Architecture decisions made, not yet fully executed

- **SISLDOLayerV / DISLDOLayerV → one unified high-precision class.**
  SparseLinearLayer already unifies DISLDO-dense-input +
  SISLDO-sparse-input (forward/backward + forward_sparse/backward_sparse)
  in one class using DeltaCSRWeights<...,FP4BiPacked,...>. The V-suffixed
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
