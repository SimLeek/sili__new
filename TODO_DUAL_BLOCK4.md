# Integrating block4 into SparseLinearLayer (production)

Context: `feature/sili-ell-benchmark` (PR #22, not merged) proved the design
works -- real speedup (2-13x, scaling with density, on the real MiniCPM5
checkpoint), verified no quality cost vs. disldo's own FP4 loss, working
forward+backward. That branch stays as prototype/reference; it isn't
production code (separate `dense_block4.hpp` header + separate
`dual_layer_proto` pybind module, not integrated into `SparseLinearLayer`
itself, and several rough edges: promotion is a single global fill-fraction
threshold with no synaptogenesis/pruning-time hook, backward rebuilds the
transpose from scratch every call).

This branch (`feature/sparse-linear-layer-dual-block4`) is the real,
focused integration: `SparseLinearLayer` itself becomes a dual-CSR+block4
structure, natively supporting everything the class already does
(`forward_dense`/`backward_dense`, `build_probes`/`synap_step` growth,
`compact`/`expand_headroom*` memory management, per-row/per-col scale and
importance accessors) -- not a second, bolted-on object.

## Comparison infrastructure (done)

- `sili__new_baseline/` -- a plain clone of `sili__new` at `main` (pre-
  block4, commit 7297458), for a true before/after comparison.
- `.venv_baseline` (sibling to the project's main `.venv`) -- isolated
  venv, `numpy<2.5.0`+`scipy` PyPI wheels (OpenBLAS, matching the main
  venv's own BLAS fix so that isn't a confound), editable-installed from
  `sili__new_baseline`. Verified isolated (resolves the baseline package
  correctly regardless of CWD).
- TODO: a bash script that runs the SAME benchmark/quality/correctness
  suite against both venvs (old vs. new `sili._cpu.SparseLinearLayer`)
  and reports a clean diff -- speed, memory, and quality (relative error
  vs. a dense reference, same methodology as `dual_matrix_vs_disldo.md`)
  side by side. Not written yet.

## Design decisions (settled)

1. **Promotion/demotion threshold**: a **compile-time fixed integer**
   (default 2 for a 4x4 tile, matching the 10%-of-16=1.6 reasoning),
   overridable at build time (CMake `-D` for the unit-test build,
   `extra_compile_args`/env-driven for the real `setup.py` pybind
   extension build -- both need the override wired through, not just
   CMake, since production installs via `setup.py`/pip, not CMake).
   Not runtime-configurable, not adaptive/per-region -- deliberately
   simple.
2. **Directional hook, not a symmetric check everywhere**: "synaptogenesis
   maybe promotes and pruning maybe demotes but synaptogenesis never
   demotes and pruning never promotes." Concretely:
   - On a NEW synapse added by growth: if that synapse landed in a tile
     currently in the SCATTERED representation, check just that tile's
     local live count (scan its <=4 rows' worth of CSR entries within
     the tile's column range -- bounded, cheap, not a full-structure
     scan) and promote if it now meets the threshold. Never demotes.
   - On a synapse REMOVED by pruning: if it was in a tile currently in
     BLOCK4, check that tile's live count (O(1), already a 16-slot byte
     scan) and demote if it now falls below threshold. Never promotes.
   - Checked at the specific growth/pruning EVENT that touched that
     specific tile, not a periodic sweep and not a blind check on every
     `synap_step` call regardless of what changed.
3. **Memory layout**: `SparseLinearWeightsDelta` gets a real block4
   member (not a separate prototype struct) alongside the existing
   `DeltaCSRWeights`. `compact()`/`expand_headroom*()` need to handle
   both.
4. **Backward's transpose cost**: not resolved yet, proceeding with the
   prototype's rebuild-every-call approach for a first working version;
   revisit once the rest is in and benchmarked -- may be acceptable for
   now (real profiling needed before optimizing further), matching this
   session's own "measure before assuming" pattern throughout.
5. **Tile size**: hardcoded to 4 as a compile-time constant (same
   override mechanism as the promotion threshold), matching the
   promotion threshold's own settled mechanism -- not templated/
   multi-size yet (that's the noted-not-scoped future idea in
   `prototypes/sili_ell/BLOCK4_NOTES.md`).

## Once the design is settled

- [x] Extend `SparseLinearWeightsDelta`/`SparseLinearLayer` (delta_csr_types.hpp,
      cpu_backend.cpp) with the block4 member + promotion/demotion logic.
- [x] Update `forward_dense`/`backward_dense` to combine both paths
      internally (callers shouldn't need to know about the split). Done at
      the `disldo_forward`/`disldo_backward` level (linear_disldo.hpp),
      which `SparseLinearLayer::forward_dense`/`backward_dense`
      (cpu_backend.cpp) already call through -- verified via ASan/UBSan
      correctness tests (test_disldo_block4_forward.cpp,
      test_disldo_block4_backward.cpp).
- [x] Update `build_probes`/`synap_step`/`synap_row_step` to handle
      promotion (and pruning paths for demotion). Done in
      `delta_csr_synap_row_step` (delta_csr_memory.hpp) via
      `block4_maybe_promote`/`block4_demote_tile`; `build_probes` itself
      unchanged (it only proposes candidates, doesn't need to know about
      block4). Verified end-to-end (test_disldo_block4_promotion.cpp):
      growth promotes at threshold, pruning demotes below it, out_degree
      only moves on real prunes not representation transfers. Found and
      fixed a real bug along the way, in two passes: first, using the
      weight nibble alone as a slot's liveness bit (indistinguishable
      from empty for a freshly-grown weight=0.0 synapse); then, after
      switching to "whole byte == 0", a DEEPER version of the same bug
      surfaced via the comparison script (see below) -- a genuinely-live
      synapse can have BOTH weight and importance quantize to
      FP4_TABLE's zero entry (common, not rare: 0.0 is nearest for
      anything near the origin). Block4Tile now has a real `presence`
      bitmask, independent of the stored byte, as the sole source of
      truth for liveness everywhere (forward's own value-based skip is
      the one exception, and is fine as-is -- 0 contributes 0 regardless
      of the liveness label).
- [x] Update `compact()`/`expand_headroom*()` for the combined structure.
      Turned out to need no changes -- both only touch
      `weights.connections`, and `block4` is a separate struct member
      untouched by that reassignment. Auditing the rest of this surface
      found two real bugs instead: `nnz()` (both block4-capable classes)
      and the `get_weights_vals`/`get_indices`/`get_ptrs`/`get_importance`
      save/export path only ever read `weights.connections`, so any
      block4-promoted synapse was silently invisible to both the reported
      count and any save taken after a promotion. Fixed via a new
      `delta_csr_combined_to_absolute()` (delta_csr_memory.hpp) that
      merges block4 entries back into column order per row; verified
      under ASan/UBSan plus a real pybind rebuild + Python round trip.
- [ ] Real per-row (or finer) FP4 value_scale calibration for block4,
      matching disldo's own (already proven necessary on real weights,
      not optional -- see PR #22's bug #4).
- [x] Bash comparison script: old venv vs. new venv, same suite, speed +
      quality. `scripts/compare_block4_venvs.sh` (+ `bench_block4_layer.py`,
      `diff_bench_reports.py`) runs the SAME script unmodified under both
      `.venv_baseline` and this repo's own venv -- block4 has no new
      Python API surface, so no separate baseline/new variants were
      needed. This is what surfaced the presence-bitmask bug above: nnz
      silently diverged between venvs before that fix, matches exactly
      after it (both at 2% and 15% synthetic density).
      RETRACTED FINDING: an earlier pass of this script (single
      sequential baseline-then-new run, no warmup) reported this branch
      as measurably SLOWER (0.57x-0.88x). That was a benchmark artifact,
      not real: `.venv` resolves numpy to scipy-openblas64
      (MAX_THREADS=64) while `.venv_baseline` resolves to plain system
      BLAS -- sili's own forward_dense/backward_dense never call BLAS at
      all, but an unbounded 64-thread pool contending for this machine's
      8 hardware threads slowed down ANY CPU-bound work sharing that
      process, block4-unrelated pure-C++ loops included. Confirmed via a
      native (no Python/pybind) A/B comparison -- same process, both
      repos' headers, identical production flags, 40 interleaved runs --
      landing at ~parity (0.95x-0.99x median), which is what sent the
      investigation to the Python layer instead. Pinning
      OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1 for the
      benchmark subprocess (now done in compare_block4_venvs.sh) restores
      parity at the Python level too.
      CURRENT FINDING (interleaved, BLAS-pinned, 7 repeats each): at 2%
      density (37 tiles / 93 synapses, 1.99% of nnz) medians are
      0.89x-1.12x; at 15% density (251 tiles / 806 synapses, 2.30% of
      nnz) medians are 0.91x-1.09x -- both within measurement noise of
      parity, no real win or loss on this synthetic uniform-random
      benchmark either way. Real-checkpoint validation via sili_peridot
      (not yet done) is still the actual test of whether structured
      (not uniform-random) sparsity produces PR #22's measured 2-13x --
      this synthetic benchmark was never going to reproduce that pattern
      regardless of the BLAS confound, since it doesn't cluster the way
      real transformer weights do. Memory not yet measured (only speed +
      correctness so far).
- [ ] Once verified clean (no regression, real win), sili_peridot should
      pin its `sili` dependency to this specific commit/tag rather than
      floating on whatever's installed -- exact mechanism (git submodule?
      pinned commit in requirements/setup instructions? a version tag on
      this repo?) not yet decided.

## Explicitly NOT changing

- The prototype branch (`feature/sili-ell-benchmark`, PR #22) and its
  code stay as-is, kept for reference/testing, not merged, not deleted.
- `sili_ell`'s own banked/packed codecs are not being integrated --
  this work is specifically about block4 + the existing disldo/CSR path.
