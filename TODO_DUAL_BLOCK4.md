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
   - **Demotion metric, settled**: not sum/average/L1/L2/weighted-sum of
     the tile's importances -- `is_greater_than` count against the SAME
     `importance_cutoff` pruning itself already uses. A synapse survives a
     real prune iff its importance exceeds that cutoff, so a tile's
     "effective live count" for demotion purposes is just how many of its
     16 importances would currently survive that same check -- reusing
     pruning's own threshold instead of inventing a second, separate
     aggregate criterion. Computable as 4 SIMD compare+cumulative-sum ops
     (one `Block4Vec` compare-and-hsum per tile row) rather than a 16-wide
     scalar loop. Not yet implemented (`count_live()` still does the
     simpler "byte != 0" scan) -- this replaces it once written.
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
- [x] Remove `presence`/`live_count` bitmask bookkeeping entirely --
      `Block4Tile` is now just `uint8_t data[16]` (128 bits/tile total, no
      per-tile index, no extra liveness state), every slot in a promoted
      tile treated as a real synapse unconditionally (weight=0.0 included),
      `count_live()` demoted to an on-demand O(16) cold-path scan used only
      by promotion/demotion/reporting. `layer.block4_tiles`/
      `block4_synapses` flat methods replaced with a proper nested
      `layer.block4.tiles`/`.synapses` view (`Block4View`, `cpu_backend.cpp`).
- [x] Make backward's block4 hot path genuinely SIMD (it was online-learning
      -- no reason backward should be scalar when forward is). Real
      auto-vectorization blockers found and fixed in order: `FP4_TABLE[code]`
      gather + per-element scale calls inside the batch loop (fixed by
      precomputing once per tile-column outside it); a genuine 4-way
      scatter into `mcol[]` (fixed via local-accumulate-then-flush); a
      *hidden* gather where the target column was actually contiguous but
      array-indexing obscured that from GCC's cost model (fixed via direct
      `col_base+lj` arithmetic + a `full_tile_cols` fast path); and a
      genuine cross-batch recurrence in the per-column weight/importance
      state that auto-vectorization's SLP pass could prove vectorizable but
      wouldn't commit to across plain scalar arrays -- required an explicit
      `Block4Vec` (GCC/Clang `vector_size` extension, float x4) to actually
      force it. Verified via real disassembly (not just `-fopt-info-vec`'s
      diagnostic text): 103 packed `vmulps`/`vaddps`/`vsubps` instructions
      in the compiled hot path.
- [x] Diagnose and fix a false "0 tiles instead of 251" alarm surfaced by
      `compare_block4_venvs.sh` after the SIMD rewrite. After extensive
      bisection (native C++ repro with exact numpy-dumped data, exact
      `equalize_to_capacity`/budget replication, matched RNG seed -- all
      ruled out), the real cause turned out to be trivial: `bench_block4_layer.py`
      still read the OLD flat `layer.block4_tiles`/`block4_synapses`
      properties via `getattr(layer, 'block4_tiles', 0)`, which silently
      fell back to 0 once those were replaced with `layer.block4.tiles`/
      `.synapses` above -- never a real regression in growth/promotion
      logic. Fixed via a `block4_counts()` helper that reads the new
      `.block4` view when present (and still returns `(0, 0)` unmodified
      against the pre-block4 baseline venv, which has neither API).
- [x] The residual tiny nnz mismatch `diff_bench_reports.py` still flags
      after that fix (35073 vs 35029 at 15% density, 4678 vs 4677 at 2% --
      both ~0.1%) is real but benign: quality error stays at float32 noise
      (~1e-6-1e-5) on both sides, growth throws are identical, and the
      divergence is proportionally tiny at both densities. Consistent with
      floating-point reordering (SIMD's backward now sums in a different
      order than the scalar baseline) nudging a handful of values across
      FP4's *stochastic*-rounding thresholds over 300 growth cycles --
      the same kind of seed-sensitive cascading already documented above
      for unseeded RNG state, just triggered by build-to-build math-order
      differences instead of an unseeded RNG. `diff_bench_reports.py`'s own
      "should match exactly" assumption is arguably too strict now that
      block4 legitimately changes evaluation order; left as a documented
      known-benign mismatch rather than "fixed" (there's nothing to fix).
- [x] Isolated the SIMD rewrite's real speedup on backward, separate from
      the venv-vs-venv comparison above (which is dominated by the
      non-block4 scattered-CSR majority at realistic uniform-random
      densities -- growth-driven promotion turns out to be fundamentally
      collision-limited on uniform-random data: increasing density *or*
      growth cycles 10x [0.1 density x 3000 cycles] still only produced
      305/16384 possible tiles, ~1.7% of nnz, because promotion needs 2+
      synapses landing in the *same* tile, which random insertion rarely
      does regardless of overall density). Instead, built a same-commit
      A/B (`SILI_BLOCK4_FORCE_SCALAR_BACKWARD` compile-time toggle in
      `linear_disldo.hpp`, off by default, benchmark-only) and a native
      harness that *directly* fills a chosen fraction of all possible 4x4
      tile positions (bypassing growth entirely) so the block4 hot path's
      share of total work is representative, not the <3%-of-nnz growth
      alone can produce. Result (12 interleaved reps at 30% tile fraction,
      5 each at 10/60/90%, `-O3 -march=native -ffast-math`, single
      thread):

      | tile fraction | scalar median | SIMD median | speedup |
      |---|---|---|---|
      | 10% (1638 tiles)   | 1.718ms  | 1.596ms  | 1.076x |
      | 30% (4915 tiles)   | 5.776ms  | 4.902ms  | 1.178x |
      | 60% (9830 tiles)   | 11.515ms | 9.845ms  | 1.170x |
      | 90% (14745 tiles)  | 18.687ms | 15.857ms | 1.178x |
      | 100% (16384 tiles, whole matrix is block4, no scattered CSR at all) | 9.393ms | 8.299ms | 1.132x (10 reps) |

      Every tile at every fraction above is already fully dense (all 16
      slots real, per the "no presence bitmask" design -- there's no
      partial-fill mode to test separately; "tile fraction" is how much of
      the matrix's tile-position grid got converted, not per-tile
      occupancy). Consistent, real, reproducible ~1.05x-1.27x per-rep
      speedup from the Block4Vec rewrite alone, flat across tile-fill
      fraction including the 100%-coverage case -- confirms this is a
      per-tile hot-path speedup, not something that scales with how much
      of the matrix block4 covers. Modest, not the 2-13x PR #22 measured
      on real checkpoint structure (that number is about block4 vs.
      fully-scattered CSR, a different comparison than SIMD vs. scalar
      *within* block4) -- but real and free money regardless.
- [x] Bit-shift FP4 encode/decode (`fp4quant.hpp`'s `fp4_decode_bits`/
      `fp4_encode_bits`), replacing `FP4_TABLE[code]`/the old 15-entry
      linear scan for CPU. Not an approximation: FP4_TABLE turns out to be
      exactly OCP MXFP4 E2M1 (2 exponent bits, 1 mantissa bit, bias 1)
      with one repurposed slot (code 8, sign=1/exp=00/mant=0, stores NaN
      instead of IEEE E2M1's -0.0) -- decode is a straight field re-bias
      into a float32's own exponent/mantissa (exact, verified bit-for-bit
      against FP4_TABLE for all 16 codes), encode is the standard
      low-precision-ML round-via-carry technique (add a rounding bias at
      the mantissa truncation point, let integer addition's carry
      propagate the rounding through the exponent) with one explicit
      3-way threshold split for the awkward subnormal/normal transition
      region ([0, 0.25)/[0.25,0.75)/[0.75,1.0) magnitudes). Per-direction
      (table isn't a frozen external format, this codebase's own choice):
      encode's exact tie-break no longer needs to match the old linear
      scan's arbitrary "lower table index wins" convention -- only that
      it lands on a nearest-or-tied value, verified over 516k swept +
      random values (`test_fp4_bitshift.cpp`, now wired into ctest, see
      below). `fp4_quantize()` is now defined in terms of `fp4_encode_bits`
      (same name/signature, faster body) -- FP4_TABLE itself, and
      `fp4_quantize_stochastic`'s own interpolation logic, are UNCHANGED
      and still lookup-table-based, per direction: keep the table
      available for GPU/other-device use and the non-block4 scattered
      path, this only replaces the CPU-side deterministic encode/decode.
      `block4_vec_decode_fp4` (block4.hpp) is the 4-wide SIMD version
      (same bit formula, GCC vector-extension compare+blend instead of
      if/else -- verified bit-exact against the scalar version for all
      65536 4-tuples of codes), wired into both forward's and backward's
      block4 precompute (replacing `FP4_TABLE[code]` there).
- [x] Real gap found and fixed: `test_disldo_block4_*.cpp` (4 files) were
      never actually wired into the test suite at all -- own `int main()`,
      never added to `CMakeLists.txt`'s `sili_tests` (would collide with
      Catch2's own `main`) or any other build target, only ever run via
      ad-hoc manual `g++` invocations during this integration's own
      development. Fixed via a separate `SILI_STANDALONE_TESTS` CMake
      loop (own executables + `add_test`), `run_cpp_tests.sh` now runs
      `ctest` (covers both `sili_tests`'s Catch2 cases and these) instead
      of just `./sili_tests` directly. Added `test_fp4_bitshift.cpp` to
      the same list.
- [x] Real, measured bug found via `valgrind --tool=callgrind` line-level
      profiling (prompted by "we should still be getting ~4x, what isn't
      SIMD" -- the direct answer): `block4_vec_hsum`/`block4_vec_broadcast`
      (block4.hpp) used a `for (i=0..BLOCK4_TILE)` loop indexing a
      GCC vector-extension type with a RUNTIME variable -- this compiles
      to genuine scalar memory traffic (GCC can't treat a
      runtime-indexed vector-extension lane as a register), not the
      single broadcast/horizontal-add instruction the code looked like it
      should be. Confirmed via profiling: these two functions alone were
      ~20% of ALL instructions in a profiled `disldo_backward` call.
      Fixed by hardcoding the 4-lane case directly (brace-init literal /
      constant-index sum) instead of looping -- justified since this file
      already explicitly commits to exactly 4 lanes (see the file's own
      "don't widen this to be safe for a size this isn't" comment). Found
      and fixed the same bug in a smaller form in both precompute sites'
      code-gather/decode-extract steps too. Combined effect (30%
      tile-fill fraction, 12+ interleaved reps): total instruction count
      down ~13% (560M -> 485M `valgrind --tool=callgrind` Ir count), SIMD-
      vs-scalar backward speedup improved from ~1.05x-1.27x to
      ~1.2x-1.9x. Remaining top costs, per the same profiling pass: the
      batch loop's own control overhead and genuine multiply-accumulate
      work (~40% of instructions, `linear_disldo.hpp`), and the writeback
      (`fp4_quantize_stochastic`, still table/interpolation-based, NOT
      yet SIMD -- ~31%, unchanged by this pass) -- the next real target
      if pursuing this further, per direction ("fp4 encode/decode can be
      made simd, though this one specifically isn't 4x... uleb128 can be
      made simd... all of backward/forward/optim ops should be able to be
      fully simd for block4").
- [x] SIMD stochastic FP4 encode (`block4_vec_quantize_stochastic_fp4`,
      block4.hpp), replacing block4's writeback `fp4_quantize_stochastic`
      calls (8 scalar calls/tile -> 2 SIMD calls/tile). The old
      `FP4_SORTED_IDX`-linear-bracket-scan algorithm doesn't translate
      directly to bit tricks (nonuniform table spacing), but its
      underlying MATH does, in two pieces: for |v| >= 1.0 (E2M1's
      "normal" region), the classic interpolation-fraction-equals-
      discarded-mantissa-bits property (provable algebraically -- both
      neighbouring representable values share one IEEE exponent bracket)
      makes the standard ML dithered-rounding trick exact: add a UNIFORM
      RANDOM integer spanning the discarded bits' range (not the
      deterministic encoder's fixed bias) and let integer-add carry
      propagate the rounding, same structure as `fp4_encode_bits` just
      probabilistic instead of round-to-nearest. For |v| < 1.0
      (subnormal/normal-transition magnitudes 0/0.5/1.0), the
      interpolation fraction is just linear in v (`2v` or `2v-1`), no
      bit tricks needed there. Per direction (table isn't a frozen
      external format): only required to stay UNBIASED
      (E[decode(stochastic_encode(v))] == v, verified via 200k-sample
      mean-convergence + saturation + exact-value + NaN-slot checks,
      both scalar `test_fp4_stochastic.cpp` and the 4-wide SIMD version)
      -- NOT to reproduce the old scan's exact per-draw sequence for a
      given seed (checked first: no test anywhere pins exact stochastic
      codes for a seed, only uses seeding for general run-to-run
      reproducibility of aggregate/statistical outcomes). `FP4_SORTED_IDX`
      and the scalar scan-based algorithm are gone from
      `fp4_quantize_stochastic`'s own CPU implementation (replaced with
      the same dithering formula, scalar form) -- `FP4_SORTED_IDX` itself
      stays defined, for GPU/other-device use per direction (documents
      the sort order, generically useful even if unused by this file's
      own scan anymore).

      RNG draws stay genuinely scalar (4 independent
      `fp4_stochastic_next_u64()` calls per SIMD call, one per lane) --
      the SIMD win is in the branch/arithmetic, not the RNG itself,
      which is a handful of xorshift ops regardless of lane count.
      Callgrind confirms this was the right target: `fp4quant.hpp`'s
      absolute instruction contribution to a profiled `disldo_backward`
      run dropped from 150M to 32M (~78%) after wiring this in --
      combined with the earlier hsum/broadcast fix, total instruction
      count is down ~23% from where this window started (560M -> 432M),
      and the isolated SIMD-vs-scalar backward speedup (direct-fill
      harness, 30% tile fraction) improved from ~1.05x-1.27x to
      ~1.8x-3.0x. Remaining top costs per the same profiling pass are now
      genuine work, not more of the same bug class: `block4_vec_hsum`
      (called every batch step, not a bug anymore -- just called a lot),
      real per-batch multiply-accumulate, and the (now unavoidably
      scalar, by design) 4 RNG draws per tile.
- [ ] Not yet done: uleb128 SIMD decode for the non-block4 scattered CSR
      path (a wholly separate, already-referenced-elsewhere branch/idea,
      out of block4's own scope); OMP scheduling tuned so different
      threads decode/process far-apart regions (reduces redundant
      cache-line contention on uleb128 decode specifically). Neither
      started yet.
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
