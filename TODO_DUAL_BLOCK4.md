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

## Next redesign: ULEB128 tile indexing + sparse/dense hybrid tile encoding

Per direction (2026-07-29): `Block4Store`'s `unordered_map<uint64_t, Block4Tile>`
costs 64 bits/tile just for the index key (128 tile + 64 key = 192 bits
total), where the scattered path's own ULEB128 delta-encoded column index
typically costs only 8-16 bits (confirmed empirically this session: 100%
single-byte at realistic density). Two separable pieces, in priority
order:

### Part A -- ULEB128 tile indexing (replaces the hash map)

Give `Block4Store` its own `DeltaCSRLayout` (row_ptr/byte_ptr bookkeeping,
already value-agnostic, reused as-is) + ULEB128-encoded `indices_buf` over
BLOCK coordinates (block_row = row/4, block_col = col/4) + a parallel
`std::vector<Block4Tile>` values array, instead of the hash map + separate
`by_block_row` reverse-index. 128 (tile) + 8-16 (uleb128 tile index) =
136-144 bits total, down from 192.

Not a drop-in reuse of `delta_csr_row_insert_col`/`delta_csr_row_remove_col`
-- those are shaped around `ValueAccessor<VALUES_TYPE>`'s weight+importance
FLOAT PAIR interface (`insert_col(..., value_type weight, value_type
importance)`), and `Block4Tile` is one opaque 128-bit blob, not a
(weight, importance) pair -- forcing it through that exact signature would
be a worse fit than writing a parallel set of functions. Plan:

- [x] `block4_row_insert_tile`/`block4_row_remove_tile` (block4.hpp) mirror
      `delta_csr_row_insert_col`/`remove_col`'s *algorithm* exactly (uleb128
      delta re-encoding, byte-shift via memmove, headroom checks) but
      operate on `std::vector<Block4Tile>` directly -- not a call to the
      existing functions (confirmed: those are shaped around
      `ValueAccessor<VALUES_TYPE>`'s weight+importance FLOAT PAIR, wrong
      fit for one opaque 128-bit blob). `DeltaCSRLayout`/`DeltaCSRRowCursor`
      (delta_csr_types.hpp) ARE reused directly, unchanged -- confirmed
      value-agnostic, needed only a header-include reorder (block4.hpp now
      included after both, not before -- see delta_csr_types.hpp). Also
      needed `block4_row_shift`/`block4_grow_last_row`/
      `block4_ensure_row_headroom` (new -- Block4Store's own growth-on-
      demand, exactly-enough-for-one-more-tile increment, no
      equalize_to_capacity-style API needed given block4's inherently
      small population) -- not originally called out as separate line
      items but a necessary part of "give it its own capacity management,"
      mirroring `SparseLinearLayer::equalize_to_capacity`'s own real
      last-row special case (delta_csr_shift_row's algorithm intentionally
      no-ops on the last row; growing it needs a different, simpler path).
- [x] `Block4Store` reworked: `DeltaCSRLayout block_layout` +
      `std::vector<uint8_t> indices_buf` + `std::vector<Block4Tile>
      tile_values`, hash map and `by_block_row` both gone. Public API
      shape preserved (`get_or_create`/`find`/`erase`/`n_tiles`/
      `live_synapses`) plus a new `init(n_in, n_out)` (sizes the block-
      granularity layout, zero initial per-row headroom -- growth is lazy
      on first insert, matching the whole point of this redesign: don't
      reserve space most block-rows will never use). `find`/`get_or_create`
      are now O(row_nnz) cursor walks, not O(1) hash lookup -- measured,
      not assumed: total instruction count in a profiled disldo_backward
      run was UNCHANGED (716.2M vs. 715.8M pre-change, <0.1% difference) --
      block-row populations are small enough (block4's own collision-
      limited growth, established earlier this session) that the
      complexity change is free in practice.
- [x] Real bug found via ASan while stress-testing (not hypothetical):
      `get_or_create()` called directly on a never-`init()`'d Block4Store
      (`block_layout.rows == 0`) indexed `byte_start`/`elem_start` out of
      bounds -- a real SEGV, caught in my own `bench_block4_direct.cpp`
      scratch harness (which called `get_or_create` directly, bypassing
      promotion's safety net). Fixed two ways: (1) `block4_maybe_promote`
      (the ONLY place a tile is ever first created through the real growth
      path) now lazily self-inits from `weights.connections.layout`'s own
      rows/cols if not yet sized, so no caller going through promotion
      needs to remember an explicit `init()` call; (2) `get_or_create()`
      itself now throws `std::out_of_range` with a clear message for an
      out-of-bounds/uninitialized block-row, instead of silently
      corrupting memory, for any caller (tests, future code) that invokes
      it directly. `SparseLinearLayer`'s own constructor (cpu_backend.cpp)
      calls `init()` explicitly too, belt-and-suspenders.
- [x] All ~8 point-lookup call sites in `delta_csr_memory.hpp`
      (`block4_maybe_promote`, `block4_demote_tile`, the discovery/step
      hooks in `delta_csr_synap_row_step`, `delta_csr_combined_to_absolute`)
      and the 2 flat-tile-collection preambles in `linear_disldo.hpp`
      (forward's and backward's) updated -- `by_block_row.find(br)`-style
      checks became `br < block_layout.rows` + a `row_cursor(br)` walk;
      the flat-collection loops now walk every block-row's cursor instead
      of iterating a hash map, order doesn't matter to the (already fully
      parallel, no cross-tile dependency) per-tile math that follows.
- [x] Multiply/divide-by-4: `Block4Store::init` sizes `block_layout.rows/cols`
      as `ceil(n_in/BLOCK4_TILE)`/`ceil(n_out/BLOCK4_TILE)`. Verified
      against non-4-divisible dimensions (511x509) under ASan/UBSan with
      3000 growth cycles -- exercises the boundary/partial last block-row
      and block-col cleanly, no errors -- plus the direct-fill stress test
      below (100% tile-fill fraction touches every block-row including
      the last one).
- [x] All 4 `test_disldo_block4_*.cpp` files updated (manual
      `tiles[Block4Store::key(br,bc)]` construction -> `get_or_create(br,bc)`
      + `init(n_in,n_out)`) -- all still pass, including
      `test_disldo_block4_promotion.cpp`, which drives real promotion/
      demotion through `delta_csr_synap_row_step` (the actual growth path),
      not hand-built fixtures, so it's real coverage of the new insert/
      remove machinery, not just the manually-constructed cases.
- [x] Verify: full ctest suite (86 tests, same pre-existing 2-4 flaky
      thread-scheduling-order failures as before this change, confirmed
      flaky not regressed by re-running individually); ASan/UBSan on all
      4 block4 test files individually AND three purpose-built stress
      harnesses -- `bench_realistic_mixed` (natural growth, up to 8000
      cycles, multiple densities including 0.3/0.05, both 512x512 and
      511x509), `bench_block4_direct` (artificial 10%/50%/100% tile-fill
      via direct `get_or_create` calls -- the 100% case inserts all
      16384 possible tiles in one pass, exercising every block-row
      including the last/boundary one) -- all clean, zero ASan/UBSan
      findings; a real pybind rebuild + `bench_block4_layer.py`
      correctness check (quality err unchanged at ~5.7e-6, nnz/tile
      counts sane). REAL measured bit budget (not just the 136-144
      estimate): 88 tiles at 512x512/15%-density/300-growth-cycles used
      exactly 88 bytes of index buffer -- **136 bits/tile measured**
      (128 tile + 8-bit/1-byte index), the low end of the estimate,
      confirming the scattered path's own "100% single-byte at realistic
      density" finding applies here too (block-column values 0-127 at
      this problem size, n_out/BLOCK4_TILE=128, all fit in one ULEB128
      byte). Real, measured 29.2% reduction from the old 192 bits/tile.

### Part B -- sparse/dense hybrid tile encoding (scalar, threshold-configurable, do after Part A)

**Prototyped and measured in `/tmp` first, per direction, before touching
the repo.** Layout (16 bytes, matching the tile's 128-bit budget):
byte[0]=count (0-10), bytes[1..5]=10 position nibbles (2-bit local row +
2-bit local col, 2/byte), bytes[6..15]=10 value bytes (byte-identical
format to today's dense `data[]`). 10\*12+8=128 bits exactly (the
original math holds); 11 active synapses would need 132 bits, over
budget, so 11+ stays dense -- this arithmetic is exact, not a tuning
knob, even though the SWITCH POINT itself (see below) now is.

**SIMD unpack: tried, measured, real negative result -- scalar wins.**
Three variants prototyped and correctness-verified (55k randomized
round-trips, 0 failures, all three bit-exact against a scalar reference):
1. The originally-planned binary-decomposition idea (turn the runtime
   count into up to 4 compile-time-fixed-size unrolled blocks of 1/2/4/8
   one-element SIMD scatter ops -- broadcast position, broadcast value,
   compare-against-lane-index, and, or -- cursor advanced via OR, which
   is provably identical to addition here since it only ever combines
   distinct powers of two in increasing order).
2. A one-shot `__builtin_shuffle`-based gather (build a 16-byte index
   vector from the position nibbles, one shuffle instruction places
   every value).
3. A variant of (2) avoiding a runtime-indexed vector-extension write
   when building the index vector (the same anti-pattern class already
   fixed once this session, `block4_vec_hsum`/`broadcast`) -- built the
   index as a plain scalar array first, loaded as a vector once.

Confirmed via disassembly this was real SIMD codegen (`vpbroadcastb`/
`vpcmpeqb`/`vpand`/`vpor`), not another hidden version of that same bug.
Benchmarked per EXACT count (not a mixed average, which would hide
whether scalar only wins at the high end) across the full 0-10 range,
1M tiles/count, 5 interleaved repeats: **plain scalar unpack (a trivial
`for i in 0..count: dense[pos[i]] = val[i]` loop) wins outright at EVERY
count from 0 to 10** -- the gap actually WIDENS with count rather than
favoring SIMD at scale (binary-decomposition variant: 1.16x slower at
n=1, 2.15x slower at n=8; one-shot shuffle: up to 5.9x slower). Each SIMD
scatter costs ~5 real vector instructions per element; the scalar loop's
per-element cost is one indexed byte store -- for N this small (<=10
elements, 16 bytes total), the compiler's own scalar codegen already
beats any hand-rolled SIMD attempted here. Conclusion: **trust the
compiler, use scalar unpack/repack.** (Scratch harness:
`sparse_tile_unpack.cpp`, not part of the repo -- kept as a reference of
what was tried in case someone else has a sharper idea later.)

**Design pivot, per direction:** given scalar wins, the whole
sparse-mode encoding is no longer mandatory/automatic at a hardcoded
threshold -- it becomes an OPTIONAL, per-layer (or per-network)
**compression parameter**: a configurable switch point (default 10,
matching the exact 10\*12+8=128 arithmetic above, but user-adjustable)
that decides, per tile, whether to store compactly (sparse mode, scalar
pack/unpack, saves memory, costs a small scalar unpack/repack on every
access) or always store dense (128 bits flat, no pack/unpack cost at
all, matching today's behavior exactly). Setting the switch point to 0
disables compression entirely (always dense, byte-for-byte today's
behavior) -- a real, meaningful lever between "slower, more memory-
efficient" and "faster, uncompressed," not a fixed design decision baked
into the format.

- [x] Design the switch-point parameter's home: per-`SparseLinearLayer`
      (simplest, matches how e.g. `num_cpus` is already a per-layer
      constructor arg) vs. some network-wide default with a per-layer
      override -- decide before implementing, since it affects the
      constructor/pybind signature.
      **Decided (per direction): lives on `Block4Store` itself**
      (`Block4Store::switch_point`, default `BLOCK4_SPARSE_MAX_COUNT`=10),
      exposed read/write on the layer via `layer.block4.switch_point`
      (`Block4View`, `cpu_backend.cpp`) -- accessible/modifiable from
      `SparseLinearLayer`/`DISLDOLayerV` without adding a constructor arg.
- [x] Implement the scalar pack (dense -> sparse, i.e. repack) and unpack
      (sparse -> dense) functions in block4.hpp, using the winning
      scalar approach from the prototype above -- straightforward,
      already proven correct and fast, no SIMD needed.
      `block4_sparse_pack`/`block4_sparse_unpack` (block4.hpp): 16-byte
      sparse layout is byte[0]=count(0-10), bytes[1-5]=10 packed 4-bit
      position nibbles (2-bit local row + 2-bit local col, 2/byte),
      bytes[6-15]=10 value bytes (byte-identical to the dense format's
      per-slot byte). `SILI_BLOCK4_SPARSE_MAX_COUNT`=10 is a
      `static_assert`-enforced exact constant (10*12+8=128 bits exactly;
      11 would need 132, over budget) -- NOT the same knob as the
      runtime-configurable `switch_point`.
- [x] `Block4Tile` (or a new type wrapping it) needs a mode discriminator
      -- likely just "count <= switch_point" checked against the stored
      byte layout's own count field when in sparse mode, vs. today's
      always-16-bytes-dense assumption everywhere forward/backward reads
      `tile.at(li,lj)` directly. Decide how forward/backward access this
      without a full decode-to-scratch on every hot-path call when a
      tile is ALREADY dense (should stay a zero-cost direct read in that
      case, same as today) -- only sparse-mode tiles pay the unpack cost.
      **`Block4StoredTile`** bundles `uint8_t data[16]` + `bool is_sparse`
      in one struct (not a parallel array) so a single memmove during row
      insert/remove/shift keeps both fields in sync atomically -- the
      cost is per-tile, not per-byte, amortized over the >=2 synapses
      every promoted tile holds. Access goes through a new RAII
      **`Block4TileHandle`** (returned by `find()`/`get_or_create()`
      instead of a raw `Block4Tile*`): caches `Block4StoredTile*` at
      construction for the dense fast path (`.at()` is a direct pointer
      read/write, zero-cost, matches pre-redesign behavior exactly); for
      sparse tiles, unpacks ONCE into an internal `uint8_t scratch_[16]`
      at construction, and `.at()` reads/writes scratch. Move-only.
- [x] Wire pack/unpack into wherever a tile crosses the switch point
      (promotion time already scans/builds a tile from scratch; backward's
      writeback changes byte values in place today via
      `tile.at(li,lj) = ...`, which needs to become "unpack once at the
      start of this tile's processing if sparse, work on a dense
      scratch buffer, re-pack once at the end if now `<=` switch point,
      stay dense otherwise" -- avoid unpack/repack more than once per
      call).
      Handled by the handle's destructor, not automatically on every
      write: for DENSE tiles the destructor does nothing (no per-call
      O(16) scan -- compression is a deliberate, explicit decision, never
      an automatic side effect of a forward/backward call); for SPARSE
      tiles that were touched (`dirty_`), the destructor re-packs once,
      promoting to dense if the live count now exceeds `switch_point`.
      A new **`Block4Store::maybe_compress(br, bc)`** is the only place
      that packs dense -> sparse, called explicitly at the same
      promotion-event checkpoints that already decide block4<->scattered
      demotion (`block4_maybe_promote`'s two branches), so compression
      never adds cost to every forward/backward call, only to actual
      structural growth events.
      **Real lifetime hazard, found via auditing every `find()`/
      `get_or_create()` call site in `delta_csr_memory.hpp`:** two sites
      (`block4_demote_tile`, and Step 5's pruning check in
      `delta_csr_synap_row_step`) have a handle for (br,bc) still in
      scope when a later call in the same scope erases that SAME
      (br,bc) via `Block4Store::erase()` (directly, or transitively
      through `block4_demote_tile`). A naively-cached-pointer handle's
      destructor would write through a dangling pointer after that
      erase()'s internal memmove. Fixed by having the destructor
      re-fetch the sparse case's storage slot **by coordinate**
      (`raw_find(br,bc)`), not a cached pointer -- gracefully no-ops if
      the entry is gone -- plus explicit nested-block scoping at both
      call sites as defense-in-depth/readability.
- [x] `count_live()`/promotion/demotion: sparse mode's count is free (the
      stored count byte), no O(16) scan needed there; dense mode keeps
      today's scan.
- [x] Correctness tests: round-trip pack/unpack for all counts 0-16,
      the switch-point boundary in both directions, ASan/UBSan, plus a
      test with switch_point=0 (compression fully disabled) verifying
      byte-identical behavior to Part A's dense-only tiles.
      `tests/unit/test_block4_sparse_tile.cpp` (wired into
      `SILI_STANDALONE_TESTS` in `tests/unit/CMakeLists.txt`, runs under
      ctest): 22000-case pack/unpack round trip (counts 0-10, 2000
      trials each), dense passthrough, `switch_point=0` disables
      compression, explicit `maybe_compress` in both directions
      (compresses/doesn't), sparse->dense promotion mid-lifetime on
      write, the erase-while-handle-alive lifetime hazard (both sparse
      and dense cases), move semantics. All pass under both `-O0`/ASan/
      UBSan and `-O3 -march=native`/ASan/UBSan. Full `ctest` suite (87
      tests) also run clean apart from 4 known pre-existing flaky
      failures unrelated to block4 (stats/num_cpus/importance_scale --
      see `run_cpp_tests.sh`'s own history).
- [x] Real, measured bit budget across a realistic count distribution
      (not just endpoints) at a few different switch-point settings, and
      the real speed cost of the scalar pack/unpack call overhead itself
      (small per the prototype above, but should be measured in the real
      forward/backward hot path, not assumed from the isolated
      microbenchmark).
      Realistic mixed layer (512x512, 15% density, 500 growth cycles +
      50 backward calls, default `switch_point=10`): `n_tiles=90`,
      `sparse=18 dense=72` (20% of live tiles genuinely compressed),
      `live_synapses=1068`. Clean under ASan/UBSan.
      **A real, worth-documenting finding about `switch_point` itself:**
      once a tile is promoted into block4 it is treated as a dense 4x4
      micro-block for forward/backward purposes -- `disldo_backward`'s
      Hebbian update touches all 16 local `(li,lj)` slots on every call
      it processes that tile, not just the synapses that originally
      triggered promotion. So a promoted tile's live count climbs toward
      capacity as ordinary training proceeds, independent of the growth
      event that created it. Verified with a periodically-instrumented
      stress run at `switch_point=3` (barely above
      `BLOCK4_PROMOTE_MIN_LIVE`=2): tiles were genuinely sparse right
      after growth (`sparse=15` of 22 live tiles), but 30 subsequent
      `disldo_backward` calls with real learning (`lr=0.01`) monotonically
      drove `sparse` to 0 within ~15 calls as each tile's live count grew
      past 3. This is NOT a bug -- `maybe_compress` is deliberately
      never re-checked on every write (see above), so a tile that grows
      past `switch_point` after being compressed correctly stays
      compressed only until its next dirty-write flush, at which point
      the handle destructor decompresses it, and it isn't re-evaluated
      for compression again until another promotion-event checkpoint.
      Practical implication: pick `switch_point` well above
      `BLOCK4_PROMOTE_MIN_LIVE`, and expect the sparse fraction to
      reflect each tile's *steady-state* trained density, not its
      density at the moment of promotion.

## Part C: batch=1 real-time speed (sili's actual target workload)

Motivating question, per direction: sili's purpose is real-time online
learning, where batch is almost always 1, not a training-sized batch.
"Load an entire network into block4, then prune/grow while gathering
importance data" only makes sense if block4 ops are close to a normal
dense matmul's speed at batch=1 -- the earlier batch=64 numbers (Part B's
own benchmark run) don't answer that question at all.

- [x] **Direct block4-vs-scattered-CSR comparison at batch=1**, bypassing
      growth-driven promotion entirely (which only touches a small,
      probe-budget-limited fraction of an already-populated matrix per
      run -- 50% uniform density only reached 8.33% block4 fill after
      200 growth cycles, nothing like "load the whole network"). Built a
      native harness that bulk-loads an entire dense weight matrix
      directly into block4 tiles (mirroring the planned workflow) vs the
      identical weights as pure scattered CSR, at densities 10%-100%.
      **Finding: at batch=1, block4 was slower than scattered CSR at
      EVERY density including 100% fill**, with or without threading
      (0.66x forward / 0.43x backward at 100% density, single-threaded).
      Batch=64 numbers (0.9x-3.9x depending on density) are irrelevant
      to this workload -- block4's fixed per-tile overhead needs batch
      size to amortize against, and real-time inference doesn't have
      that.
- [x] **Root cause #1, found and fixed: redundant per-tile lookup.**
      `disldo_forward`/`disldo_backward`'s block4 loop already walks
      each row's block4 cursor once (single-threaded, sequential) to
      discover which tiles exist, building a flat coordinate list for
      the parallel compute loop -- then threw that position away and
      called `find(br, bc)` per tile in the parallel loop, which
      re-derives the same tile's position from scratch via a SECOND
      O(row_nnz) cursor scan. At batch=1 a 4x4 tile is only 16 FLOPs,
      nowhere near enough to amortize even one such scan, let alone two.
      Fixed via `Block4Store::at_index()` / a `Block4TileHandle`
      fast-path constructor that takes the already-known storage index
      directly (the collection loop already has it for free --
      `block_layout.elem_start[row]+k`), skipping `find()`'s internal
      `raw_find()` re-scan entirely. Safe even when a tile's own
      destructor re-packs it sparse<->dense mid-loop, since
      `Block4StoredTile` is fixed-size -- repacking never moves any
      tile's slot, so other iterations' captured indices stay valid.
      **Result at batch=1, 100% density: forward 0.66x -> 1.71x** (a
      real win, was a loss at every density before) **, backward 0.43x
      -> 0.72x** (roughly halves the absolute per-tile overhead, second
      cost identified below).
- [x] **Investigated and ruled out via profiling, not assumed:** two
      further hypotheses for backward's remaining gap, both directly
      measured with callgrind/cachegrind on a native repeated-call
      harness (not part of the repo) rather than guessed:
      - *Vector allocation churn*: `tile_br`/`tile_bc`/`tile_elem`/
        `row_live_count`/`t_row_grad` were freshly heap-allocated every
        call. Added persistent scratch buffers on `Block4Store`
        (`scratch_tile_br`/`_bc`/`_elem`/`_row_live_count`/`_row_grad`),
        reused (resized in place) across calls; also folded
        `row_live_count`'s precompute into the SAME collection loop
        instead of a separate second pass over the just-collected list.
        Re-profiling showed this barely moved the exclusive cost
        (11.75% -> 11.95%) -- the real cost wasn't allocation, it was
        `push_back`'s per-call capacity check across ~49k calls (3
        vectors x 16384 tiles at 100% density), independent of whether
        the backing store was fresh or reused. Switched to
        `resize()`+direct indexing (pays that check once per vector, not
        once per element) -- correct and real, but wall-clock impact was
        small.
      - *Cache/memory latency*: cachegrind with `--cache-sim=yes` on the
        same harness showed D1 miss rate 0.1%, LL miss rate ~0% --
        ruled out.
      **Conclusion: backward's remaining gap is genuine per-tile
      arithmetic, not a lookup/allocation/cache bug.** At 100% density
      both paths update the identical total element count (262144 on a
      512x512 layer), and block4 is still ~37% slower PER ELEMENT than
      scattered CSR in that exact case (no wasted work on empty slots to
      blame either). The extra cost is the per-lane FP4 decode
      (`Block4Vec`) and per-column scale composition
      (`combined_scale4`/`combined_imp_scale4`/`col_valid4` etc.) block4
      does for every tile-column, structurally more work per synapse
      than scattered CSR's simpler single-value update.
- [x] **Fair recalibration: "dense matmul" must include FP4 codec cost.**
      A plain float32 dense matmul (no quantization at all) isn't the
      right comparison point -- FP4 encode/decode is a required cost
      regardless of representation (block4 or scattered), not something
      to blame on block4. Measured a dense matmul over FP4-packed
      weights (decode via `FP4_TABLE`, encode via `fp4_quantize`, same
      round-trip real online learning pays) at 512x512/batch=1: forward
      0.222ms, backward 0.639ms. Against THIS fair floor, block4 (0.74ms
      fwd / 1.79ms bwd at the time) was ~3x off, not ~20x -- a real,
      closeable gap, not an unreasonable target.
- [x] **Isolated the gap's real source: removable indirection, confirmed
      by a direct A/B.** A hand-written scalar loop operating directly
      on `Block4StoredTile::data[16]` bytes (bypassing
      `Block4TileHandle`, SIMD `Block4Vec`, and per-lane bookkeeping
      arrays entirely) reached forward=0.218ms on the SAME weight data
      -- essentially identical to the dense-FP4 floor (0.222ms). This
      confirmed the gap was genuinely closeable indirection, not
      something fundamental to block4's design.
- [x] **Precisely isolated WHICH indirection, via a proper controlled
      comparison -- and corrected a real methodological error along the
      way.** First attempt crossed {`Block4TileHandle` vs direct byte
      access} x {SIMD `block4_vec_decode_fp4` vs `FP4_TABLE[code]`
      scalar} and concluded "SIMD loses" -- WRONG framing, caught after
      swapping the fix into the real code and finding forward regressed
      in the full benchmark despite winning the isolated test (1.71x ->
      ~1.55x at 100% density, reproduced consistently). Root cause of
      the wrong conclusion: `FP4_TABLE[code]` (array lookup) and
      `block4_vec_decode_fp4` (bit-shift formula, vectorized) are TWO
      DIFFERENT ALGORITHMS, not a scalar/SIMD pair -- comparing them
      conflated "which algorithm" with "vectorized or not" into one
      axis. Redone properly with `fp4_decode_bits()` (the actual scalar
      equivalent of `block4_vec_decode_fp4`'s formula) as the true
      scalar baseline: an isolated microbenchmark showed SIMD winning by
      5.3x over true scalar bit-shift decode -- confirming SIMD
      genuinely beats scalar bit-shift, consistent with this file's
      earlier documented finding (the `SILI_BLOCK4_FORCE_SCALAR_BACKWARD`
      toggle, which covers a DIFFERENT section -- the batch-loop
      gradient math + stochastic re-encode, not this decode step --
      re-verified still winning at batch=1 too, ~7-8% here, in the
      2-repeat range of the originally-documented 1.05x-1.27x). Then
      tested all three decode options in the REAL, full
      `disldo_backward` benchmark (not an isolated microbenchmark, which
      the forward regression already proved can mislead): scalar
      `fp4_decode_bits` was clearly WORST (bwd 2.71ms at 100% density);
      SIMD `block4_vec_decode_fp4` (original) and `FP4_TABLE[code]` were
      close, with `FP4_TABLE` a real, reproducible ~6% faster (bwd
      1.649ms vs 1.755ms, 3 repeats each, clean non-overlapping ranges).
      **Net, corrected conclusion: kept `FP4_TABLE[code]` for backward's
      decode specifically (real ~6% win, a different-algorithm effect,
      not a SIMD-vs-scalar one) and reverted forward back to
      `block4_vec_decode_fp4` (SIMD genuinely wins there, matching the
      established pattern) -- the two functions' surrounding code
      apparently interacts with this choice differently enough that the
      same swap doesn't transfer between them.** This does NOT
      contradict the earlier "SIMD beats scalar" finding anywhere in
      this file or the sparse-tile pack/unpack investigation -- both
      remain true; what's corrected is that `FP4_TABLE` was never a
      valid stand-in for "scalar" in the first comparison.
- [x] Result after all Part C fixes, batch=1, 100% density, single
      thread: forward 1.71x (unchanged from the `at_index()` fix alone),
      backward 0.72x -> ~0.79x-0.80x (the `FP4_TABLE` decode swap's real,
      if modest, contribution on top of `at_index()`/scratch-buffer
      reuse). Real progress, not yet at the dense-FP4 floor for
      backward -- see below.
- [ ] **Not yet done**: reduce block4 backward's remaining per-element
      arithmetic cost further (block4 is still measurably slower than
      the dense-FP4 floor, ~0.639ms, though much closer than the
      original ~20x-off float32-only comparison suggested). Candidates:
      whether any of the per-lane scale composition can be hoisted or
      shared across tiles in the same block-row (value_scale is already
      per-ROW, so it's currently recomputed identically for every
      tile-column sharing that row -- worth checking if that's real,
      avoidable redundancy or already amortized); whether the
      gradient-math/encode SIMD section (the
      `SILI_BLOCK4_FORCE_SCALAR_BACKWARD`-guarded code) has any further
      batch=1-specific headroom despite already winning there. Any
      further change here MUST be verified the same way this section
      was -- full-benchmark A/B, not an isolated microbenchmark alone,
      since the forward-regression episode above is direct, reproduced
      proof that isolated microbenchmarks can point the wrong way for
      this codebase's actual hot paths.
- [x] **Re-ran the batch=1 density sweep -- and caught a second real
      methodology error, corrected before it misled anything further.**
      The FIRST density sweep (`bench_full_block4.cpp`, all six
      densities looped sequentially inside one process/one `main()`)
      showed forward staying nearly FLAT (~0.69ms-0.76ms) across the
      entire 10%-100% density range, despite a 10x tile-count spread --
      seemingly proving block4 wasn't taking advantage of sparsity at
      all, dominated by some large fixed per-call cost. Investigated via
      callgrind: instruction count DID scale ~10x proportionally with
      tile count (matches expectations for genuine per-tile work);
      cachegrind (`--cache-sim=yes`) showed miss rates similarly tiny at
      both 10% and 100% density (0.4%/0.1% D1, ~0% LL) -- neither
      explained the wall-clock flatness. Re-ran each density point as a
      fully separate, isolated process (`isolated_density_point.cpp`,
      one process per data point, no within-process sequential loop)
      instead: **forward scales cleanly and proportionally with tile
      count after all** (0.072ms at 10% -> 0.717ms at 100%, ~10x for a
      ~10x tile-count change) -- the flatness in the first sweep was a
      real measurement artifact, same class of issue as the
      thermal/frequency-scaling confound `compare_block4_venvs.sh`
      already had to correct for once before in this file (running
      multiple conditions sequentially in one process/run lets earlier
      conditions run at a different effective clock state than later
      ones). **Lesson: any single-process sweep over multiple
      conditions in this codebase needs either isolated processes per
      condition or genuine interleaving across repeats, never a bare
      sequential loop -- a lesson now hit twice (venv comparison,
      density sweep), worth remembering for any FUTURE density/config
      sweep too.**
      Corrected, trustworthy density sweep (isolated process per point,
      batch=1, single thread, vs the dense-FP4 floor: fwd 0.195ms/bwd
      0.556ms):

      | density | fwd (block4) | fwd vs dense | bwd (block4) | bwd vs dense |
      |---|---|---|---|---|
      | 10% (1617 tiles)  | 0.072ms | 0.37x (2.7x FASTER) | 0.164ms | 0.29x (3.4x FASTER) |
      | 25% (4100 tiles)  | 0.179ms | 0.92x (~tied)        | 0.410ms | 0.74x (faster) |
      | 50% (8261 tiles)  | 0.363ms | 1.86x slower         | 0.824ms | 1.48x slower |
      | 75% (12338 tiles) | 0.538ms | 2.76x slower         | 1.237ms | 2.23x slower |
      | 90% (14754 tiles) | 0.639ms | 3.28x slower         | 1.470ms | 2.65x slower |
      | 100% (16384 tiles)| 0.717ms | 3.68x slower         | 1.640ms | 2.95x slower |

      **Real, actionable finding for the planned prune/grow workflow:**
      there's a clear crossover around **~20-25% tile-fill density**.
      Below it, block4 genuinely beats a naive dense-FP4 matmul (up to
      ~3x faster at 10%); above it, block4 loses, worse as density
      climbs toward 100%. The earlier ~3x/~3.7x "gap at 100% density"
      framing above is the WORST case (fully-loaded network, matching
      the specific "load the whole network into block4 first" plan) --
      once pruning brings density down toward the actual crossover,
      block4 should already be winning, not losing.
- [x] **Found and fixed one more real, removable indirection:
      `Block4TileHandle::at()` re-branches on `was_sparse_` every single
      call.** Forward's decode reads all 16 bytes of a tile via 16
      separate `.at()` calls (4 lj iterations x 4 li each); backward's
      decode does the same. `was_sparse_` can't change mid-tile, so
      re-checking it 16 times per tile is pure waste. Added
      `Block4TileHandle::raw_data()` -- resolves the `was_sparse_ ?
      scratch_ : stored_->data` branch ONCE, returns a `const uint8_t*`
      valid for the tile's whole read-only decode phase; writes still go
      through `.at()` (marks dirty, needed for the sparse-repack
      destructor logic). Verified correctness: `Block4Tile::slot_index`
      maps different `li` values to DISJOINT byte positions, so hoisting
      the pointer across the `li` loop can't read stale data even when
      an earlier `li` iteration wrote via `.at()` in the same tile.
      Measured via the NOW-TRUSTED isolated-process methodology (see
      above): forward improved ~9-11% (100% density: 0.713ms ->
      0.634ms; 50%: 0.358ms -> 0.326ms; 10%: 0.071ms -> 0.065ms) --
      real, modest, consistent across the density range, not
      density-dependent. `compare_block4_venvs.sh` re-run afterward,
      consistent with prior runs, no regression.
- [x] **Root cause found via `-fopt-info-vec`: block4's per-tile loop
      structurally cannot auto-vectorize, unlike the dense-FP4 floor's
      simple flat loop.** GCC's own diagnostic for forward's `lj` loop
      (4 output columns per tile): *"loop nest containing two or more
      consecutive inner loops cannot be vectorized"* -- the `li`-decode
      loop and the `b`-batch loop, both nested sequentially inside it,
      block the vectorizer from treating the 4-column dimension as SIMD
      lanes at all. The dense-FP4 baseline's single flat loop vectorizes
      cleanly by contrast. This, not any remaining lookup/allocation/
      branch cost, is the structural explanation for most of the
      residual gap above the crossover density.
- [x] **Fixed for forward**: replaced the runtime `lj` loop with a
      templated lambda (`process_col<LJ>`, C++20 template-lambda syntax)
      called 4 times with compile-time-constant `LJ`, matching the
      hand-unroll pattern already used for the decode step. Confirmed
      via `-fopt-info-vec`: the inner `li` loop now vectorizes in each
      of the 4 instantiations, where it couldn't before at all. Real,
      consistent ~9-10% wall-clock improvement across the WHOLE density
      range (100%: 0.643ms -> 0.585ms; 50%: 0.327ms -> 0.297ms; 10%:
      0.066ms -> 0.059ms) -- unlike the earlier allocation-churn/
      `size()`-hoist fixes, this one actually moved wall-clock time,
      consistent with genuinely unlocking vectorization rather than
      just reducing instruction count the CPU was already absorbing.
      Committed.
- [x] **Attempted the SAME fix for backward -- real, measured
      regression, reverted.** Backward's analogous outer loop (`li`,
      over 4 ROWS) reports the identical `-fopt-info-vec` diagnostic,
      but its structure is fundamentally different from forward's: the
      actual heavy compute (gradient accumulation, importance damping,
      stochastic re-quantization) is ALREADY hand-vectorized via
      `Block4Vec` (the `SILI_BLOCK4_FORCE_SCALAR_BACKWARD`-guarded
      section, independently re-verified still winning at batch=1 this
      session, see above) -- it was never relying on GCC's
      auto-vectorizer for the expensive part the way forward's `lj` loop
      was. Wrapping the WHOLE `li`-loop body (the entire SIMD gradient
      section, importance damping, stochastic encode -- hundreds of
      lines) into a templated `process_row<LI>` lambda and calling it 4x
      quadruples that body's code size with no new vectorization to show
      for it. Measured via the isolated-process methodology: a real,
      clear, consistent REGRESSION (100% density: 1.660ms -> 2.178ms,
      ~31% SLOWER; 50%: 0.830ms -> 1.091ms, ~31% slower) -- reverted
      immediately (`git checkout`, working tree confirmed clean, matches
      the last good commit). Verified correct under both the default and
      `SILI_BLOCK4_FORCE_SCALAR_BACKWARD=1` builds before reverting
      (both passed test_disldo_block4_backward.cpp), so this was a real
      PERFORMANCE regression caught before commit, not a correctness
      bug -- but a regression regardless, and a useful negative result:
      **the "unroll a small loop into a templated lambda" fix is not a
      universal win -- it only helps when the loop body ISN'T already
      hand-vectorized and is small enough that 4x code duplication
      doesn't matter. Apply it selectively, re-measure every time, never
      assume it transfers from one function to a structurally different
      one** (the SAME lesson, in a new form, as the decode-algorithm
      episode earlier in this file).
- [x] **The "backward gap above the crossover density" mostly wasn't
      real -- found via a corrected, fair comparison.** Two remaining
      methodology bugs in the dense-FP4 floor itself, found by
      questioning whether it was doing the same WORK as block4, not
      just the same FP4 codec: (1) it never included
      value_scale/output_scale composition or per-synapse importance
      tracking/damping, and used deterministic (not stochastic)
      re-quantization; (2) the block4 benchmarks it was compared
      against were accidentally using `learning_rate=0`, which skips
      that same work on block4's side too, but for an unrelated reason
      -- neither side was doing equivalent work. Fixed in
      `scripts/bench_block4_vs_dense_fp4.cpp` (new, permanent,
      committed): both sides now run the exact same feature set (real
      nonzero learning_rate, `damp_by_importance=true`, stochastic
      requantize of both weight and importance nibbles) -- the dense
      floor replicates block4's per-element math on a flat array, no
      tile/CSR indirection. **Result: block4 backward beats the fair
      dense floor at EVERY density, including 100% fill** (100%: dense
      7.68ms vs block4 4.07ms, 1.88x faster; 50%: 3.74x faster; 10%:
      15.85x faster) -- almost certainly because block4's SIMD
      stochastic-quantize (`block4_vec_quantize_stochastic_fp4`, 2
      vector calls/tile) beats 8 scalar `fp4_quantize_stochastic()`
      calls by a wide margin, an advantage the earlier, unfair
      comparison never gave it credit for. **Per direction: this real,
      corrected result is sufficient reason to merge the branch** --
      forward's smaller, genuinely-still-open ~5x gap (real, since
      forward at `learning_rate=0` never exercises the
      importance/stochastic machinery backward wins on) is a real
      follow-up, not a blocker.
- [ ] Forward's remaining gap (real, not a baseline artifact) is still
      open, lower priority given the merge decision above. Candidates
      not yet tried: whether any of the col4/out_scale4-equivalent
      setup could be shared/hoisted; whether a batch=1-specific fast
      path makes sense given forward's SIMD decode is already the
      right choice (established this session) but the surrounding
      per-tile-column code isn't otherwise special-cased for batch=1.

## Memory safety + real variable-length compression (done)

Follow-up to the sections above: neither the scattered CSR side nor
block4 actually enforced `max_weights` during synaptogenesis, and
block4's "compression" (`is_sparse` flag) never shrank real memory --
`Block4StoredTile` was a fixed 16(+1) bytes regardless of occupancy, so
a compressed tile was repacked bytes *within the same slot*, not a
smaller allocation.

- **Scattered CSR cap enforcement.** `set_limits()` existed but was
  never called anywhere; wired into `SparseLinearLayer`/`DISLDOLayerV`
  constructors and re-applied after `compact()`/`expand_headroom()`/
  `expand_headroom_to()`/`load_weights()` (each rebuilds `connections`
  via `delta_csr_from_absolute`, which defaults to unbounded). The real
  dominant bug was `delta_csr_shift_row()` -- called every
  synaptogenesis cycle via `equalizer_step()` -- growing `ibuf`/`values`
  via raw `.resize()` with zero budget check, confirmed via a stress
  test reaching 127x `max_weights` with no resistance. Fixed with real
  `throw std::bad_alloc()` checks before every resize.
- **block4 cap enforcement.** Same class of fix
  (`block4_row_shift`/`grow_last_row`/`ensure_row_headroom` all gained
  real budget checks) plus a budget-formula correction: sizing by
  `max_weights / BLOCK4_PROMOTE_MIN_LIVE` (2) assumed tiles stay
  minimally filled, but tiles trend toward FULL occupancy under real
  training (every `disldo_backward` touch writes all 16 slots) --
  `max_weights / BLOCK4_TILE_SLOTS` (16) is the correct, empirically
  verified divisor.
- **Real variable-length tile storage.** `Block4StoredTile` (fixed
  16+1 bytes) replaced with a flat `tile_data: vector<uint8_t>` byte
  buffer + per-row `tile_byte_start`/`tile_byte_end` (mirrors
  `DeltaCSRLayout`'s own byte/elem split, but for tile bytes -- a
  second, independently-growable row-byte layout alongside the
  existing one for `indices_buf`) + `tile_is_sparse: vector<uint8_t>`
  (explicit per-tile flag, since a sparse tile at
  `BLOCK4_SPARSE_MAX_COUNT` is byte-length-identical to dense).
  `block4_sparse_pack`/`unpack`/`get_pos`/`set_pos` now take pointers,
  not fixed `uint8_t[16]` arrays, and the packed length scales with the
  tile's ACTUAL live count (`1 + ceil(count/2) + count` bytes, e.g. 2
  live synapses = 4 bytes = 32 bits, matching the "24 bits for 2
  params" estimate from the original ask), not a fixed per-store size
  tied to `switch_point` -- real compression at the DEFAULT
  `switch_point` (10) already, not just when tuned down. Verified with
  an exact-byte-count unit test
  (`test_real_compression_shrinks_footprint`,
  `tests/unit/test_block4_sparse_tile.cpp`).
- **A real concurrency hazard this redesign introduced, found and
  fixed before it shipped.** `disldo_backward`'s parallel per-tile loop
  used to be safe under the OLD fixed-16-byte design specifically
  *because* no tile's resize could ever move any OTHER tile's storage.
  Under real variable-length storage, resizing one tile (sparse<->dense
  transition) memmoves every LATER tile in its row -- two threads
  processing different tiles in the same block-row could race on that
  row's `tile_data`. Fixed via `Block4Store::force_dense_at()`: backward's
  existing SERIAL collection pass (which already walks every tile it's
  about to touch, in order) now promotes any still-sparse tile to dense
  THERE, before the parallel region starts -- justified by the same
  full-occupancy finding above (backward touches all 16 slots on every
  call, so a touched tile ends up dense immediately regardless).
  Verified with a dedicated ThreadSanitizer stress harness driving real
  `disldo_forward`/`disldo_backward` at `num_cpus=8` with `switch_point=2`
  (forces heavy sparse<->dense churn every call): all flagged races
  involved "main thread" stack reuse across sequential,
  barrier-separated `omp parallel` regions (a known GCC-libgomp+TSan
  instrumentation gap -- the identical pattern reproduces between two
  calls of the UNMODIFIED `disldo_forward` alone, with zero block4
  involvement), and zero races were found between two genuinely
  concurrent worker threads.
- **A real, measured speed regression, found and fixed before merge.**
  The initial version of `force_dense` re-derived a tile's
  `elem_pos`/`byte_pos` via its own `raw_find()` row scan every call --
  called once per tile from a collection loop that was ALREADY walking
  the row in order, turning that loop into an O(row_nnz²) scan. Caught
  via `scripts/bench_block4_vs_dense_fp4.cpp`: backward's speedup over
  the dense floor dropped from the documented ~1.88x to ~1.44x at 100%
  fill. Fixed by splitting out `force_dense_at(br, elem_pos, byte_pos)`,
  an O(1) core the collection loop calls directly with values it
  already has. Post-fix, speed matches or slightly beats every
  previously-documented density point (100% fill: 1.93x vs documented
  1.88x; 50%: 3.83x vs 3.74x; 10%: 16.11x vs 15.85x; forward's ~5x
  gap, `learning_rate=0`'s real cost, unchanged).
- **Not yet done**: the dedicated memory-cap + compression benchmark
  (permanent, committed, verifying max memory is never exceeded during
  a real synaptogenesis stress run AND that compression genuinely
  reduces `total_tile_alloc_bytes()`/`total_tile_used_bytes()`) --
  planned as a follow-up, not blocking this redesign.
- **Not yet done**: routing sparse CSR input through sisldo ops (dense
  input already goes through disldo) and finding/documenting the
  density crossover point where sparse input becomes faster.

## Explicitly NOT changing

- The prototype branch (`feature/sili-ell-benchmark`, PR #22) and its
  code stay as-is, kept for reference/testing, not merged, not deleted.
- `sili_ell`'s own banked/packed codecs are not being integrated --
  this work is specifically about block4 + the existing disldo/CSR path.
