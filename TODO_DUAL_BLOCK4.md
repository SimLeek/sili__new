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

## Open design questions (need answers before writing C++)

1. **Promotion/demotion threshold, precisely.** User's own framing: "10%
   of 16 is 1.6 so if there are 2 synapses in a 4x4 block then promotion
   starts to make sense, but it depends on which exact local group does
   better." Concretely:
   - The threshold should be an INTEGER synapse count per tile (>=2 for a
     4x4 tile, not a bare fraction), since fractional synapses aren't
     real -- what's the right rounding rule in general (ceil(0.10*16)=2,
     but is 10% still the right fill fraction now that this is a real,
     re-verifiable-per-machine number, not the prototype's one-off
     measurement)?
   - "Depends on which exact local group does better" -- does this mean
     the decision should be based on a per-tile MEASURED or ESTIMATED
     comparison (e.g., track actual disldo vs. block4 cost for that
     region over time) rather than a single fixed global threshold? Or
     does it mean different SUFFIXES/roles (q_proj vs. gate_proj, etc.)
     may have different real breakevens worth separately calibrating?
     Needs to be settled with the user before implementing.
2. **Memory layout.** Today `SparseLinearWeightsDelta` holds one
   `DeltaCSRWeights` (ULEB128-delta CSR + FP4BiPacked values). The dual
   version needs to hold BOTH that (for the scattered remainder) AND a
   block4 structure (`Block4Weights`-equivalent, now as a real member,
   not a separate prototype struct) -- plus keep `compact()`/
   `expand_headroom*()` working correctly across both.
3. **Synaptogenesis hook.** `build_probes`/`synap_step` grow new
   synapses today via the scattered CSR only. New connections need a
   real decision: land in the scattered part by default (current
   behavior) and get PROMOTED to block4 once their tile crosses the
   threshold (checked when? every synap_step call? a separate,
   explicitly-invoked maintenance pass, matching Fable's own controller
   cadence idea?) -- and DEMOTION needs the same kind of hook when
   pruning drops a tile below threshold.
4. **Backward's transpose cost.** The prototype rebuilds block4's
   transpose from scratch every `backward()` call (O(nnz)) -- fine for
   batch training, likely a real bottleneck for genuine online (single-
   token) training at real model scale. Needs either an incremental
   transpose-update (paired with promotion/demotion, since those are the
   only things that change which tiles exist) or a deliberate decision
   that online training doesn't get block4's speed benefit for backward
   yet (forward could still use it).
5. **Tile size**: fixed at 4x4 for this machine (verified: this CPU's
   real SIMD ceiling is ~4x, not the assumed 8x; 2x2 is a measured
   regression, not just a non-improvement -- see PR #22 / BLOCK4_NOTES.md
   for the numbers). Should the production version hardcode 4, or be
   templated/configurable now so a later architecture-detection pass
   (noted as a future idea, not scoped) doesn't require another rewrite?

## Once the design is settled

- [ ] Extend `SparseLinearWeightsDelta`/`SparseLinearLayer` (delta_csr_types.hpp,
      cpu_backend.cpp) with the block4 member + promotion/demotion logic.
- [ ] Update `forward_dense`/`backward_dense` to combine both paths
      internally (callers shouldn't need to know about the split).
- [ ] Update `build_probes`/`synap_step`/`synap_row_step` to handle
      promotion (and pruning paths for demotion).
- [ ] Update `compact()`/`expand_headroom*()` for the combined structure.
- [ ] Real per-row (or finer) FP4 value_scale calibration for block4,
      matching disldo's own (already proven necessary on real weights,
      not optional -- see PR #22's bug #4).
- [ ] Bash comparison script: old venv vs. new venv, same suite, speed +
      memory + quality, on both synthetic and (if available) real
      checkpoint data.
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
