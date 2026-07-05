# Completed refactoring tasks

Items moved here from refactoring_todo.md once both fixed AND confirmed working.

---

## Write this document / write TODO.md
Done (refactoring_todo.md written, TODO.md maintained separately).

---

## importance_scale / rescale_importance infrastructure (item 3, C++ layer)

**Status: DONE**

- `importance_scale` field on `SparseLinearWeightsDelta`, threaded through all kernels
- `hoyer_value()` / `hoyer_importance()` from O(1) running stats
- `value_l1` / `value_l2_sq` / `value_max_abs` and importance equivalents, maintained incrementally
- `max_abs_decay` for approximate live max without O(n) rescan
- Thread-safety: per-thread local accumulation, one aggregate call per thread (not per synapse)
- `recompute_stats()` for exact from-scratch answer
- All exposed on `SparseLinearLayer` for Python-side use
- FP4 quantization bug found and fixed: stats must use the value read back AFTER `set()`, not the pre-quantization float
- `FP4BiPacked` shared-ptr aliasing footgun documented with regression tests

**NOT done (still in refactoring_todo.md):** the adaptive POLICY for when to trigger rescale (Python-side). C++ owns the cheap stats; Python owns the heuristic.

---

## Synaptogenesis: prune + grow with delta-CSR (new work this session)

**Status: DONE**

- `delta_csr_synap_row_step`: early-return on empty probes now only skips when `n_exist == 0 AND probes empty`. Previously returned early whenever probes were empty, blocking the prune-only path for fully-dense rows.
- `expand_headroom_to(min_nnz_per_row)`: new function in `delta_csr_ops.hpp` -- sets total budget to `max(actual_nnz, rows * min_nnz_per_row)`.
- `equalize_to_capacity(target_elems_per_row)`: new method on `SparseLinearLayer` -- grows each row to at least target_elems reserved space. Fixes up `byte_end`/`elem_end` for all subsequent rows after each shift (delta_csr_shift_row only updates _start arrays; stale _end causes `row_nnz` underflow in bulk equalization).
- `SparseLinearLayer.expand_headroom_to` pybind binding.
- `SparseLinearLayer.equalize_to_capacity` pybind binding.
- Per-row `importance_scale = lr / FP4_MAX` set at `from_descriptor` time so activity-correlation importance updates of magnitude ~lr are representable from the first step. (Weight values are only changed by backward_dense() -- importance updates in forward_dense() do NOT change weight values.)
- `FoldedLayer.synaptogenesis()`: full-sweep (all n_inputs rows), staggered `equalizer_step` (1 row per call), `zero_accum()` after.
- `FoldedLayer.nnz_total()`: monitoring helper.
- `SynaptogenesisSchedule`: wrapper with configurable cadence and optional sine-wave `max_row_weights` for grow/shrink testing.
- Sine-wave test passing: nnz varies across grow and shrink phases of the sine wave.

**Open issue (see refactoring_todo.md):** `equalize_to_capacity` grows memory dynamically. The correct design is fixed total budget upfront (sized for peak `max_row_weights`), then staggered `equalizer_step` redistribution only. Needs rework to not call `delta_csr_shift_row` with `use_b > cur_b` -- instead, pick budget at layer construction time and let equalization handle the rest.
