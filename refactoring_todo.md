# Refactoring status and plan

High-level "where are we and what's left" for the cpu_sparse_io +
optim_merge/master + sili_old consolidation into `sili_new`. For granular
code-level bugs/features, see `TODO.md` instead — this document is the map,
that one is the item list.

## Goal / definition of done

- `sili_new` contains the union of everything worth keeping from all three
  sources, actually working (tests passing), not just present.
- The three source clone directories (`sili_cpu_sparse_io`, `sili_optim_merge`
  /`sili_main`, `sili_old`) end up empty (or contain only a `.MIGRATED`
  marker per file) — an empty directory is the "done" signal, not something
  tracked separately.
- No file over ~1k lines where reasonably avoidable (ongoing constraint, not
  a one-time pass — see "File sizes" below).

## Priority ordering (established this session, applies throughout)

1. **This session's own SILi work** (built in this chat, before the repo
   consolidation started) — highest priority of all sources. Its C++ core
   (delta-CSR, FP4, ULEB128, synaptogenesis, parallel pointers,
   `importance_scale`) is more complete and more heavily tested than any
   single source repo's own state. Its Python-level tests
   (`test_sili.py`, `multimodal_sparse_rnn.py`, `sparse_tcnn_audio.py`,
   preserved in `test/python/`) are the **highest priority within that** —
   explicit instruction: keep these as-is through the consolidation,
   updated only for the `forward_dense`/`backward_dense` +
   `forward_sparse`/`backward_sparse` rename (see "Immediate next step"
   below — not yet done).
2. **`optim_merge`/master** (the actual GitHub repo) — second priority.
   Confirmed most-recent by both date and git ancestry (`optim_merge` =
   `master` + one commit, 2026-07-01). This is what `sili_new` was branched
   from.
3. **`cpu_sparse_io`** — explicitly the old direction (2026-03-10, predates
   master). Only relevant where it has something genuinely unique that
   neither of the above covers.
4. **`sili_old`** — oldest (Jan 2025), a different repo entirely, GPU/vision-
   shader-focused. Mostly superseded; `to_csr`/`to_coo` are the confirmed
   exception (see below).

Where two sources describe the same concept, higher priority wins by
default — this has been the standing rule all session (e.g. delta-CSR
always wins over absolute CSR for index encoding; the generic
`sparse_struct`/`ValueAccessor` framework was kept over a from-scratch
concrete reimplementation once confirmed to already exist).

## File sizes

Checked this session (`find ... | xargs wc -l`, current as of this commit).
Files over ~1k lines:

- `sili/conversion/rnn_fold.py` (1389) — pre-existing (optim_merge), only
  bug-fixed this session, not restructured. Not touched for size yet.
- `test/python/test_sili.py` (1245) — this session's own test file,
  **explicitly instructed to keep as-is** (content-wise) through the
  consolidation. Splitting it is not obviously compatible with that
  instruction — would need to ask before doing this one specifically.
- `sili/conversion/model_reconstruct.py` (1195) — pre-existing, bug-fixed
  only.
- `sili/lib/headers/coo.hpp` (1145) — pre-existing, not touched at all yet.
- `sili/conversion/sparse_runtime.py` (1005) — pre-existing, right at the
  boundary.

`sparse_struct.hpp` (was the largest file in the repo, 1520 lines, the one
grown most directly this session) has been split into
`delta_csr_types.hpp` (448) / `delta_csr_memory.hpp` (676) /
`delta_csr_ops.hpp` (435), with `sparse_struct.hpp` left as a 21-line
umbrella include for backward compatibility. See TODO.md-adjacent commit
history for the exact split rationale.

The remaining oversized files are all pre-existing content this session
hasn't deeply restructured (as opposed to bug-fixed) — splitting them
would need enough familiarity with their internals to do safely, which
hasn't been built up yet the way it has for the delta-CSR core. Flagging
rather than attempting blind.

## Source directory status

### This session's own SILi work
- C++ core: fully absorbed into `sili_new`'s delta-CSR framework
  (`SparseLinearWeightsDelta`/`ValueAccessor` generic pattern), not
  copied wholesale — capabilities (synaptogenesis, `compact`,
  `expand_headroom`) ported individually onto the real repo's existing
  (more generic) type system rather than importing a second parallel
  implementation. `importance_scale`/parallel pointers not yet ported
  (see TODO.md).
- Python tests: files preserved in `test/python/`, **not yet updated** to
  the current API (see "Immediate next step").

### optim_merge / master
- This *is* `sili_new`'s base (branched from `optim_merge`). "Status" here
  is really "what's been fixed since branching" — see git log. Two
  standalone-registration bugs remain uninvestigated
  (`make_weights`/`make_weights_v`, see TODO.md).
- `unittest_sisldo.cpp` parked (stale pre-refactor signature), 7 test cases
  not yet triaged individually.

### cpu_sparse_io
Full inventory taken this session (`find sili_cpu_sparse_io/sili -type f`).
Categorized, not deeply read line-by-line — confidence noted per category:

- **Already handled**: `outer_product.hpp` (confirmed superseded, deleted,
  `.MIGRATED` marker left). `fiber.hpp`/`old_fiber.hpp` (backburnered,
  tracked in TODO.md).
- **High-confidence superseded, not yet deleted**: `sparse_struct.hpp`,
  `csr.hpp`, `coo.hpp`, `linear_sisldo.hpp`, `linear_sidlso.hpp` — this is
  the old (pre-generic-template, pre-delta-CSR) generation of the same
  concepts `sili_new` already has in more current form. `scan.hpp`/
  `unique_vector.hpp` likely same lineage (`sili_new`'s own `scan.hpp`,
  inherited from optim_merge, does NOT depend on `unique_vector.hpp` at
  all — different, newer implementation approach). Not yet deleted because
  not yet individually confirmed the way `outer_product.hpp` was (grep for
  callers, check dates) — should get the same treatment before removal,
  not just assumed.
- **Not yet evaluated, genuinely unclear**: `quantized_arrays.hpp`,
  `rolling_linear.hpp`, `cache_info.h`/`cache_info.cpp`, `csf.h`/`csf.cpp`.
  No obviously-named counterpart in `sili_new` (unlike the items above),
  but also not read deeply enough this session to say whether they're
  live, useful, unique concepts or further old-generation cruft. Genuinely
  needs a real look, not a guess.
- **Python-side device/runner abstraction** (`core/devices/gpu.py`,
  `core/runners.py`, `core/strategies.py`, `core/module.py`, `core/perf.py`,
  `core/serial.py`, `core/buffers.py`, `buffers/*.py`): this is the Kompute-
  wrapper / GPU device abstraction layer. Directly relevant to the
  already-tracked TODO.md backburner item ("eventually wants its own
  runner system... GPU ops as part of a GPU 'device' abstraction") — a
  source to draw from when that's picked up, not something to port now.
- **`util/energy.py` / `energy_2.py` / `system_energy.py` / `stress_2.py`**:
  checked directly — this is SYSTEM/HARDWARE power and clock-speed
  monitoring (`psutil`, Hz tracking), NOT the neural homeostatic energy
  dynamics system from earlier this session. Different "energy" entirely,
  but plausibly a genuine, non-superseded piece: the real energy dynamics
  function's own docstring says its hard ceiling (`p`) should be informed
  by "GPU/CPU temperature monitors (thermal throttle)" and "battery level"
  — this could be the infrastructure that *feeds* that constraint, not
  something superseded by it. Worth a real look when energy dynamics
  wiring is picked up, not before.
- **Vision/image modules** (`buffers/image.py`, `buffers/pyramid.py`,
  `buffers/conv.py`, `modules/image_pyramid_*`, `modules/pyramid_conv_*`,
  `modules/unready/*`): image-pyramid and conv-pyramid GPU kernels for
  vision processing specifically. Out of scope for the current sparse
  neural network core consolidation — not evaluated further, no plan to
  port unless a vision-specific need comes up later.

### sili_old
- **Confirmed valuable, tracked in TODO.md**: `modules/to_csr/`,
  `modules/to_coo/` (fixed-IO-size GPU shader groups, real PCIe bandwidth
  win). Not yet integrated.
- **Everything else in `modules/`** (`csr_matmul_csr*`, `depth_pyr_conv`,
  `horiz_pyr_conv`, `vert_pyr_conv`, `pyr_FAST*`, `pyr_conv`,
  `pyr_local_max_sparsity_enforce`, `pyr_patch_max_sparsity_enforce`,
  `image_pyramid`, `radacon`, `adacon`, `mse_loss`, `reduction`,
  `multi_matrix_inverse`, `to_spvec`): vision/pyramid-conv GPU kernels,
  same category as cpu_sparse_io's vision modules above — out of scope for
  now, not evaluated further. `to_spvec` specifically has two files
  literally named `*_not_working.comp` — a hint that even within
  `sili_old`'s own history this particular thing wasn't finished/working,
  lower priority than the rest even if vision work is picked up later.
- `sili_old/TODO.md` — historical GPU design notes (workgroup sizing,
  edge-detector separability), cross-referenced from this repo's `TODO.md`,
  not merged in. Different scope (implementation notes for specific old
  shaders, not a live task list).

## Immediate next step (not started)

Update `test/python/test_sili.py` and `test/python/multimodal_sparse_rnn.py`
to use the current API: `forward_dense`/`backward_dense` instead of
whatever they currently call for dense-input forward/backward, and
`forward_sparse`/`backward_sparse` where sparse-input is intended. Per
explicit instruction, this is the highest-priority concrete piece of work
remaining — these tests were called out specifically as things to actively
preserve and adapt, not just leave sitting in `test/python/` unrun.
Haven't yet checked exactly what API surface these files currently assume
(likely this session's original unified `SparseLinearLayer.forward()`/
`.forward_disldo()` naming, which doesn't match either the old repo's
naming or the new `forward_dense`/`forward_sparse` split) — that's the
actual first step, not yet done.
