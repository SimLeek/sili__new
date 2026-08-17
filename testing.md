# Testing sili__new

Quick reference for verifying changes after a commit. Run these from
the repo root (`sili__new/`) using the project's venv
(`/home/simleek/claude_code/.venv/bin/python` / `pip`).

## 1. Rebuild the C++ extension

Any change under `sili/lib/headers/*.hpp` or `sili/cpu_backend.cpp`
requires a rebuild before Python sees it:

```bash
/home/simleek/claude_code/.venv/bin/pip install -e . --no-build-isolation
```

A clean rebuild takes 1-3 minutes. Watch for `static assertion failed:
The number of argument annotations does not match the number of
function arguments` -- this means a pybind `.def(...)` call's
`py::arg(...)` list doesn't match the actual C++ function signature.
Several classes (`SparseLinearLayer`, `SparseLinearLayerResync`,
`SparseLinearLayerNoScale`, `SparseLinearLayerDeterministic`,
`SparseLinearLayerResyncDeterministic`,
`SparseLinearLayerNoScaleDeterministic`) all instantiate the SAME
`SparseLinearLayerImpl` template -- changing a shared method's
signature (e.g. `build_probes`/`synap_row_step`/`synap_step`) requires
updating the pybind bindings for ALL SIX classes, not just the one you
were testing with.

## 2. Python unit tests (primary regression check)

```bash
/home/simleek/claude_code/.venv/bin/python -m pytest tests/unit/python/ -q
```

**Known pre-existing failures, unrelated to recent work** (confirmed
against `git show HEAD` before touching anything this session -- these
were already broken, not caused by any change here):
- `tests/unit/python/test_sili.py` -- ~64 failures + 23 errors, all
  `TypeError: __init__(): incompatible constructor arguments`. The
  test file's `SparseLinearLayer(...)` constructor calls don't match
  the current pybind signature. Needs updating to the current
  constructor API, or the API needs a compat shim -- your call.
- `tests/unit/python/test_forward_output_not_aliased.py::TestForwardDenseOutputNotAliased::test_previous_result_survives_a_second_call`
  -- calls `forward_dense(x, 0.0)` (2 args); current binding only
  accepts `forward_dense(x)` (1 arg, no learning_rate). Stale test.

Everything else should pass. If a NEW test starts failing after a
change, that's a real regression -- these two are the known baseline
noise to filter out.

## 3. C++ unit tests (Catch2, via CMake)

```bash
cd tests/unit/build
cmake --build . -j4
./sili_tests           # runs everything registered in this binary
./test_disldo_block4_backward   # individual test binaries also exist
```

**Known pre-existing broken (does not even compile), unrelated to
recent work**: `test_disldo_block4_forward.cpp`,
`test_disldo_block4_promotion.cpp`, `test_disldo_synaptogenesis.cpp`
all call `disldo_forward(...)` with 7 arguments (including a
`learning_rate` param); the current (and HEAD-committed) signature in
`linear_disldo.hpp` only takes 6. These test files are stale relative
to a `disldo_forward` signature change that predates this session --
need updating to match the current signature, or the signature needs
an overload. This blocks the WHOLE CMake build (`make` aborts on the
first compile error), so currently `cmake --build .` cannot succeed
until these are fixed.

**Workaround until fixed**: build/run individual test binaries by
target name to skip the broken ones, e.g.:
```bash
cmake --build . -j4 --target test_disldo_block4_backward
./test_disldo_block4_backward
cmake --build . -j4 --target test_fp4_bitshift
./test_fp4_bitshift
```

## 4. Targeted smoke tests for this session's changes

No committed test file exists yet for the empty-CSR + synaptogenesis
growth path (`_preseed_empty`, `empty_init=True`,
`DISLDOLayer.synaptogenesis()`) or the pruning-protection eps
(`delta_csr_synap_row_step`'s `importance_eps`) / `max_prune_per_step`
cap. Ad hoc verification used this session (not committed as files --
recreate as needed, or ask to have them turned into real pytest
tests):

```python
import numpy as np
from sili.sparse_rnn import DISLDOLayerDeterministic
from sili.tensor import Tensor, reduce_sum, power

# Confirms: empty_init starts with 0 connections, growth via
# synaptogenesis() actually adds real synapses, and those synapses'
# WEIGHTS (not just nnz) escape 0 given a real training loss.
rng = np.random.default_rng(42)
layer = DISLDOLayerDeterministic(8, 8, 32, rng=rng, empty_init=True)
assert layer._c.nnz == 0
x = Tensor(rng.standard_normal(8).astype(np.float32))
target = rng.standard_normal(8).astype(np.float32) * 2.0
for step in range(500):
    out = layer.forward(x, 0.05)
    loss = reduce_sum(power(out - Tensor(target), 2))
    loss.backward()
    layer.synaptogenesis(k=3, importance_cutoff=0.0, max_row_weights=layer._max_row_weights)
assert layer._c.nnz > 0
assert np.any(np.asarray(out.data) != 0.0)  # weights actually moved
```

```python
# Confirms max_prune_per_step actually caps removals in one call --
# raise importance_cutoff far above any real importance and check nnz
# only drops by <= the cap, not to 0 in one shot.
layer2._c.build_probes(4)
before = layer2._c.nnz
layer2._c.synap_step(1e6, layer2._max_row_weights, max_prune_per_step=2)
removed = before - layer2._c.nnz
assert removed <= 2
```

## Known findings worth re-verifying after future changes

- **Growth mechanism**: `_preseed_empty` + `synaptogenesis()` genuinely
  grows real synapses from a zero-connection layer (confirmed:
  nnz 0→16, weights escape 0, loss decreases, all in ~0.07s for 500
  steps on an isolated 8x8 layer -- dramatically faster than the older
  `all_zero_init` dense-preload approach).
- **The "lockstep" stall**: a layer whose INPUT has never gone nonzero
  (e.g. `o_proj` downstream of a still-zero `v_proj`) accumulates
  `neuron_input_accum≡0`, giving every candidate probe score
  (`input_accum*grad_accum`) exactly 0 -- a real, structural symmetry
  -breaking problem, not a bug. `use_energy=True` (forced-firing
  independent of upstream weights) breaks it decisively; the
  `importance_eps`/`max_prune_per_step` changes in this session protect
  against premature pruning of low/zero-importance synapses but do NOT
  by themselves fix the "input never goes nonzero" case -- confirmed by
  direct test (`growth-from-tied-zero-accum` stays at nnz=0 even with
  the eps fix, since `build_probes` is intentionally NOT floored --
  see the next point).
- **`importance_eps` is a read-time-only "ghost" value** -- it exists
  ONLY inside `delta_csr_synap_row_step`'s `importance_cutoff`
  comparison. It is never written into FP4 storage anywhere
  (`delta_csr_build_probes`'s probe scores are the real, unfloored
  `input_accum*grad_accum` product, stored verbatim). Do not "fix"
  premature-pruning issues by inflating a stored value to survive FP4
  quantization (FP4's smallest nonzero magnitude is 0.5 -- writing
  that as a fake importance would itself be a large, distorting value
  in this project's typical scales) -- always float the *comparison*,
  never the *storage*.
