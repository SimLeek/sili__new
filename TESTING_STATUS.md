# Testing status: process + known issues

See `testing.md` for exact commands to run everything below. This
file is the "what's actually broken right now and why" inventory --
read this before trusting a green/red result, and before starting
work to fix any of these.

## Current testing process

1. Rebuild the C++ extension after touching anything under
   `sili/lib/headers/*.hpp` or `sili/cpu_backend.cpp`:
   `pip install -e . --no-build-isolation`.
2. Python unit tests: `pytest tests/unit/python/ -q` -- the primary,
   currently-working regression gate.
3. C++ unit tests (Catch2, via CMake): `cd tests/unit/build && cmake
   --build . -j4` -- **currently cannot complete a full build** (see
   below), so run individual targets by name to skip the broken ones.
4. Training-convergence checks live in the sili_peridot repo, not
   here (`scripts/landmark_checklist.py` and friends) -- see that
   repo's own `testing.md`.

## Known issues -- pre-existing, confirmed unrelated to recent work

Both confirmed by diffing against `git show HEAD` before any of this
session's changes touched anything -- these were already broken, not
newly caused.

### 1. `tests/unit/python/test_sili.py` -- ~64 failures + 23 errors

Every failure is `TypeError: __init__(): incompatible constructor
arguments`. The test file's `SparseLinearLayer(...)` construction
calls (and everything downstream of them: forward/backward, buffer
access, synaptogenesis, serialisation, parallel-pointer tests) don't
match the CURRENT pybind constructor signature. This is drift, not a
regression from a single recent change -- the constructor has clearly
moved on since this test file was last updated, and the test file
itself needs updating to the current API (or the API needs a
backward-compat constructor overload, if that's preferred).

Not investigated further this session -- there wasn't time to trace
exactly which constructor-signature change orphaned this file, only
to confirm it's real drift and not something touched recently.

### 2. `tests/unit/python/test_forward_output_not_aliased.py` -- 1 failure

`TestForwardDenseOutputNotAliased::test_previous_result_survives_a_second_call`
calls `forward_dense(x, 0.0)` (2 positional args); the current binding
only accepts `forward_dense(x)` (1 arg, no `learning_rate`). Same
drift pattern as above, isolated to one test.

### 3. CMake C++ test suite -- 3 files fail to even COMPILE, blocking the whole build

`test_disldo_block4_forward.cpp`, `test_disldo_block4_promotion.cpp`,
`test_disldo_synaptogenesis.cpp` all call `disldo_forward(...)` with 7
arguments (including a `learning_rate` parameter). The current (and
HEAD-committed, i.e. not a recent regression) signature in
`linear_disldo.hpp` only takes 6 -- there is no `learning_rate` param
on `disldo_forward` (that's `disldo_backward`'s job; `disldo_forward`
is read-only). Since `cmake --build .` aborts on the first compile
error, this currently blocks building the WHOLE `sili_tests` binary,
not just these three files -- every other C++ test in the suite is
untestable via the normal build command until these three are fixed
(either update the calls to drop the stray argument, or confirm
there's a real missing 7-arg overload that should exist instead).

**Workaround**: build/run individual test targets by name to skip the
three broken ones -- see `testing.md`'s section 3 for the exact
commands.

## Known issue -- newly found this session, NOT yet root-caused

Large-scale training-convergence checks (multi-seed, many-step,
accuracy-only) can silently pass while the model is producing
degenerate, constant output. Confirmed directly: a zero-init +
energy_rl config reported eval_acc=0.08-0.13 across 5 seeds looking
like modest real learning, but a direct check showed the model
predicting the SAME constant token for all 100 held-out samples --
the accuracy number was purely chance-matching that token's
~8% base rate in the (skewed) target distribution, not learning.
Root cause: `q_proj`/`k_proj`/`v_proj`/`o_proj` all train correctly in
this config (weights move by orders of magnitude), but `lm_head`
stays essentially at its zero-init value even after 15000 steps and
even with energy applied uniformly to its own output (this session's
`_ENERGY_TAPS` fix) -- not yet diagnosed why `lm_head` specifically
resists escaping zero when its own gradient inputs (real nonzero
`pooled` input, real nonzero cross-entropy gradient) both look fine on
paper. Deferred per current project priority (inference-speed
conversion work takes precedence over further zero-init training
research right now) -- flagging here so it isn't silently forgotten
and so nobody re-trusts an accuracy-only check on this code path
without also checking prediction diversity + per-layer weight escape
directly (see sili_peridot's `testing.md`, "General diagnostic
pattern" section, for the exact probe).
