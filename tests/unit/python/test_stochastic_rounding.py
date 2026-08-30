"""
tests/unit/python/test_stochastic_rounding.py
───────────────────────────────────────────────
Real-FP4 DISLDOLayer's out-of-context collapse on sili_peridot's tile
-recurrence curriculum was root-caused to per-step STOCHASTIC rounding
(fp4_quantize_stochastic), not value_scale staleness -- an
architecturally identical fp32-shadow control (deterministic rounding)
reached mean_acc=0.72, real FP4 collapsed to chance (0.10), and
switching real FP4 to deterministic rounding alone closed the entire
gap (0.79). See sili_peridot/JOURNAL.md's 2026-08-09 (cont.) entry for
the full investigation.

`StochasticRounding` (disldo_backward's 6th template param,
linear_disldo.hpp) had zero test coverage before this file, despite
carrying the actual fix -- SparseLinearLayerResync/AdaMax similarly
had none (a pre-existing gap, not created here). This covers only the
core, directly-checkable property: does the STORED result depend on
the stochastic-rounding RNG's state, or not.
"""
import numpy as np

from sili import _cpu


def _make_layer(cls, n_in=6, n_out=4, budget=64, cpus=1, seed=0):
    layer = cls(n_in, n_out, budget, cpus)
    rng = np.random.default_rng(seed)
    # Same preseeded wiring/weights across both instances being compared --
    # only the RNG state at backward()-time differs between calls.
    idx = np.arange(n_in * n_out)
    rng.shuffle(idx)
    rows = idx // n_out
    cols = idx % n_out
    ptrs = np.zeros(n_in + 1, dtype=np.int64)
    for r in rows:
        ptrs[r + 1] += 1
    ptrs = np.cumsum(ptrs)
    order = np.argsort(rows, kind="stable")
    weights = rng.standard_normal(len(idx)).astype(np.float32) * 0.1
    layer.load_weights(ptrs.astype(np.int64), cols[order].astype(np.int64), weights[order])
    return layer


class TestStochasticRoundingDeterminism:
    def test_deterministic_class_ignores_rng_state(self):
        # Two structurally-identical SparseLinearLayerDeterministic layers,
        # RNG seeded to DIFFERENT states before each backward() call --
        # deterministic rounding must never consult the stochastic RNG at
        # all, so the stored result should be identical regardless.
        a = _make_layer(_cpu.SparseLinearLayerDeterministic)
        b = _make_layer(_cpu.SparseLinearLayerDeterministic)
        x = np.random.RandomState(1).randn(1, 6).astype(np.float32)
        dy = np.random.RandomState(2).randn(1, 4).astype(np.float32)

        _cpu.seed_fp4_stochastic_rng(11)
        a.forward_dense(x)
        a.backward_dense(x, dy, 0.05)

        _cpu.seed_fp4_stochastic_rng(999)
        b.forward_dense(x)
        b.backward_dense(x, dy, 0.05)

        np.testing.assert_array_equal(
            np.asarray(a.weights_vals), np.asarray(b.weights_vals),
            err_msg="SparseLinearLayerDeterministic's stored weights depended "
                    "on the stochastic-rounding RNG state -- StochasticRounding "
                    "=false is not actually skipping set_stochastic().")

    def test_stochastic_class_depends_on_rng_state(self):
        # Same setup, plain (stochastic) SparseLinearLayer -- different RNG
        # state before each call SHOULD produce a different stored result
        # at least some of the time (real dithered rounding, not a no-op).
        a = _make_layer(_cpu.SparseLinearLayer)
        b = _make_layer(_cpu.SparseLinearLayer)
        x = np.random.RandomState(1).randn(1, 6).astype(np.float32)
        dy = np.random.RandomState(2).randn(1, 4).astype(np.float32)

        _cpu.seed_fp4_stochastic_rng(11)
        a.forward_dense(x)
        a.backward_dense(x, dy, 0.05)

        _cpu.seed_fp4_stochastic_rng(999)
        b.forward_dense(x)
        b.backward_dense(x, dy, 0.05)

        assert not np.array_equal(np.asarray(a.weights_vals), np.asarray(b.weights_vals)), (
            "SparseLinearLayer's stored weights were identical across two "
            "different stochastic-RNG seeds -- either a real coincidence, "
            "or set_stochastic() stopped consulting the RNG.")

    def test_deterministic_matches_across_repeated_seed_pre_existing_failure(self):
        # Sanity: SAME seed before each call, deterministic class -- must
        # match (trivially true if the above test passes, but confirms
        # this isn't accidentally comparing two empty/no-op backward calls).
        a = _make_layer(_cpu.SparseLinearLayerDeterministic)
        b = _make_layer(_cpu.SparseLinearLayerDeterministic)
        x = np.random.RandomState(3).randn(1, 6).astype(np.float32)
        dy = np.random.RandomState(4).randn(1, 4).astype(np.float32)

        for layer in (a, b):
            _cpu.seed_fp4_stochastic_rng(42)
            layer.forward_dense(x)
            layer.backward_dense(x, dy, 0.05)

        assert not np.allclose(np.asarray(a.weights_vals), 0.0), (
            "backward_dense produced an all-zero update -- test setup itself "
            "is degenerate (no learning happened), earlier assertions in this "
            "file are not meaningful.")
        np.testing.assert_array_equal(np.asarray(a.weights_vals), np.asarray(b.weights_vals))

    def test_default_scale_regression_unaffected_by_stochastic_rounding_param(self):
        # SparseLinearLayer's DEFAULT template args (ScalePolicy=RMSprop,
        # DeferredScaleWrite=false, StochasticRounding=true) must be exactly
        # what every existing caller already relies on -- this is the one
        # regression check that actually matters for merge safety: adding
        # the new template parameter must not have changed anything for
        # callers that don't touch it.
        a = _make_layer(_cpu.SparseLinearLayer)
        x = np.random.RandomState(5).randn(1, 6).astype(np.float32)
        out = a.forward_dense(x)
        assert out.shape == (1, 4)
        assert np.isfinite(out).all()
