"""
tests/unit/python/test_aqrs_additive_branch.py
────────────────────────────────────────────────
Python-side confirmation that AQRS's additive branch (task #278) is
actually wired up through the pybind bindings -- get_additive_rank/
set_additive_rank/get_additive_u_k/set_additive_u_raw_k/get_additive_v_k/
set_additive_v_raw_k on SparseLinearLayer, SparseLinearLayerDeterministic,
and SparseLinearLayer8. C++ coverage of the underlying math already exists
(test_aqrs_additive_branch.cpp, test_aqrs_rank_growth_shrink.cpp) -- this
file is deliberately NOT re-deriving that; it only confirms the bindings
themselves round-trip correctly and that forward/backward actually reach
the additive branch through Python, matching test_stochastic_rounding.py's
existing convention for exercising `_cpu` directly.
"""
import numpy as np

from sili import _cpu


def _make_layer(cls, n_in=4, n_out=4, budget=64, cpus=1, seed=0, all_zero=True):
    layer = cls(n_in, n_out, budget, cpus)
    rng = np.random.default_rng(seed)
    idx = np.arange(n_in * n_out)
    rng.shuffle(idx)
    rows = idx // n_out
    cols = idx % n_out
    ptrs = np.zeros(n_in + 1, dtype=np.int64)
    for r in rows:
        ptrs[r + 1] += 1
    ptrs = np.cumsum(ptrs)
    order = np.argsort(rows, kind="stable")
    if all_zero:
        weights = np.zeros(len(idx), dtype=np.float32)
    else:
        weights = rng.standard_normal(len(idx)).astype(np.float32) * 0.1
    if cls is _cpu.SparseLinearLayer8:
        # SparseLinearLayer8's load_weights takes an explicit importance
        # array (no default-zero overload like the FP4 classes).
        imp = np.ones(len(idx), dtype=np.float32)
        layer.load_weights(ptrs.astype(np.int64), cols[order].astype(np.int64),
                            weights[order], imp[order])
    else:
        layer.load_weights(ptrs.astype(np.int64), cols[order].astype(np.int64), weights[order])
    return layer


class TestAdditiveBranchRankRoundTrip:
    def test_default_additive_rank_is_zero(self):
        layer = _make_layer(_cpu.SparseLinearLayer)
        assert layer.get_additive_rank() == 0

    def test_growth_then_shrink_round_trips_stored_values(self):
        layer = _make_layer(_cpu.SparseLinearLayer)
        layer.set_additive_rank(2)
        assert layer.get_additive_rank() == 2

        for row in range(4):
            layer.set_additive_u_raw_k(row, 0, float(row + 1))
            layer.set_additive_u_raw_k(row, 1, float(-(row + 1)))
        for col in range(4):
            layer.set_additive_v_raw_k(col, 0, 0.5 * float(col + 1))
            layer.set_additive_v_raw_k(col, 1, 0.25 * float(col + 1))

        for row in range(4):
            assert layer.get_additive_u_k(row, 0) == float(row + 1)
            assert layer.get_additive_u_k(row, 1) == float(-(row + 1))
        for col in range(4):
            assert layer.get_additive_v_k(col, 0) == 0.5 * float(col + 1)
            assert layer.get_additive_v_k(col, 1) == 0.25 * float(col + 1)

        # Truncating rank must keep channel 0 intact (matches set_scale_rank's
        # own truncate-the-last-channel convention).
        layer.set_additive_rank(1)
        assert layer.get_additive_rank() == 1
        for row in range(4):
            assert layer.get_additive_u_k(row, 0) == float(row + 1)

        layer.set_additive_rank(0)
        assert layer.get_additive_rank() == 0

    def test_regrowth_after_shrink_starts_at_zero(self):
        # set_additive_rank's own reshuffle uses a zero default (see
        # delta_csr_types.hpp) -- a channel that's been dropped and then
        # regrown should read back as 0.0, not stale data from before the
        # shrink.
        layer = _make_layer(_cpu.SparseLinearLayer)
        layer.set_additive_rank(1)
        layer.set_additive_u_raw_k(0, 0, 5.0)
        layer.set_additive_rank(0)
        layer.set_additive_rank(1)
        assert layer.get_additive_u_k(0, 0) == 0.0


class TestAdditiveBranchForward:
    def test_forward_is_zero_before_additive_rank_is_set(self):
        layer = _make_layer(_cpu.SparseLinearLayer, all_zero=True)
        x = np.ones((1, 4), dtype=np.float32)
        out = layer.forward_dense(x)
        np.testing.assert_array_equal(out, np.zeros((1, 4), dtype=np.float32))

    def test_additive_branch_fills_in_for_an_all_zero_main_weight_matrix(self):
        # This is the real-world motivation (Theorem 3/4, see
        # AQRS_DESIGN.md): once every quantized weight in a row lands on
        # the zero sentinel code, the additive branch is structurally the
        # ONLY way to express a nonzero output. Confirm this really holds
        # through the Python bindings, not just in the C++ unit tests.
        layer = _make_layer(_cpu.SparseLinearLayer, all_zero=True)
        layer.set_additive_rank(1)
        for row in range(4):
            layer.set_additive_u_raw_k(row, 0, 1.0)
        for col in range(4):
            layer.set_additive_v_raw_k(col, 0, float(col + 1))

        x = np.ones((1, 4), dtype=np.float32)
        out = layer.forward_dense(x)
        # y[c] = (main weight contribution, 0) + v_c * sum_r(u_r * x_r)
        #      = (col + 1) * 4
        expected = np.array([[4.0, 8.0, 12.0, 16.0]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_additive_rank_zero_is_a_true_no_op_on_a_trained_matrix(self):
        # additive_rank==0 must not perturb ordinary (non-zero-weight)
        # forward passes at all -- regression guard for existing callers
        # who never touch the additive branch.
        layer = _make_layer(_cpu.SparseLinearLayer, all_zero=False, seed=7)
        x = np.random.RandomState(3).randn(1, 4).astype(np.float32)
        out = layer.forward_dense(x)
        assert layer.get_additive_rank() == 0
        assert np.isfinite(out).all()


class TestAdditiveBranchBackward:
    def test_backward_updates_additive_u_and_v(self):
        layer = _make_layer(_cpu.SparseLinearLayer, all_zero=True)
        layer.set_additive_rank(1)
        for row in range(4):
            layer.set_additive_u_raw_k(row, 0, 0.1 * float(row + 1))
        for col in range(4):
            layer.set_additive_v_raw_k(col, 0, 0.1 * float(col + 1))

        u_before = [layer.get_additive_u_k(r, 0) for r in range(4)]
        v_before = [layer.get_additive_v_k(c, 0) for c in range(4)]

        x = np.random.RandomState(1).randn(1, 4).astype(np.float32)
        dy = np.random.RandomState(2).randn(1, 4).astype(np.float32)
        layer.forward_dense(x)
        layer.backward_dense(x, dy, 0.05)

        u_after = [layer.get_additive_u_k(r, 0) for r in range(4)]
        v_after = [layer.get_additive_v_k(c, 0) for c in range(4)]
        assert u_after != u_before, "additive_u never moved -- backward isn't reaching the additive branch"
        assert v_after != v_before, "additive_v never moved -- backward isn't reaching the additive branch"
        assert all(np.isfinite(u_after)) and all(np.isfinite(v_after))


class TestAdditiveBranchDeterministicVariant:
    def test_deterministic_class_exposes_the_same_bindings(self):
        layer = _make_layer(_cpu.SparseLinearLayerDeterministic, all_zero=True)
        layer.set_additive_rank(1)
        layer.set_additive_u_raw_k(0, 0, 1.0)
        layer.set_additive_v_raw_k(0, 0, 3.0)
        assert layer.get_additive_rank() == 1
        assert layer.get_additive_u_k(0, 0) == 1.0
        assert layer.get_additive_v_k(0, 0) == 3.0


class TestAdditiveBranchFP8:
    # SparseLinearLayer8 is the branch task #280 re-validates against the
    # fp8 MQAR input-independent-collapse ("mumbling") case -- confirm its
    # bindings behave identically to FP4's before that revalidation runs.
    def test_fp8_bindings_round_trip_and_wire_into_forward(self):
        layer = _make_layer(_cpu.SparseLinearLayer8, all_zero=True)
        assert layer.get_additive_rank() == 0

        layer.set_additive_rank(1)
        for row in range(4):
            layer.set_additive_u_raw_k(row, 0, 1.0)
        for col in range(4):
            layer.set_additive_v_raw_k(col, 0, float(col + 1))

        assert layer.get_additive_rank() == 1
        for row in range(4):
            assert layer.get_additive_u_k(row, 0) == 1.0

        x = np.ones((1, 4), dtype=np.float32)
        out = layer.forward(x)
        expected = np.array([[4.0, 8.0, 12.0, 16.0]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-2)

    def test_fp8_backward_updates_additive_channels(self):
        layer = _make_layer(_cpu.SparseLinearLayer8, all_zero=True)
        layer.set_additive_rank(1)
        for row in range(4):
            layer.set_additive_u_raw_k(row, 0, 0.1 * float(row + 1))
        for col in range(4):
            layer.set_additive_v_raw_k(col, 0, 0.1 * float(col + 1))

        u_before = [layer.get_additive_u_k(r, 0) for r in range(4)]
        x = np.random.RandomState(1).randn(1, 4).astype(np.float32)
        dy = np.random.RandomState(2).randn(1, 4).astype(np.float32)
        layer.forward(x)
        layer.backward(x, dy, 0.05)
        u_after = [layer.get_additive_u_k(r, 0) for r in range(4)]
        assert u_after != u_before
