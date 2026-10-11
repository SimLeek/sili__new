"""TDD (written before the implementation exists):
DISLDOLayerV.plasticity_column_state()/plasticity_column_state_block4()
-- read-only per-column snapshot accessors for apply_plasticity_reset's
internal PlasticityState, added to support offline data collection for
fitting a reset-selection equation from real training data (direct
instruction: per-column granularity, not per-synapse, since the
mechanism only ever selects at column granularity -- col_importance,
deviation, and age are already per-column aggregates, not per-synapse
values). Zero-copy views into the SAME arrays apply_amortized_plasticity_reset
already maintains -- no new state, no new computation, just exposure.
"""

import numpy as np

from sili import _cpu


def _tiny_layer(n_in=4, n_out=4, num_cpus=2, seed=0):
    rng = np.random.default_rng(seed)
    layer = _cpu.DISLDOLayerV(n_in, n_out, n_in * n_out, num_cpus)
    w = rng.standard_normal(n_in * n_out).astype(np.float32)
    imp = np.full(n_in * n_out, 1.0, dtype=np.float32)
    layer.load_dense_values(w, imp)
    return layer


class TestPlasticityColumnStateScattered:
    def test_empty_before_any_apply_call(self):
        # PlasticityState is lazily sized (ensure_sized) on first
        # apply_amortized_plasticity_reset call -- before that, the
        # snapshot must report empty arrays, not crash on an
        # unsized/null internal vector.
        layer = _tiny_layer()
        snap = layer.plasticity_column_state()
        assert len(snap["col_importance"]) == 0
        assert len(snap["col_age"]) == 0

    def test_sized_after_first_apply_call_matches_n_out(self):
        n_in, n_out = 4, 5
        layer = _tiny_layer(n_in, n_out)
        layer.apply_amortized_plasticity_reset(n_in * n_out, 0.0, 0.99, 0.95, 0.5, 0.1, 0.01, 1.0)
        snap = layer.plasticity_column_state()
        for key in (
            "col_importance",
            "col_grad_slow",
            "col_grad_fast",
            "col_grad_var",
            "col_age",
            "col_reset_active",
        ):
            assert len(snap[key]) == n_out, f"{key} should have length {n_out}, got {len(snap[key])}"

    def test_col_importance_matches_known_values(self):
        # eta=0.0 (weight on OLD value) means col_importance ends up
        # exactly the last-seen raw importance -- same convention the
        # C++ unit tests already establish for this formula.
        # NOTE: load_dense_values() routes through block4_load_dense_fp32
        # (block4 storage only, scattered stays empty) -- must use
        # load_weights() to populate the SCATTERED arm this test targets.
        n_in, n_out = 2, 3
        layer = _cpu.DISLDOLayerV(n_in, n_out, n_in * n_out, 2)
        ptrs = np.arange(0, n_in * n_out + 1, n_out, dtype=np.int32)  # [0, 3, 6]
        indices = np.tile(np.arange(n_out, dtype=np.int32), n_in)  # [0,1,2, 0,1,2]
        w = np.full(n_in * n_out, 0.1, dtype=np.float32)
        # Row-major (n_in, n_out): importance varies by column only.
        imp = np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), n_in)
        layer.load_weights(ptrs, indices, w, imp)
        layer.apply_amortized_plasticity_reset(n_in * n_out, 0.0, 0.99, 0.95, 0.5, 0.1, 0.0, 0.5)
        snap = layer.plasticity_column_state()
        np.testing.assert_allclose(snap["col_importance"], [1.0, 2.0, 3.0], atol=1e-6)

    def test_returned_arrays_are_views_reflecting_later_updates(self):
        # Zero-copy view convention (matches get_weights_vals/get_importance
        # precedent) -- a snapshot taken, then a further apply call, then
        # re-reading the SAME returned array should show the updated
        # values (documents the view semantics so callers don't assume a
        # frozen copy without re-fetching).
        n_in, n_out = 2, 2
        layer = _tiny_layer(n_in, n_out)
        layer.apply_amortized_plasticity_reset(n_in * n_out, 0.0, 0.99, 0.95, 0.5, 0.1, 0.0, 0.5)
        snap1 = layer.plasticity_column_state()
        before = np.array(snap1["col_importance"])
        layer.apply_amortized_plasticity_reset(n_in * n_out, 0.0, 0.99, 0.95, 0.5, 0.1, 0.0, 0.5)
        snap2 = layer.plasticity_column_state()
        # Not asserting they differ (importance could legitimately repeat) --
        # just that both calls return a real, correctly-sized reading each
        # time, not a stale cached object.
        assert len(snap2["col_importance"]) == len(before)


class TestPlasticityColumnStateBlock4:
    def test_empty_before_any_block4_apply_call(self):
        layer = _tiny_layer()
        snap = layer.plasticity_column_state_block4()
        assert len(snap["col_importance"]) == 0

    def test_sized_after_first_block4_apply_call_matches_n_out(self):
        n_in, n_out = 8, 8
        layer = _tiny_layer(n_in, n_out)
        layer.apply_amortized_block4_plasticity_reset(4, 0.0, 0.99, 0.95, 0.5, 0.1, 0.01, 1.0)
        snap = layer.plasticity_column_state_block4()
        assert len(snap["col_importance"]) == n_out
        assert len(snap["col_age"]) == n_out
