"""
tests/unit/python/test_aqrs_scale_overflow_guard.py
──────────────────────────────────────────────────────
Task #295 follow-up: raising scale_rank_max/additive_rank_max past the
old hardcoded 4 let a real fp8 MQAR curriculum run's per-channel
value_scale_k/output_scale_k grow unbounded (get_scale()'s combined
envelope S(row,col) has no clamp anywhere in delta_csr_types.hpp),
overflowing S in the forward pass and NaN-collapsing training (real
run, see conversation/JOURNAL.md). This file tests the fix: sili.
sparse_rnn._overflow_guard_array (the pure elementwise correction) and
DISLDOLayer/DISLDOLayer8.apply_scale_overflow_guard (the bulk C++
round-trip wiring). C++ coverage of the underlying bulk-vector
accessors already exists (test_aqrs_scale_overflow_guard.cpp) -- this
file is deliberately NOT re-deriving that; it confirms the Python-side
correction MATH and that it's actually reachable through a real layer.
"""
import numpy as np

from sili.sparse_rnn import DISLDOLayer, DISLDOLayer8, _overflow_guard_array


class TestOverflowGuardArrayMath:
    def test_values_within_near_pass_through_unchanged(self):
        x = np.array([-5.0, -1.0, 0.0, 3.0, 10.0], dtype=np.float32)
        out = _overflow_guard_array(x, clip=200.0, near=20.0, coef=0.1)
        np.testing.assert_allclose(out, x, atol=1e-5)

    def test_values_past_near_get_shrunk_toward_zero(self):
        # near=20, so a value of 50 has excess=30 -- corrected should be
        # strictly closer to zero than the original, and same sign.
        x = np.array([50.0, -50.0], dtype=np.float32)
        out = _overflow_guard_array(x, clip=200.0, near=20.0, coef=0.1)
        assert abs(out[0]) < abs(x[0])
        assert abs(out[1]) < abs(x[1])
        assert out[0] > 0 and out[1] < 0  # sign preserved, not flipped

    def test_hard_clip_never_exceeded(self):
        x = np.array([1e9, -1e9, 5e6], dtype=np.float32)
        out = _overflow_guard_array(x, clip=200.0, near=20.0, coef=0.1)
        assert np.all(np.abs(out) <= 200.0 + 1e-3)

    def test_nan_and_inf_are_sanitized_not_passed_through(self):
        # The real bug: np.clip(nan, ...) == nan, so a plain clip alone
        # would NOT fix an already-NaN value_scale_k -- confirms the
        # explicit nan_to_num step actually does something.
        x = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
        out = _overflow_guard_array(x, clip=200.0, near=20.0, coef=0.1)
        assert np.all(np.isfinite(out))
        assert out[1] > 0  # +inf -> +clip (sign preserved)
        assert out[2] < 0  # -inf -> -clip

    def test_near_at_or_below_zero_is_a_pure_hard_clip(self):
        # near=0 degenerates to "always apply correction" -- sanity
        # check the formula doesn't blow up at the boundary.
        x = np.array([0.5, -0.5], dtype=np.float32)
        out = _overflow_guard_array(x, clip=200.0, near=0.0, coef=0.1)
        assert np.all(np.isfinite(out))


def _make_dense_layer(cls, n_in=8, n_out=8, budget=2000, cpus=1, seed=0):
    rng = np.random.default_rng(seed)
    return cls(n_in, n_out, budget, cpus, rng=rng, dense=True)


class TestApplyScaleOverflowGuardIntegration:
    def test_fp4_layer_shrinks_artificially_inflated_value_scale(self):
        layer = _make_dense_layer(DISLDOLayer)
        layer._c.set_scale_rank_max(4)
        layer._c.set_scale_rank(4)
        # Directly inflate one channel to a dangerous magnitude, as if
        # several RMSprop steps had driven it there.
        layer._c.set_value_scale_raw_k(0, 1, 5000.0)
        before = layer._c.get_value_scale_k(0, 1)
        assert before == 5000.0

        layer.apply_scale_overflow_guard(clip=200.0, near=20.0, coef=0.1)
        after = layer._c.get_value_scale_k(0, 1)
        assert after < before
        assert after <= 200.0 + 1e-3
        assert np.isfinite(after)

    def test_fp4_layer_untouched_channel_stays_within_near(self):
        layer = _make_dense_layer(DISLDOLayer)
        layer._c.set_scale_rank_max(4)
        layer._c.set_scale_rank(4)
        before = layer._c.get_value_scale_k(0, 0)  # component 0 default 1.0
        layer.apply_scale_overflow_guard(clip=200.0, near=20.0, coef=0.1)
        after = layer._c.get_value_scale_k(0, 0)
        assert after == before  # well within `near`, untouched

    def test_fp4_layer_rank0_additive_is_a_safe_noop(self):
        # additive_rank defaults to 0 -- get_additive_u_raw_vector()
        # should be empty, and apply_scale_overflow_guard must not
        # error on an empty array (see _apply_scale_overflow_guard's
        # own `if au.size:` guard).
        layer = _make_dense_layer(DISLDOLayer)
        assert len(layer._c.get_additive_u_raw_vector()) == 0
        layer.apply_scale_overflow_guard()  # must not raise

    def test_fp8_layer_shrinks_artificially_inflated_output_scale(self):
        layer = _make_dense_layer(DISLDOLayer8)
        layer._c.set_scale_rank_max(4)
        layer._c.set_scale_rank(4)
        layer._c.set_output_scale_raw_k(0, 2, -8000.0)
        before = layer._c.get_output_scale_k(0, 2)
        assert before == -8000.0

        layer.apply_scale_overflow_guard(clip=200.0, near=20.0, coef=0.1)
        after = layer._c.get_output_scale_k(0, 2)
        assert after > before  # moved toward zero (less negative)
        assert after >= -200.0 - 1e-3
        assert np.isfinite(after)

    def test_fp4_layer_additive_channel_gets_corrected_when_present(self):
        layer = _make_dense_layer(DISLDOLayer)
        layer._c.set_additive_rank_max(4)
        layer._c.set_additive_rank(4)
        layer._c.set_additive_u_raw_k(0, 1, 3000.0)
        before = layer._c.get_additive_u_k(0, 1)
        layer.apply_scale_overflow_guard(clip=200.0, near=20.0, coef=0.1)
        after = layer._c.get_additive_u_k(0, 1)
        assert after < before
        assert np.isfinite(after)

    def test_repeated_calls_converge_not_diverge(self):
        # Applying the guard repeatedly (as it would be every training
        # step) must monotonically shrink an over-threshold value, not
        # oscillate or grow -- a real correctness property of the
        # auto-correcting design, not just a one-shot check.
        layer = _make_dense_layer(DISLDOLayer)
        layer._c.set_scale_rank_max(4)
        layer._c.set_scale_rank(4)
        layer._c.set_value_scale_raw_k(0, 1, 1000.0)
        prev = 1000.0
        for _ in range(20):
            layer.apply_scale_overflow_guard(clip=200.0, near=20.0, coef=0.1)
            cur = layer._c.get_value_scale_k(0, 1)
            assert cur <= prev + 1e-6
            prev = cur
        assert prev < 1000.0
