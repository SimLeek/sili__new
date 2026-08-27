"""
tests/unit/python/test_aqrs_channel_orthogonality.py
─────────────────────────────────────────────────────
Follow-up to task #295's scale-overflow-guard fix: nothing in AQRS's
existing design stops two rank channels from converging to duplicate
directions -- neurogenesis's own health check (abs_gamma_k/grad_ema) is
purely magnitude-based, and l1_sparsity_coef (sili_peridot) only sees
the SUMMED output after every channel's already combined, with no
visibility into the per-channel decomposition. Direct instruction:
prefer an ONGOING per-step orthogonality penalty over residual-targeted
growth (an init-time-only fix that doesn't stop channels drifting back
toward redundancy as training continues), since it's also simpler --
no new C++ state needed, since the penalty only depends on the CURRENT
parameter values (not any batch's dy/x), so it reuses the exact same
bulk-array plumbing task #295's overflow guard already built.

This file tests sili.sparse_rnn._orthogonality_penalty_array (the pure
math) and DISLDOLayer/DISLDOLayer8.apply_channel_orthogonality_penalty
(the real-layer wiring). No C++ changes were needed for this feature,
so there's no companion .cpp test.
"""
import numpy as np

from sili.sparse_rnn import DISLDOLayer, DISLDOLayer8, _orthogonality_penalty_array


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class TestOrthogonalityPenaltyArrayMath:
    def test_rank_le_1_is_a_noop(self):
        flat = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = _orthogonality_penalty_array(flat, rank=1, coef=0.5)
        np.testing.assert_array_equal(out, flat)

    def test_orthogonal_channels_are_left_alone(self):
        # 4 rows, 2 channels that are already exactly orthogonal
        # (e_0-like vs e_1-like vectors) -- correction should be exactly
        # zero since the off-diagonal Gram entry is already zero.
        m = np.array([[1.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 1.0],
                      [0.0, 1.0]], dtype=np.float32)
        flat = m.reshape(-1)
        out = _orthogonality_penalty_array(flat, rank=2, coef=0.5).reshape(4, 2)
        np.testing.assert_allclose(out, m, atol=1e-5)

    def test_duplicate_channels_are_pushed_apart(self):
        # Two NEAR-duplicate channels (small but real asymmetric
        # perturbation, not bit-identical) -- maximally redundant in
        # practice. One correction step should measurably reduce their
        # cosine similarity (can't reach exactly -1/0 in one step, but
        # must move the right way). Deliberately NOT an exact duplicate:
        # an exact tie is a symmetric fixed point of this penalty (both
        # columns get shrunk by the identical scalar factor, since the
        # correction is identical for both when the columns are
        # identical -- cosine similarity is scale-invariant, so it
        # can't change at all from a perfectly symmetric start). That's
        # expected, not a bug -- real channels never start as exact
        # float ties, since growth already seeds new channels with
        # independent noise (see _seed_scale_rank/_seed_additive_rank's
        # own symmetry-breaking rationale) -- but it means this test
        # must use a realistic near-duplicate, not a constructed exact
        # one, to test the property that actually matters.
        n, rank = 8, 2
        rng = np.random.default_rng(0)
        base = rng.standard_normal(n).astype(np.float32)
        near_dup = base + 0.05 * rng.standard_normal(n).astype(np.float32)
        m = np.stack([base, near_dup], axis=1)
        flat = m.reshape(-1)
        before_cos = _cosine(m[:, 0], m[:, 1])
        assert before_cos > 0.99  # sanity: really is near-duplicated

        out = _orthogonality_penalty_array(flat, rank, coef=0.05).reshape(n, rank)
        after_cos = _cosine(out[:, 0], out[:, 1])
        assert after_cos < before_cos

    def test_repeated_application_keeps_reducing_correlation(self):
        n, rank = 8, 3
        rng = np.random.default_rng(1)
        # start all 3 channels highly correlated (small random perturbations
        # of the same base direction)
        base = rng.standard_normal(n).astype(np.float32)
        m = np.stack([base + 0.05 * rng.standard_normal(n) for _ in range(rank)], axis=1).astype(np.float32)
        flat = m.reshape(-1)

        def max_pairwise_cos(mat):
            best = 0.0
            for i in range(rank):
                for j in range(i + 1, rank):
                    best = max(best, abs(_cosine(mat[:, i], mat[:, j])))
            return best

        initial = max_pairwise_cos(m)
        assert initial > 0.9
        prev = initial
        for _ in range(30):
            flat = _orthogonality_penalty_array(flat, rank, coef=0.05)
            cur = max_pairwise_cos(flat.reshape(n, rank))
            assert cur <= prev + 1e-4
            prev = cur
        # Relative decrease, not an arbitrary absolute cutoff -- the
        # real invariant is "measurably decorrelated", not any specific
        # residual correlation value.
        assert prev < initial - 0.02

    def test_channel_own_norm_is_not_penalized(self):
        # A single-channel-times-itself (diagonal) term must not shrink
        # a channel toward zero on its own -- only cross terms are
        # penalized (fill_diagonal(gram, 0) is the mechanism under test).
        n = 8
        rng = np.random.default_rng(2)
        m = rng.standard_normal((n, 3)).astype(np.float32) * 5.0  # orthogonal-ish random
        # Make columns pairwise near-orthogonal via QR so the only
        # signal left is each column's own norm.
        q, _ = np.linalg.qr(m)
        m = (q * 5.0).astype(np.float32)
        flat = m.reshape(-1)
        out = _orthogonality_penalty_array(flat, rank=3, coef=0.5).reshape(n, 3)
        for k in range(3):
            norm_before = np.linalg.norm(m[:, k])
            norm_after = np.linalg.norm(out[:, k])
            # near-orthogonal input -> off-diagonal Gram entries near 0
            # -> norm should barely move
            assert abs(norm_after - norm_before) < 0.5

    def test_real_world_magnitude_does_not_blow_up(self):
        # REGRESSION test for a real bug (see conversation): the first
        # version of this penalty computed the correction directly in
        # raw (non-normalized) space, so it scaled CUBICALLY with
        # channel magnitude (correction ~ M @ (M^T@M), and M^T@M itself
        # scales with magnitude^2). Every earlier test in this file
        # used near-unit-magnitude synthetic vectors, where that's
        # invisible -- but real training pushes AQRS channel magnitudes
        # into the same 10s-100s range apply_scale_overflow_guard's own
        # near=20/clip=200 thresholds exist to handle, and there the
        # cubic term exploded: NaN-collapsed a real fp8 MQAR run at
        # step 12650 (even earlier than the original unguarded-envelope
        # bug this whole mechanism is downstream of). Confirms the
        # normalized-space fix keeps the correction bounded and
        # reasonable at magnitudes drawn from that exact real range.
        n, rank = 128, 32  # matches the real q_proj/k_proj/v_proj/o_proj shape
        rng = np.random.default_rng(5)
        base_norms = rng.uniform(20.0, 200.0, size=rank).astype(np.float32)
        m = rng.standard_normal((n, rank)).astype(np.float32)
        m = m / np.linalg.norm(m, axis=0, keepdims=True) * base_norms  # exact target norms
        flat = m.reshape(-1)

        out = _orthogonality_penalty_array(flat, rank, coef=0.01).reshape(n, rank)
        assert np.all(np.isfinite(out))
        # The whole point: the correction must NOT be wildly larger
        # than the channels' own starting scale. A cubic blowup would
        # produce corrections in the millions/billions; the fixed
        # (normalized-space) version should keep the per-step move
        # within a small multiple of each channel's own norm.
        delta = np.linalg.norm(out - m, axis=0)
        assert np.all(delta < 50.0 * base_norms), (
            f"correction magnitude blew up relative to channel scale: "
            f"delta={delta}, base_norms={base_norms}")

    def test_repeated_application_at_real_magnitude_stays_finite(self):
        # Same real-magnitude setup as above, but applied repeatedly
        # (as it would be every training step) -- confirms the fix
        # holds up over many steps, not just one.
        n, rank = 128, 32
        rng = np.random.default_rng(6)
        base_norms = rng.uniform(20.0, 200.0, size=rank).astype(np.float32)
        m = rng.standard_normal((n, rank)).astype(np.float32)
        m = m / np.linalg.norm(m, axis=0, keepdims=True) * base_norms
        flat = m.reshape(-1)
        for _ in range(200):
            flat = _orthogonality_penalty_array(flat, rank, coef=0.01)
            assert np.all(np.isfinite(flat))
            assert np.max(np.abs(flat)) < 1e5  # nowhere near fp32 overflow


def _make_dense_layer(cls, n_in=8, n_out=8, budget=2000, cpus=1, seed=0):
    rng = np.random.default_rng(seed)
    return cls(n_in, n_out, budget, cpus, rng=rng, dense=True)


class TestApplyChannelOrthogonalityPenaltyIntegration:
    def test_fp4_layer_rank1_is_a_safe_noop(self):
        layer = _make_dense_layer(DISLDOLayer)
        before = layer._c.get_value_scale_raw_vector()
        layer.apply_channel_orthogonality_penalty()  # scale_rank defaults to 1
        after = layer._c.get_value_scale_raw_vector()
        assert list(before) == list(after)

    def test_fp4_layer_additive_rank0_is_a_safe_noop(self):
        layer = _make_dense_layer(DISLDOLayer)
        assert len(layer._c.get_additive_u_raw_vector()) == 0
        layer.apply_channel_orthogonality_penalty()  # must not raise

    def test_fp4_layer_duplicate_value_scale_channels_get_decorrelated(self):
        layer = _make_dense_layer(DISLDOLayer, n_in=8)
        layer._c.set_scale_rank_max(4)
        layer._c.set_scale_rank(2)
        rng = np.random.default_rng(3)
        base = rng.standard_normal(8).astype(np.float32)
        near_dup = base + 0.05 * rng.standard_normal(8).astype(np.float32)
        for r in range(8):
            layer._c.set_value_scale_raw_k(r, 0, float(base[r]))
            layer._c.set_value_scale_raw_k(r, 1, float(near_dup[r]))  # near-duplicate channel -- see
            # test_duplicate_channels_are_pushed_apart's own docstring for why
            # NOT an exact duplicate (symmetric fixed point of this penalty)

        def get_cols():
            flat = np.asarray(layer._c.get_value_scale_raw_vector(), dtype=np.float32).reshape(8, 2)
            return flat[:, 0], flat[:, 1]

        c0, c1 = get_cols()
        before_cos = _cosine(c0, c1)
        assert before_cos > 0.99

        for _ in range(10):
            layer.apply_channel_orthogonality_penalty(coef=0.05)
        c0, c1 = get_cols()
        after_cos = _cosine(c0, c1)
        assert after_cos < before_cos
        assert np.all(np.isfinite(c0)) and np.all(np.isfinite(c1))

    def test_fp8_layer_duplicate_additive_channels_get_decorrelated(self):
        layer = _make_dense_layer(DISLDOLayer8, n_in=8)
        layer._c.set_additive_rank_max(4)
        layer._c.set_additive_rank(2)
        rng = np.random.default_rng(4)
        base = rng.standard_normal(8).astype(np.float32)
        near_dup = base + 0.05 * rng.standard_normal(8).astype(np.float32)
        for r in range(8):
            layer._c.set_additive_u_raw_k(r, 0, float(base[r]))
            layer._c.set_additive_u_raw_k(r, 1, float(near_dup[r]))  # near-duplicate, not exact -- see
            # test_duplicate_channels_are_pushed_apart's own docstring

        def get_cols():
            flat = np.asarray(layer._c.get_additive_u_raw_vector(), dtype=np.float32).reshape(8, 2)
            return flat[:, 0], flat[:, 1]

        c0, c1 = get_cols()
        before_cos = _cosine(c0, c1)
        assert before_cos > 0.99
        for _ in range(10):
            layer.apply_channel_orthogonality_penalty(coef=0.05)
        c0, c1 = get_cols()
        assert _cosine(c0, c1) < before_cos

    def test_combined_with_overflow_guard_stays_finite(self):
        # Real usage: both correction passes run every step. Confirm
        # they compose without producing NaN/Inf even with an
        # artificially large starting value.
        layer = _make_dense_layer(DISLDOLayer, n_in=8)
        layer._c.set_scale_rank_max(4)
        layer._c.set_scale_rank(3)
        layer._c.set_value_scale_raw_k(0, 1, 500.0)
        layer._c.set_value_scale_raw_k(0, 2, 500.0)
        for _ in range(20):
            layer.apply_channel_orthogonality_penalty(coef=0.02)
            layer.apply_scale_overflow_guard(clip=200.0, near=20.0, coef=0.1)
        flat = np.asarray(layer._c.get_value_scale_raw_vector(), dtype=np.float32)
        assert np.all(np.isfinite(flat))
        assert np.all(np.abs(flat) <= 200.0 + 1e-3)
