"""
tests/integration/test_disldo_layer8.py
─────────────────────────────────────────
DISLDOLayer8 (sili/sparse_rnn.py -> _cpu.SparseLinearLayer8) -- real 8-bit
(OCP MX E4M3) storage, same disldo_forward/disldo_backward kernels as
production DISLDOLayer (FP4) and the DISLDOLayer32 diagnostic fallback,
generic via ValueAccessor<FP8BiValues> (fp8quant.hpp/delta_csr_types.hpp).

Built after sili_peridot's toy-model quantization sweep validated
"8-bit + rank-1 scale (row*col), weight AND importance both quantized"
as consistently, substantially better than native FP4 across three task
families (tanh-RNN, transformer MQAR, tile-recurrence MQAR) -- ten
configs, never lost to FP4. See sili_peridot's JOURNAL.md for the full
writeup. These tests check the real C++/pybind plumbing that scheme now
runs on, not the scheme itself (already validated in Python).

SCOPE: scattered CSR path only, matching DISLDOLayer32's own scope
(no block4 dense-tile SIMD promotion yet -- real follow-up, see
DISLDOLayer8's own class docstring for why that's not just a template
swap).

Run: python -m pytest tests/integration/test_disldo_layer8.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from sili.sparse_rnn import DISLDOLayer8, DISLDOLayer32

HIDDEN, VOCAB = 12, 10
MAX_WEIGHTS = HIDDEN * VOCAB  # generous, fully-dense-capable at this toy scale


class TestDISLDOLayer8Basic:
    def test_forward_shape_and_finite(self):
        layer = DISLDOLayer8(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2,
                             rng=np.random.default_rng(0))
        x = np.random.RandomState(1).randn(4, HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.0)
        assert out.data.shape == (4, VOCAB)
        assert np.all(np.isfinite(out.data))

    def test_online_1d_input_squeezes_back_to_1d(self):
        layer = DISLDOLayer8(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2,
                             rng=np.random.default_rng(0))
        x = np.random.RandomState(2).randn(HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.0)
        assert out.data.shape == (VOCAB,)

    def test_weights_change_after_backward_with_nonzero_lr(self):
        layer = DISLDOLayer8(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2,
                             rng=np.random.default_rng(0))
        before = np.array(layer._c.weights_vals, copy=True)
        x = np.random.RandomState(3).randn(4, HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.05)
        out.grad = np.ones_like(out.data)
        out.backward()
        after = layer._c.weights_vals
        assert np.all(np.isfinite(after))
        assert not np.allclose(before, after)

    def test_eval_call_with_zero_learning_rate_does_not_change_weights(self):
        layer = DISLDOLayer8(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2,
                             rng=np.random.default_rng(0))
        before = np.array(layer._c.weights_vals, copy=True)
        x = np.random.RandomState(4).randn(4, HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.0)
        out.grad = np.ones_like(out.data)
        out.backward()
        after = layer._c.weights_vals
        np.testing.assert_allclose(before, after)

    def test_load_weights_round_trips_within_e4m3_quantization_error(self):
        # Unlike DISLDOLayer32 (exact fp32 storage), DISLDOLayer8 really
        # does quantize -- weights_vals should be CLOSE to what was
        # loaded, not bit-identical. Checks the codec is actually wired
        # in (not silently bypassed) without duplicating fp8_bitshift's
        # own exact-precision tests.
        layer = DISLDOLayer8(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2,
                             rng=np.random.default_rng(0))
        rng = np.random.RandomState(5)
        ptrs = np.array(layer._c.ptrs, copy=True)
        indices = np.array(layer._c.indices, copy=True)
        n = len(indices)
        vals = (rng.standard_normal(n) * 0.3).astype(np.float32)
        imp = np.zeros(n, dtype=np.float32)
        layer._c.load_weights(ptrs, indices, vals, imp)
        stored = layer._c.weights_vals
        assert np.all(np.isfinite(stored))
        # E4M3's relative error is a few percent per value away from
        # zero -- generous absolute+relative tolerance, not pinning an
        # exact bound (that's fp8_bitshift.cpp's job).
        np.testing.assert_allclose(stored, vals, atol=0.02, rtol=0.15)
        # And it's NOT bit-exact (proves quantization actually happened).
        assert not np.array_equal(stored, vals)


class TestDISLDOLayer8RankOneScale:
    """The C++ value_scale/output_scale mechanism (SparseLinearWeightsDelta,
    VALUES_TYPE-agnostic) is what makes "8-bit + rank-1" the scheme
    actually validated, not just row-scale alone -- checked directly
    that both halves are real and reachable from Python."""

    def test_default_scales_are_one(self):
        layer = DISLDOLayer8(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2,
                             rng=np.random.default_rng(0))
        assert layer._c.get_value_scale(0) == 1.0
        assert layer._c.get_output_scale(0) == 1.0

    def test_set_value_and_output_scale_raw(self):
        layer = DISLDOLayer8(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2,
                             rng=np.random.default_rng(0))
        layer._c.set_value_scale_raw(2, 3.5)
        layer._c.set_output_scale_raw(4, 0.25)
        assert layer._c.get_value_scale(2) == pytest.approx(3.5)
        assert layer._c.get_output_scale(4) == pytest.approx(0.25)
        # untouched rows/cols stay at the default
        assert layer._c.get_value_scale(0) == 1.0
        assert layer._c.get_output_scale(0) == 1.0


class TestDISLDOLayer8Synaptogenesis:
    """Growth/pruning through the real C++/pybind path (build_probes ->
    synap_row_step -> block4 promotion/demotion, delta_csr_memory.hpp).
    Regression test for a real gap found this session: Block4View8 (the
    class SparseLinearLayer8.block4 returns) was never registered with
    pybind, so this raised "Unregistered type: Block4View8" from Python
    despite compiling and working fine in pure C++ -- the underlying
    promotion/demotion logic itself is validated much more thoroughly at
    the C++ level (test_disldo_block4_promotion_fp8.cpp, under ASan)."""

    def test_growth_and_block4_introspection_do_not_crash(self):
        layer = DISLDOLayer8(8, 8, 256, num_cpus=1, rng=np.random.default_rng(0))
        x = np.zeros((1, 8), dtype=np.float32)
        for i in range(8):
            xi = x.copy()
            xi[0, i] = 1.0
            layer._c.forward(xi)
            layer._c.backward(np.ones((1, 8), dtype=np.float32), 0.05,
                              lr_per_row_nnz=False, damp_by_importance=True)

        layer._c.build_probes(1, per_row=True)
        row = 0
        for _ in range(8):
            layer._c.synap_row_step(row, -1e9, 4)
            row = (row + 1) % 8

        # The real regression check: touching .block4 must not raise
        # "Unregistered type" (or anything else).
        assert layer._c.block4.tiles >= 0
        assert layer._c.block4.synapses >= 0
        assert np.all(np.isfinite(layer._c.weights_vals))
        assert np.all(np.isfinite(layer._c.importance))

        out = layer.forward(np.random.RandomState(1).randn(1, 8).astype(np.float32), learning_rate=0.05)
        out.grad = np.ones_like(out.data)
        out.backward()
        assert np.all(np.isfinite(layer._c.weights_vals))

        for i in range(8):
            layer._c.synap_row_step(i, 1e9, 4)
        assert layer._c.block4.tiles >= 0  # still reachable after pruning


class TestDISLDOLayer8TrainingConvergence:
    """Real online-regression convergence check (predict a fixed random
    target layer's own output), comparable to DISLDOLayer32 at a
    properly-tuned learning rate -- NOT at an aggressive rate (this
    session's own transformer-harness investigation found DISLDOLayer
    -family's raw `learning_rate` diverges at rates tuned for Adam's
    normalized step, for EVERY storage type including fp32 -- an
    LR-mismatch property of the online update itself, not something
    specific to FP8). Confirms the real C++/pybind path trains, not
    just that it computes finite numbers."""

    N_IN, N_OUT, MAX_W = 6, 6, 24
    LR = 1e-3
    N_STEPS = 800

    def _run(self, cls):
        target = cls(self.N_IN, self.N_OUT, self.MAX_W, num_cpus=1,
                     rng=np.random.default_rng(1))
        trainee = cls(self.N_IN, self.N_OUT, self.MAX_W, num_cpus=1,
                      rng=np.random.default_rng(2))
        rng = np.random.default_rng(3)
        losses = []
        for _ in range(self.N_STEPS):
            x = rng.standard_normal(self.N_IN).astype(np.float32)
            y = target.forward(x, 0.0).data
            out = trainee.forward(x, self.LR)
            err = out.data - y
            losses.append(float(np.mean(err ** 2)))
            out.grad = (2.0 / self.N_OUT * err).astype(np.float32)
            out.backward()
        return float(np.mean(losses[:20])), float(np.mean(losses[-20:]))

    def test_loss_decreases_and_stays_finite(self):
        first, last = self._run(DISLDOLayer8)
        assert np.isfinite(first) and np.isfinite(last)
        assert last < first, f"loss did not decrease: first={first:.4f} last={last:.4f}"

    def test_comparable_to_fp32_reference_at_same_lr(self):
        f8_first, f8_last = self._run(DISLDOLayer8)
        f32_first, f32_last = self._run(DISLDOLayer32)
        assert np.isfinite(f8_last) and np.isfinite(f32_last)
        # Generous bound (not "matches exactly") -- 8-bit storage is
        # expected to trail fp32 somewhat, per the harder-task results
        # in sili_peridot's own validation sweep, not required to be a
        # tight match.
        assert f8_last < f32_last * 3.0, (
            f"FP8 final loss {f8_last:.4f} far worse than FP32's {f32_last:.4f} "
            f"(3x bound) -- possible real regression, not just the expected "
            f"8-bit-vs-fp32 storage gap")
