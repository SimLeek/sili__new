"""
tests/unit/python/test_disldo_layer32_energy_lr_scale.py
────────────────────────────────────────────────────────
Arm H: per-output-neuron post-clip learning-rate multiplier, threaded
from DISLDOLayer32.forward() through _cpu.DISLDOLayerV.backward_sparse's
new energy_lr_scale parameter. The underlying C++ multiplication is
covered directly in
tests/unit/test_disldo_backward_sparse_grad_energy_lr_scale.cpp; this
file covers the Python-level routing/guard behavior specifically.
"""

from __future__ import annotations

import numpy as np
import pytest

from sili.sparse_rnn import DISLDOLayer32
from sili.tensor import Tensor


def _make_layer(n=4, seed=0):
    return DISLDOLayer32(n, n, n * n, num_cpus=1, dense=True, rng=np.random.default_rng(seed))


class TestEnergyLrScaleRouting:
    def test_requires_sparsity_routing(self):
        layer = _make_layer()
        x = Tensor(np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32))
        with pytest.raises(ValueError, match="energy_lr_scale requires routing through backward_sparse"):
            layer.forward(x, learning_rate=0.05, energy_lr_scale=np.ones(4, dtype=np.float32))

    def test_works_with_dy_sparsity_p(self):
        layer = _make_layer()
        x = Tensor(np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32))
        out = layer.forward(x, learning_rate=0.05, dy_sparsity_p=1.0, energy_lr_scale=np.ones(4, dtype=np.float32))
        out.backward()  # must not raise
        assert x.grad is not None

    def test_works_with_dy_gate_mask(self):
        layer = _make_layer()
        x = Tensor(np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32))
        mask = np.array([True, True, False, True])
        out = layer.forward(x, learning_rate=0.05, dy_gate_mask=mask, energy_lr_scale=np.ones(4, dtype=np.float32))
        out.backward()
        assert x.grad is not None

    def test_none_is_backward_compatible_noop(self):
        # energy_lr_scale=None (default) must behave identically to
        # never having added the parameter at all. Both the layer's own
        # weight-init RNG and the input must match for a fair comparison.
        layer_a = _make_layer(seed=42)
        layer_b = _make_layer(seed=42)
        x_np = np.array([0.31, 0.42, 0.53, 0.64], dtype=np.float32)
        xa = Tensor(x_np.copy())
        xb = Tensor(x_np.copy())
        out_a = layer_a.forward(xa, learning_rate=0.05, dy_sparsity_p=1.0)
        out_b = layer_b.forward(xb, learning_rate=0.05, dy_sparsity_p=1.0, energy_lr_scale=None)
        out_a.backward()
        out_b.backward()
        assert np.allclose(xa.grad, xb.grad)


class TestEnergyLrScaleEffect:
    def test_scale_measurably_changes_post_update_forward_output(self):
        # The exact per-column multiplicative math is verified directly
        # in the C++ test; this confirms the kwarg genuinely reaches and
        # influences training end-to-end through the Python routing,
        # not just "doesn't crash" -- a scaled column's weights should
        # have moved further, changing the NEXT forward pass's output
        # at that column vs an unscaled run from the same start point.
        n = 4
        x_np = np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32)

        def run(scale):
            layer = _make_layer(n)
            y0 = layer._c.forward_dense(x_np.reshape(1, n))
            x = Tensor(x_np.copy())
            kwargs = {"learning_rate": 0.2, "dy_sparsity_p": 1.0}
            if scale is not None:
                kwargs["energy_lr_scale"] = scale
            out = layer.forward(x, **kwargs)
            out.backward()
            y1 = layer._c.forward_dense(x_np.reshape(1, n))
            return np.asarray(y0).ravel(), np.asarray(y1).ravel()

        y0_base, y1_base = run(None)
        y0_scaled, y1_scaled = run(np.array([1.0, 5.0, 1.0, 1.0], dtype=np.float32))

        move_base = np.abs(y1_base - y0_base)
        move_scaled = np.abs(y1_scaled - y0_scaled)
        # Column 1 (scale=5x) must move noticeably more than the
        # baseline; untouched columns (scale=1x) must not.
        assert move_scaled[1] > move_base[1] * 1.5, (
            f"scaled column 1 didn't move further: base={move_base[1]:.6f} scaled={move_scaled[1]:.6f}"
        )
        for c in (0, 2, 3):
            assert np.isclose(move_scaled[c], move_base[c], rtol=0.05), (
                f"untouched column {c} moved differently: base={move_base[c]:.6f} scaled={move_scaled[c]:.6f}"
            )
