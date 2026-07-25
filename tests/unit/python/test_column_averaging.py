"""
tests/unit/python/test_column_averaging.py
────────────────────────────────────────────
Tests for the column-averaging mechanism (sili_peridot/todolist.md Phase
A3/A4): sili.energy.column_averaging_loss (A3) and
sili.sparse_rnn.FoldedColumnLayer (A4).
"""
from __future__ import annotations
import numpy as np
import pytest

from sili.tensor import Tensor, combine_losses
from sili.energy import column_averaging_loss, EnergyDynamics


def _numeric_grad(fn, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = fn()
        x[idx] = orig - eps
        f_minus = fn()
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


class TestColumnAveragingLossGradient:
    def test_matches_finite_difference(self):
        n_folds, input_size = 5, 4
        rng = np.random.default_rng(0)
        h_np = rng.standard_normal(n_folds * input_size).astype(np.float32)
        target_np = rng.standard_normal(input_size).astype(np.float32)

        h = Tensor(h_np.copy())
        target = Tensor(target_np.copy())
        loss = column_averaging_loss(h, target, n_folds=n_folds)
        loss.backward()

        def fn():
            hh = Tensor(h.data.copy())
            tt = Tensor(target_np.copy())
            return float(column_averaging_loss(hh, tt, n_folds=n_folds).data)

        numeric = _numeric_grad(fn, h.data)
        np.testing.assert_allclose(h.grad, numeric, atol=2e-3, rtol=2e-3)

    def test_gradient_uniform_within_column(self):
        # d(loss)/d(h[t,i]) should be identical for every t within column i
        # (the loss only depends on each column's MEAN, not on which fold
        # step contributed it).
        n_folds, input_size = 6, 3
        h = Tensor(np.random.randn(n_folds * input_size).astype(np.float32))
        target = Tensor(np.random.randn(input_size).astype(np.float32))
        column_averaging_loss(h, target, n_folds=n_folds).backward()
        grad2d = h.grad.reshape(n_folds, input_size)
        for i in range(input_size):
            np.testing.assert_allclose(grad2d[:, i], grad2d[0, i], atol=1e-6)

    def test_zero_loss_when_column_means_match_target_exactly(self):
        n_folds, input_size = 4, 3
        target_np = np.array([1.0, -2.0, 0.5], dtype=np.float32)
        # Every fold step holds exactly the target -> column means == target.
        h_np = np.tile(target_np, n_folds)
        h = Tensor(h_np)
        target = Tensor(target_np.copy())
        loss = column_averaging_loss(h, target, n_folds=n_folds)
        assert float(loss.data) == pytest.approx(0.0, abs=1e-6)

    def test_shape_mismatch_raises(self):
        h = Tensor(np.zeros(10, dtype=np.float32))  # not n_folds*input_size
        target = Tensor(np.zeros(4, dtype=np.float32))
        with pytest.raises(AssertionError):
            column_averaging_loss(h, target, n_folds=3)


class TestColumnAveragingLossTraining:
    def test_sgd_moves_column_mean_toward_target(self):
        # A minimal end-to-end check: treat h as a "parameter" and confirm
        # a few plain SGD steps against this loss actually drive each
        # column's mean toward its target -- not just that SOME gradient
        # exists, but that following it converges.
        n_folds, input_size = 4, 3
        rng = np.random.default_rng(1)
        h_np = rng.standard_normal(n_folds * input_size).astype(np.float32)
        target_np = rng.standard_normal(input_size).astype(np.float32)
        lr = 0.5

        for _ in range(200):
            h = Tensor(h_np.copy())
            target = Tensor(target_np.copy())
            loss = column_averaging_loss(h, target, n_folds=n_folds)
            loss.backward()
            h_np -= lr * h.grad

        final_col_mean = h_np.reshape(n_folds, input_size).mean(axis=0)
        np.testing.assert_allclose(final_col_mean, target_np, atol=1e-3)

    def test_combines_with_energy_aux_loss_via_combine_losses(self):
        # The documented intended usage: column_averaging_loss +
        # EnergyDynamics's own aux_loss, combined via combine_losses,
        # backward() called exactly once.
        n_folds, input_size = 4, 3
        state_size = n_folds * input_size
        ed = EnergyDynamics(drive=0.1, activation_cost=0.05, precision=0.01,
                            density=0.05, p=0.3)
        h_in = Tensor((np.random.randn(state_size) * 0.5).astype(np.float32))
        h_out, aux_loss, _ = ed.forward(h_in)
        target = Tensor(np.random.randn(input_size).astype(np.float32))
        col_loss = column_averaging_loss(h_out, target, n_folds=n_folds)

        total = combine_losses(aux_loss, col_loss)
        total.backward()
        assert np.all(np.isfinite(h_in.grad))
        assert not np.allclose(h_in.grad, 0.0)

    def test_weight_scales_loss_and_gradient(self):
        n_folds, input_size = 4, 3
        h_np = np.random.randn(n_folds * input_size).astype(np.float32)
        target_np = np.random.randn(input_size).astype(np.float32)

        h1 = Tensor(h_np.copy())
        loss1 = column_averaging_loss(h1, Tensor(target_np.copy()), n_folds=n_folds, weight=1.0)
        loss1.backward()

        h2 = Tensor(h_np.copy())
        loss2 = column_averaging_loss(h2, Tensor(target_np.copy()), n_folds=n_folds, weight=3.0)
        loss2.backward()

        assert float(loss2.data) == pytest.approx(float(loss1.data) * 3.0, rel=1e-5)
        np.testing.assert_allclose(h2.grad, h1.grad * 3.0, rtol=1e-5)
