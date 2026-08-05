"""
tests/unit/python/test_attention_autograd.py
─────────────────────────────────────────────
Gradient-check tests for the Tensor-graph attention wrappers in
sili/tensor.py (sparse_attention, banded_attention, sparse_banded_attention).

The C++ forward/backward kernels themselves are already covered by
test_sili.py's TestSparseAttention/TestBandedAttention/TestSparseBandedAttention
(forward shape/value checks only). What was missing -- and what this file
covers -- is that those kernels were never wired into sili.tensor.Tensor's
autograd graph, so nothing using attention could be trained end-to-end. These
tests exist to prove the wiring is actually correct (dQ/dK/dV match numerical
finite differences), not just that it runs without crashing.
"""
from __future__ import annotations
import numpy as np
import pytest

from sili.tensor import (
    Tensor, sparse_attention, banded_attention, sparse_banded_attention,
    gaussian_attention, exp,
)


def _numeric_grad(fn, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Central-difference gradient of a scalar-valued fn w.r.t. array x."""
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


def _random_qkv(T, K, d, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((T, d)).astype(np.float32) * 0.5
    k = rng.standard_normal((K, d)).astype(np.float32) * 0.5
    v = rng.standard_normal((K, d)).astype(np.float32) * 0.5
    return q, k, v


def _scalar_loss(out: Tensor) -> Tensor:
    # An asymmetric weighting so the gradient isn't accidentally uniform
    # (a bug that scales every output element the same way would still pass
    # a plain .sum() check but would fail this).
    T, d = out.shape
    weights = (np.arange(T * d, dtype=np.float32).reshape(T, d) + 1.0)
    return (out * weights).sum()


class TestSparseAttentionAutograd:
    def _loss_fn(self, q_np, k_np, v_np, top_k):
        def fn():
            q = Tensor(q_np.copy())
            k = Tensor(k_np.copy())
            v = Tensor(v_np.copy())
            out = sparse_attention(q, k, v, top_k=top_k)
            loss = _scalar_loss(out)
            return float(loss.data)
        return fn

    def test_dQ_matches_finite_difference(self):
        T, d = 6, 5
        q_np, k_np, v_np = _random_qkv(T, T, d, seed=0)
        top_k = 3

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        out = sparse_attention(q, k, v, top_k=top_k)
        loss = _scalar_loss(out)
        loss.backward()

        numeric = _numeric_grad(lambda: self._loss_fn(q.data, k_np, v_np, top_k)(), q.data)
        np.testing.assert_allclose(q.grad, numeric, atol=2e-2, rtol=2e-2)

    def test_dK_matches_finite_difference(self):
        T, d = 6, 5
        q_np, k_np, v_np = _random_qkv(T, T, d, seed=1)
        top_k = 3

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        out = sparse_attention(q, k, v, top_k=top_k)
        loss = _scalar_loss(out)
        loss.backward()

        numeric = _numeric_grad(lambda: self._loss_fn(q_np, k.data, v_np, top_k)(), k.data)
        np.testing.assert_allclose(k.grad, numeric, atol=2e-2, rtol=2e-2)

    def test_dV_matches_finite_difference(self):
        T, d = 6, 5
        q_np, k_np, v_np = _random_qkv(T, T, d, seed=2)
        top_k = 3

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        out = sparse_attention(q, k, v, top_k=top_k)
        loss = _scalar_loss(out)
        loss.backward()

        numeric = _numeric_grad(self._loss_fn(q_np, k_np, v.data, top_k), v.data)
        np.testing.assert_allclose(v.grad, numeric, atol=2e-2, rtol=2e-2)


class TestBandedAttentionAutograd:
    def _loss_fn(self, q_np, k_np, v_np, half_bw):
        def fn():
            q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
            out = banded_attention(q, k, v, half_bandwidth=half_bw)
            return float(_scalar_loss(out).data)
        return fn

    def test_dQ_matches_finite_difference(self):
        T, K, d = 7, 9, 5
        q_np, k_np, v_np = _random_qkv(T, K, d, seed=10)
        half_bw = 2

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        out = banded_attention(q, k, v, half_bandwidth=half_bw)
        _scalar_loss(out).backward()

        numeric = _numeric_grad(self._loss_fn(q.data, k_np, v_np, half_bw), q.data)
        np.testing.assert_allclose(q.grad, numeric, atol=2e-2, rtol=2e-2)

    def test_dK_matches_finite_difference(self):
        T, K, d = 7, 9, 5
        q_np, k_np, v_np = _random_qkv(T, K, d, seed=11)
        half_bw = 2

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        out = banded_attention(q, k, v, half_bandwidth=half_bw)
        _scalar_loss(out).backward()

        numeric = _numeric_grad(self._loss_fn(q_np, k.data, v_np, half_bw), k.data)
        np.testing.assert_allclose(k.grad, numeric, atol=2e-2, rtol=2e-2)

    def test_dV_matches_finite_difference(self):
        T, K, d = 7, 9, 5
        q_np, k_np, v_np = _random_qkv(T, K, d, seed=12)
        half_bw = 2

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        out = banded_attention(q, k, v, half_bandwidth=half_bw)
        _scalar_loss(out).backward()

        numeric = _numeric_grad(self._loss_fn(q_np, k_np, v.data, half_bw), v.data)
        np.testing.assert_allclose(v.grad, numeric, atol=2e-2, rtol=2e-2)


class TestSparseBandedAttentionAutograd:
    def _loss_fn(self, q_np, k_np, v_np, half_bw, inner_k):
        def fn():
            q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
            out = sparse_banded_attention(q, k, v, half_bandwidth=half_bw, inner_k=inner_k)
            return float(_scalar_loss(out).data)
        return fn

    def test_dQ_matches_finite_difference(self):
        T, K, d = 8, 8, 5
        q_np, k_np, v_np = _random_qkv(T, K, d, seed=20)
        half_bw, inner_k = 3, 2

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        out = sparse_banded_attention(q, k, v, half_bandwidth=half_bw, inner_k=inner_k)
        _scalar_loss(out).backward()

        numeric = _numeric_grad(self._loss_fn(q.data, k_np, v_np, half_bw, inner_k), q.data)
        np.testing.assert_allclose(q.grad, numeric, atol=2e-2, rtol=2e-2)

    def test_dK_matches_finite_difference(self):
        T, K, d = 8, 8, 5
        q_np, k_np, v_np = _random_qkv(T, K, d, seed=21)
        half_bw, inner_k = 3, 2

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        out = sparse_banded_attention(q, k, v, half_bandwidth=half_bw, inner_k=inner_k)
        _scalar_loss(out).backward()

        numeric = _numeric_grad(self._loss_fn(q_np, k.data, v_np, half_bw, inner_k), k.data)
        np.testing.assert_allclose(k.grad, numeric, atol=2e-2, rtol=2e-2)

    def test_dV_matches_finite_difference(self):
        T, K, d = 8, 8, 5
        q_np, k_np, v_np = _random_qkv(T, K, d, seed=22)
        half_bw, inner_k = 3, 2

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        out = sparse_banded_attention(q, k, v, half_bandwidth=half_bw, inner_k=inner_k)
        _scalar_loss(out).backward()

        numeric = _numeric_grad(self._loss_fn(q_np, k_np, v.data, half_bw, inner_k), v.data)
        np.testing.assert_allclose(v.grad, numeric, atol=2e-2, rtol=2e-2)


def _random_centers_sigmas(T, K, seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.0, K, T).astype(np.float32)
    log_sigmas = rng.uniform(-0.5, 0.5, T).astype(np.float32)  # sigma in ~[0.6, 1.6]
    return centers, log_sigmas


class TestGaussianAttentionAutograd:
    """gaussian_attention: full attention with a learnable per-query
    Gaussian log-bias -- gradient-checks centers/sigmas specifically
    (the genuinely new part; dQ/dK/dV reuse the same math already
    covered for the other three variants above), and confirms sigma is
    trained via log_sigma+exp (see sili.tensor.exp), not passed raw."""

    def _loss_fn(self, q_np, k_np, v_np, centers_np, log_sigmas_np):
        def fn():
            q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
            centers = Tensor(centers_np.copy())
            sigmas = exp(Tensor(log_sigmas_np.copy()))
            out = gaussian_attention(q, k, v, centers, sigmas)
            return float(_scalar_loss(out).data)
        return fn

    def test_dCenters_matches_finite_difference(self):
        T, K, d = 5, 6, 4
        q_np, k_np, v_np = _random_qkv(T, K, d, seed=40)
        centers_np, log_sigmas_np = _random_centers_sigmas(T, K, seed=41)

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        centers = Tensor(centers_np.copy())
        log_sigmas = Tensor(log_sigmas_np.copy())
        sigmas = exp(log_sigmas)
        out = gaussian_attention(q, k, v, centers, sigmas)
        _scalar_loss(out).backward()

        numeric = _numeric_grad(
            self._loss_fn(q_np, k_np, v_np, centers.data, log_sigmas_np), centers.data)
        np.testing.assert_allclose(centers.grad, numeric, atol=2e-2, rtol=2e-2)

    def test_dLogSigma_matches_finite_difference(self):
        # Gradient checked on log_sigma (the actual trained parameter),
        # not raw sigma -- confirms the exp() chain rule is wired
        # correctly end to end, not just gaussian_attention's own
        # dSigmas output in isolation.
        T, K, d = 5, 6, 4
        q_np, k_np, v_np = _random_qkv(T, K, d, seed=42)
        centers_np, log_sigmas_np = _random_centers_sigmas(T, K, seed=43)

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        centers = Tensor(centers_np.copy())
        log_sigmas = Tensor(log_sigmas_np.copy())
        sigmas = exp(log_sigmas)
        out = gaussian_attention(q, k, v, centers, sigmas)
        _scalar_loss(out).backward()

        numeric = _numeric_grad(
            self._loss_fn(q_np, k_np, v_np, centers_np, log_sigmas.data), log_sigmas.data)
        np.testing.assert_allclose(log_sigmas.grad, numeric, atol=2e-2, rtol=2e-2)

    def test_dQ_matches_finite_difference(self):
        T, K, d = 5, 6, 4
        q_np, k_np, v_np = _random_qkv(T, K, d, seed=44)
        centers_np, log_sigmas_np = _random_centers_sigmas(T, K, seed=45)

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        centers = Tensor(centers_np.copy())
        sigmas = exp(Tensor(log_sigmas_np.copy()))
        out = gaussian_attention(q, k, v, centers, sigmas)
        _scalar_loss(out).backward()

        numeric = _numeric_grad(
            self._loss_fn(q.data, k_np, v_np, centers_np, log_sigmas_np), q.data)
        np.testing.assert_allclose(q.grad, numeric, atol=2e-2, rtol=2e-2)

    def test_sigma_near_zero_concentrates_on_nearest_key(self):
        K, d = 5, 3
        rng = np.random.default_rng(46)
        q_np = rng.standard_normal((1, d)).astype(np.float32)
        k_np = rng.standard_normal((K, d)).astype(np.float32)
        v_np = rng.standard_normal((K, d)).astype(np.float32)
        j0 = 3

        q = Tensor(q_np); k = Tensor(k_np); v = Tensor(v_np)
        centers = Tensor(np.array([float(j0)], dtype=np.float32))
        sigmas = Tensor(np.array([1e-3], dtype=np.float32))
        out = gaussian_attention(q, k, v, centers, sigmas)

        np.testing.assert_allclose(out.data[0], v_np[j0], atol=1e-4)


class TestCausalAttentionAutograd:
    """causal=True threads through both the forward call and the backward
    closure in sili.tensor's wrappers -- gradient-check that path too, not
    just the non-causal default already covered above."""

    def test_banded_attention_causal_dQ_matches_finite_difference(self):
        T, d = 7, 5   # T == K required for causal self-attention
        q_np, k_np, v_np = _random_qkv(T, T, d, seed=30)
        half_bw = 3

        def loss_fn(q_np_, k_np_, v_np_):
            def fn():
                q = Tensor(q_np_.copy()); k = Tensor(k_np_.copy()); v = Tensor(v_np_.copy())
                out = banded_attention(q, k, v, half_bandwidth=half_bw, causal=True)
                return float(_scalar_loss(out).data)
            return fn

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        out = banded_attention(q, k, v, half_bandwidth=half_bw, causal=True)
        _scalar_loss(out).backward()

        numeric = _numeric_grad(loss_fn(q.data, k_np, v_np), q.data)
        np.testing.assert_allclose(q.grad, numeric, atol=2e-2, rtol=2e-2)

    def test_sparse_attention_causal_dQ_matches_finite_difference(self):
        T, d = 6, 5
        q_np, k_np, v_np = _random_qkv(T, T, d, seed=31)
        top_k = 4

        def loss_fn(q_np_, k_np_, v_np_):
            def fn():
                q = Tensor(q_np_.copy()); k = Tensor(k_np_.copy()); v = Tensor(v_np_.copy())
                out = sparse_attention(q, k, v, top_k=top_k, causal=True)
                return float(_scalar_loss(out).data)
            return fn

        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        out = sparse_attention(q, k, v, top_k=top_k, causal=True)
        _scalar_loss(out).backward()

        numeric = _numeric_grad(loss_fn(q.data, k_np, v_np), q.data)
        np.testing.assert_allclose(q.grad, numeric, atol=2e-2, rtol=2e-2)

    def test_causal_output_differs_from_noncausal(self):
        # Sanity check that causal=True actually changes behavior on data
        # where it should (a query early in the sequence has fewer valid
        # keys under causal masking than under full/banded attention).
        T, d = 6, 5
        q_np, k_np, v_np = _random_qkv(T, T, d, seed=32)
        q = Tensor(q_np.copy()); k = Tensor(k_np.copy()); v = Tensor(v_np.copy())
        out_causal    = banded_attention(q, k, v, half_bandwidth=T, causal=True)
        out_noncausal = banded_attention(q, k, v, half_bandwidth=T, causal=False)
        assert not np.allclose(out_causal.data[0], out_noncausal.data[0])


class TestAttentionAutogradGraphIntegration:
    """Confirm attention composes with the rest of the Tensor autograd graph
    (not just standalone) -- e.g. Q/K/V each coming from an upstream op."""

    def test_gradient_flows_through_upstream_projection(self):
        T, d = 5, 4
        rng = np.random.default_rng(99)
        x_np = rng.standard_normal((T, d)).astype(np.float32)
        w_np = rng.standard_normal((d, d)).astype(np.float32) * 0.3

        x = Tensor(x_np.copy())
        w = Tensor(w_np.copy())
        q = x @ w   # upstream projection feeding attention's Q
        out = sparse_attention(q, x, x, top_k=3)
        _scalar_loss(out).backward()

        assert w.grad is not None
        assert np.all(np.isfinite(w.grad))
        assert not np.allclose(w.grad, 0.0), "gradient failed to reach the upstream projection"
