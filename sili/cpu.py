from __future__ import annotations
import numpy as np
from sili.backend import Backend, register_backend


class CpuBackend(Backend):
    name = "cpu"

    @property
    def mod(self):
        from sili import _cpu
        return _cpu

    def move(self, buf, src: Backend):
        if src == self:
            return buf
        return self.mod.from_host(src.mod.to_host(buf))

    def from_numpy(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float32)

    def dense_to_csr(self, x: np.ndarray, threshold: float = 1e-4) -> tuple:
        return self.mod.dense_to_csr(np.asarray(x, dtype=np.float32), float(threshold))

    # ── shape / dtype — must handle CSR data too ──────────────────────────────

    def shape(self, a):
        from sili.sparse_rnn import CSR
        if isinstance(a, CSR):
            return (a.rows, a.cols)
        return tuple(np.asarray(a).shape)

    def dtype(self, a):
        from sili.sparse_rnn import CSR
        if isinstance(a, CSR):
            return np.float32
        return np.asarray(a).dtype

    # ── allocation ────────────────────────────────────────────────────────────

    def scalar(self, val) -> np.ndarray:
        return np.array(val, dtype=np.float32)

    def zeros_like(self, a) -> np.ndarray:
        from sili.sparse_rnn import CSR
        if isinstance(a, CSR):
            return np.zeros((a.rows, a.cols), dtype=np.float32)
        return np.zeros_like(np.asarray(a), dtype=np.float32)

    def ones_like(self, a) -> np.ndarray:
        from sili.sparse_rnn import CSR
        if isinstance(a, CSR):
            return np.ones((a.rows, a.cols), dtype=np.float32)
        return np.ones_like(np.asarray(a), dtype=np.float32)

    # ── elementwise ───────────────────────────────────────────────────────────

    def add(self, a, b) -> np.ndarray:
        return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)

    def mul(self, a, b) -> np.ndarray:
        return np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)

    def neg(self, a) -> np.ndarray:
        return -np.asarray(a, dtype=np.float32)

    def pow(self, a, exp: float) -> np.ndarray:
        return np.asarray(a, dtype=np.float32) ** exp

    def pow_backward(self, a, exp: float, grad) -> np.ndarray:
        return (exp * np.asarray(a, dtype=np.float32) ** (exp - 1)
                * np.asarray(grad, dtype=np.float32))

    def relu(self, a) -> np.ndarray:
        x = np.asarray(a, dtype=np.float32)
        return np.maximum(x, 0.0)

    def relu_backward(self, a, grad) -> np.ndarray:
        return np.asarray(grad, dtype=np.float32) * (np.asarray(a, dtype=np.float32) > 0)

    def silu(self, a) -> np.ndarray:
        x = np.asarray(a, dtype=np.float32)
        return x / (1.0 + np.exp(-x))

    def silu_backward(self, a, grad) -> np.ndarray:
        x   = np.asarray(a, dtype=np.float32)
        sig = 1.0 / (1.0 + np.exp(-x))
        return np.asarray(grad, dtype=np.float32) * sig * (1.0 + x * (1.0 - sig))

    def tensor_abs(self, a) -> np.ndarray:
        return np.abs(np.asarray(a, dtype=np.float32))

    def abs_backward(self, a, grad) -> np.ndarray:
        local_gradient = np.sign(np.asarray(a, dtype=np.float32))
        # The fix for synaptogenesis from zero:
        # np.sign(0.0) returns 0.0, which severs the gradient flow.
        # We assign a valid subgradient (e.g., 1.0 or a small positive value)
        # to force the optimizer to push the weights and "wake up" the neuron.
        local_gradient[a == 0.0] = 1.0

        return local_gradient * grad

    # ── linear algebra ────────────────────────────────────────────────────────

    def matmul(self, a, b) -> np.ndarray:
        return np.matmul(np.asarray(a, dtype=np.float32),
                         np.asarray(b, dtype=np.float32))

    def matmul_backward(self, a, b, grad):
        a_  = np.asarray(a,    dtype=np.float32)
        b_  = np.asarray(b,    dtype=np.float32)
        g_  = np.asarray(grad, dtype=np.float32)
        return g_ @ b_.T, a_.T @ g_

    # ── reduction / broadcast ─────────────────────────────────────────────────

    def sum(self, a, axis=None) -> np.ndarray:
        return np.sum(np.asarray(a, dtype=np.float32), axis=axis)

    def sum_axis0(self, a) -> np.ndarray:
        return np.sum(np.asarray(a, dtype=np.float32), axis=0)

    def broadcast_add(self, a, b) -> np.ndarray:
        return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)

    def broadcast_to(self, grad, target) -> np.ndarray:
        """Expand scalar/reduced grad back to target's shape."""
        from sili.sparse_rnn import CSR
        shape = (target.rows, target.cols) if isinstance(target, CSR) \
            else np.asarray(target).shape
        g = np.asarray(grad, dtype=np.float32)
        return np.broadcast_to(g, shape).copy()

    # ── norm ──────────────────────────────────────────────────────────────────

    def rmsnorm(self, x, w, eps: float) -> np.ndarray:
        x_ = np.asarray(x, dtype=np.float32)
        w_ = np.asarray(w, dtype=np.float32)
        rms = np.sqrt(np.mean(x_ ** 2, axis=-1, keepdims=True) + eps)
        return (x_ / rms) * w_

    def rmsnorm_backward(self, x, w, eps: float, grad):
        x_   = np.asarray(x,    dtype=np.float32)
        w_   = np.asarray(w,    dtype=np.float32)
        g_   = np.asarray(grad, dtype=np.float32)
        rms  = np.sqrt(np.mean(x_ ** 2, axis=-1, keepdims=True) + eps)
        xn   = x_ / rms
        dw   = np.sum(g_ * xn, axis=tuple(range(g_.ndim - 1))) if g_.ndim > 1 else g_ * xn
        dxn  = g_ * w_
        dx   = (dxn - xn * np.mean(dxn * xn, axis=-1, keepdims=True)) / rms
        return dx, dw


register_backend(CpuBackend())