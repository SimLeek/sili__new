"""
Group B (DIDLDO<->SIDLDO, dense weight storage) python-binding tests --
TODO_BATCH_BLOCKING.md "Next arc". Covers:

- DIDLDOLayerV's four explicit calls (forward_dense/forward_sparse/
  backward_dense/backward_sparse) against independent numpy references.
- The real-time engine dispatcher (forward()/backward(), engine_select.hpp)
  giving the SAME numeric result as the explicit calls regardless of
  which branch it takes internally -- dense/sparse must be interchangeable
  results-wise, only speed differs, so this is the thing that would catch
  a wiring bug in the dispatcher silently routing to the wrong shape.
- Output-not-aliased (same class of bug test_forward_output_not_aliased.py
  found for SparseLinearLayer/DISLDOLayerV -- same risk here, since
  DIDLDOLayerV/SIDLDO also reuse persistent scratch buffers internally).
- DISLDOLayerV's retrofit: forward()/backward() must match
  forward_dense()/backward_dense() exactly (regression guard -- these
  used to BE the same method before the rename+dispatcher split).
- DIDLDOLayer8/DIDLDOLayer4 raise NotImplementedError immediately.

Only runs if the mkl/mkl-include pip extra was present when sili._cpu was
built (`pip install sili[mkl]`) -- absent that, DIDLDOLayerV doesn't
exist on _cpu at all (graceful compile-out, see setup.py's _find_mkl()),
so these are skipped rather than failed.
"""

import numpy as np
import pytest

from sili import _cpu

pytestmark = pytest.mark.skipif(
    not hasattr(_cpu, "DIDLDOLayerV"),
    reason="sili._cpu built without the mkl/mkl-include extra -- DIDLDOLayerV doesn't exist",
)


def _dense_csr(x, rng, keep_prob=0.3):
    mask = rng.uniform(0, 1, x.shape) < keep_prob
    x_masked = x * mask
    ptrs = [0]
    idx = []
    vals = []
    for row in x_masked:
        nz = np.nonzero(row)[0]
        idx.extend(nz.tolist())
        vals.extend(row[nz].tolist())
        ptrs.append(len(idx))
    return (
        x_masked,
        np.array(ptrs, dtype=np.int32),
        np.array(idx, dtype=np.int32),
        np.array(vals, dtype=np.float32),
    )


def _layer(n_in=8, n_out=5, num_cpus=4, seed=0):
    layer = _cpu.DIDLDOLayerV(n_in, n_out, num_cpus)
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n_in, n_out)).astype(np.float32)
    layer.load_dense_values(w.flatten())
    return layer, w, rng


class TestDIDLDOLayerVExplicitCalls:
    def test_forward_dense_matches_reference(self):
        layer, w, rng = _layer()
        x = rng.uniform(-1, 1, (3, 8)).astype(np.float32)
        y = layer.forward_dense(x)
        np.testing.assert_allclose(y, x @ w, atol=1e-3)

    def test_forward_sparse_matches_reference(self):
        layer, w, rng = _layer()
        x_dense = rng.uniform(-1, 1, (3, 8)).astype(np.float32)
        x_masked, ptrs, idx, vals = _dense_csr(x_dense, rng)
        y = layer.forward_sparse(ptrs, idx, vals, 3)
        np.testing.assert_allclose(y, x_masked @ w, atol=1e-3)

    def test_backward_dense_lr_zero_leaves_weights_untouched(self):
        layer, w, rng = _layer()
        x = rng.uniform(-1, 1, (3, 8)).astype(np.float32)
        dy = rng.uniform(-1, 1, (3, 5)).astype(np.float32)
        w_before = layer.weights_vals.copy()
        dx = layer.backward_dense(x, dy, 0.0)
        np.testing.assert_allclose(dx, dy @ w.T, atol=1e-3)
        np.testing.assert_array_equal(layer.weights_vals, w_before)

    def test_backward_dense_lr_nonzero_updates_weights(self):
        layer, w, rng = _layer()
        x = rng.uniform(-1, 1, (3, 8)).astype(np.float32)
        dy = rng.uniform(-1, 1, (3, 5)).astype(np.float32)
        w_before = layer.weights_vals.copy()
        layer.backward_dense(x, dy, 1e-2)
        assert not np.allclose(layer.weights_vals, w_before)

    def test_backward_sparse_matches_reference(self):
        layer, w, rng = _layer()
        x = rng.uniform(-1, 1, (3, 8)).astype(np.float32)
        dy_dense = rng.uniform(-1, 1, (3, 5)).astype(np.float32)
        dy_masked, dp, di, dv = _dense_csr(dy_dense, rng)
        dx = layer.backward_sparse(x, dp, di, dv, 3, 0.0)
        np.testing.assert_allclose(dx, dy_masked @ w.T, atol=1e-3)


class TestDIDLDOLayerVSmartDispatchMatchesExplicit:
    """forward()/backward() (engine_select.hpp dispatch) must give the
    SAME result as the explicit dense/sparse calls -- dense and sparse
    are mathematically interchangeable, so any mismatch here is a real
    wiring bug in the dispatcher, not a legitimate speed/precision
    tradeoff."""

    def test_forward_dense_input_matches_forward_dense(self):
        layer, w, rng = _layer()
        x = rng.uniform(-1, 1, (3, 8)).astype(np.float32)
        np.testing.assert_allclose(layer.forward(x), layer.forward_dense(x), atol=1e-4)

    def test_forward_csr_input_matches_forward_sparse(self):
        layer, w, rng = _layer()
        x_dense = rng.uniform(-1, 1, (3, 8)).astype(np.float32)
        _, ptrs, idx, vals = _dense_csr(x_dense, rng)
        np.testing.assert_allclose(
            layer.forward(ptrs, idx, vals, 3), layer.forward_sparse(ptrs, idx, vals, 3), atol=1e-4
        )

    def test_backward_dense_dy_matches_backward_dense(self):
        layer, w, rng = _layer()
        x = rng.uniform(-1, 1, (3, 8)).astype(np.float32)
        dy = rng.uniform(-1, 1, (3, 5)).astype(np.float32)
        layer2, _, _ = _layer()  # separate instance -- backward mutates weights
        np.testing.assert_allclose(layer.backward(x, dy, 0.0), layer2.backward_dense(x, dy, 0.0), atol=1e-4)

    def test_backward_csr_dy_matches_backward_sparse(self):
        layer, w, rng = _layer()
        layer2, _, _ = _layer()
        x = rng.uniform(-1, 1, (3, 8)).astype(np.float32)
        dy_dense = rng.uniform(-1, 1, (3, 5)).astype(np.float32)
        _, dp, di, dv = _dense_csr(dy_dense, rng)
        np.testing.assert_allclose(
            layer.backward(x, dp, di, dv, 3, 0.0),
            layer2.backward_sparse(x, dp, di, dv, 3, 0.0),
            atol=1e-4,
        )


class TestDIDLDOLayerVOutputNotAliased:
    """Same class of bug test_forward_output_not_aliased.py found for
    SparseLinearLayer/DISLDOLayerV -- DIDLDOLayerV/SIDLDO also reuse
    persistent scratch buffers internally, same risk applies."""

    def test_forward_results_independent_across_calls(self):
        layer, w, rng = _layer()
        x1 = rng.uniform(-1, 1, (2, 8)).astype(np.float32)
        x2 = rng.uniform(-1, 1, (2, 8)).astype(np.float32)
        y1 = layer.forward_dense(x1)
        y2 = layer.forward_dense(x2)
        assert not np.allclose(y1, y2)
        assert not np.shares_memory(y1, y2)


class TestDISLDOLayerVRetrofitRegression:
    """forward()/backward() were RENAMED to forward_dense()/
    backward_dense() with a new smart dispatcher taking over the old
    names -- must give bit-identical results for the dense case (no
    reason for the dispatcher path to differ numerically here)."""

    def _disldo_layer(self, n_in=6, n_out=4, seed=1):
        layer = _cpu.DISLDOLayerV(n_in, n_out, n_in * n_out, 4)
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(n_in * n_out).astype(np.float32)
        imp = np.ones(n_in * n_out, dtype=np.float32)
        layer.load_dense_values(w, imp)
        return layer, rng

    def test_forward_matches_forward_dense(self):
        layer, rng = self._disldo_layer()
        x = rng.uniform(-1, 1, (2, 6)).astype(np.float32)
        np.testing.assert_allclose(layer.forward(x), layer.forward_dense(x), atol=1e-4)

    def test_backward_matches_backward_dense_lr_zero(self):
        layer, rng = self._disldo_layer()
        layer2, _ = self._disldo_layer()
        x = rng.uniform(-1, 1, (2, 6)).astype(np.float32)
        dy = rng.uniform(-1, 1, (2, 4)).astype(np.float32)
        np.testing.assert_allclose(layer.backward(x, dy, 0.0), layer2.backward_dense(x, dy, 0.0), atol=1e-4)


class TestFp8Fp4Stubs:
    def test_didldo_layer8_not_implemented(self):
        from sili.sparse_rnn import DIDLDOLayer8

        with pytest.raises(NotImplementedError):
            DIDLDOLayer8(8, 5)

    def test_didldo_layer4_not_implemented(self):
        from sili.sparse_rnn import DIDLDOLayer4

        with pytest.raises(NotImplementedError):
            DIDLDOLayer4(8, 5)


class TestDIDLDOLayer32Autograd:
    def test_forward_backward_through_tensor_autograd(self):
        from sili.sparse_rnn import DIDLDOLayer32
        from sili.tensor import Tensor

        rng = np.random.default_rng(2)
        layer = DIDLDOLayer32(8, 5, num_cpus=4, rng=rng)
        x = Tensor(rng.uniform(-1, 1, (3, 8)).astype(np.float32))

        out = layer.forward(x, learning_rate=1e-2)
        w_before = layer.weights.copy()
        out.grad = np.ones_like(out.data)
        out._backward()

        assert x.grad is not None
        assert x.grad.shape == (3, 8)
        assert not np.allclose(layer.weights, w_before)
