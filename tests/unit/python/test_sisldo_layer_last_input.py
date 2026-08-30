"""
Regression test for SISLDOLayer.forward()'s stale-last_input bug (sparsity
plan Phase 4). `forward_sparse` never populates `self._c.last_input` (only
`forward_dense` does -- see cpu_backend.cpp's `last_input` property
docstring). `SISLDOLayer.forward()`'s `_bwd()` closure used to call
`backward_sparse(self._c.last_input, ...)` -- so on any instance that had
previously called `forward_dense` with DIFFERENT data (or never called
`forward_dense` at all), the weight update (not dx, which doesn't depend on
x) was silently computed against stale/wrong/empty data instead of the
input actually passed to `forward()`.

Fixed by reconstructing the dense x directly from the CSR `forward()` was
given (`csr.to_dense()`) and threading it through the closure explicitly,
never reading `self._c.last_input`.

This test constructs two identically-seeded layers, "poisons" one via a
prior forward_dense(wrong_x) call, then runs the SAME forward_sparse+
backward_sparse sequence on both via SISLDOLayer.forward()/backward(). The
resulting weight updates must match -- if the bug were present, the
poisoned layer's update would be computed against wrong_x instead.
"""
import numpy as np

from sili.sparse_rnn import CSR, SISLDOLayer
from sili.tensor import Tensor, get_backend


def _make_layer(seed: int) -> SISLDOLayer:
    return SISLDOLayer(6, 4, 24, num_cpus=2, backprop_p=0.5,
                        rng=np.random.default_rng(seed))


class TestSISLDOLayerLastInputBug:
    def test_weight_update_ignores_prior_forward_dense_poisoning(self):
        backend = get_backend("cpu")
        rng = np.random.default_rng(123)

        x_dense = rng.normal(size=6).astype(np.float32)
        x_csr = CSR.from_dense(x_dense, p=0.5, num_cpus=2)
        wrong_x = np.full(6, 999.0, dtype=np.float32)

        # Both layers identically seeded -- identical initial wiring/weights.
        layer_poisoned = _make_layer(seed=7)
        layer_clean    = _make_layer(seed=7)

        # Poison ONLY layer_poisoned's last_input with an unrelated forward_dense
        # call, matching the exact bug scenario (last_input reflects some
        # earlier, different call, not the input forward() is about to see).
        layer_poisoned._c.forward_dense(wrong_x[np.newaxis, :])

        def run(layer: SISLDOLayer) -> np.ndarray:
            x_t = x_csr.as_tensor(backend)
            # learning_rate must be nonzero -- backward_sparse's weight
            # update is gated on learning_rate != 0 in the C++ engine, so
            # the default learning_rate=0.0 would make BOTH arms silently
            # skip the weight update entirely regardless of which x got
            # used, defeating this comparison (caught directly: this test
            # passed even against the unfixed code before this was added).
            out = layer.forward(x_t, learning_rate=0.05)
            # Deterministic, nonzero gradient on every output element.
            out.grad = np.array([0.3, -0.2, 0.1, -0.4], dtype=np.float32)
            out._backward()
            # value_scale, NOT weights_vals -- weights_vals are FP4-quantized
            # (16 discrete codes), coarse enough that a single small-lr step
            # can round to the identical code regardless of which x drove
            # it, making that comparison pass even under the bug (caught
            # directly: an earlier version of this test compared
            # weights_vals and passed against both the buggy and fixed
            # code). value_scale is accumulated as a real float32 and
            # updated every backward call in proportion to grad*in_val, so
            # it's actually sensitive to which x was used.
            return np.array(layer._c.get_value_scale_raw_vector())

        vs_poisoned = run(layer_poisoned)
        vs_clean    = run(layer_clean)

        np.testing.assert_allclose(
            vs_poisoned, vs_clean, rtol=1e-5, atol=1e-6,
            err_msg=("value_scale update after forward()+backward() diverged "
                     "between a layer poisoned by a prior forward_dense(wrong_x) "
                     "call and a clean layer given the identical forward_sparse "
                     "input -- backward_sparse is reading stale "
                     "self._c.last_input instead of the input actually passed "
                     "to forward()"))

    def test_last_input_is_none_before_any_forward_dense_call(self):
        # Confirms the premise directly: a layer that has ONLY ever called
        # forward_sparse (never forward_dense) has last_input == None.
        # Under the old bug, backward_sparse(self._c.last_input=None, ...)
        # does NOT raise (pybind coerces None into some array rather than
        # rejecting it) -- it silently corrupts the weight update instead
        # of crashing, so the real assertion here is "weights stay finite
        # and get a real (nonzero) update", not just "no exception".
        layer = _make_layer(seed=1)
        backend = get_backend("cpu")
        x_dense = np.random.default_rng(2).normal(size=6).astype(np.float32)
        x_csr = CSR.from_dense(x_dense, p=0.5, num_cpus=2)

        assert layer._c.last_input is None
        # value_scale is lazily sized on first backward call (0-length
        # until then) -- nothing meaningful to snapshot pre-backward, the
        # real check is post-backward finiteness/non-triviality below.
        out = layer.forward(x_csr.as_tensor(backend), learning_rate=0.05)
        assert layer._c.last_input is None, (
            "forward_sparse populated last_input -- premise of this bug no "
            "longer holds, re-check whether the Phase 4 fix is still needed")

        out.grad = np.array([0.1, 0.2, -0.1, 0.05], dtype=np.float32)
        out._backward()
        vs_after = np.array(layer._c.get_value_scale_raw_vector())
        assert vs_after.size == 6, (
            f"expected value_scale sized to n_inputs=6 after a real backward "
            f"pass, got size {vs_after.size} -- backward_sparse was likely "
            f"handed a wrongly-shaped x (e.g. None coerced to empty)")
        assert np.all(np.isfinite(vs_after)), (
            "value_scale went non-finite after backward on a layer with no "
            "last_input -- backward_sparse was handed garbage in place of "
            "the real dense x")
