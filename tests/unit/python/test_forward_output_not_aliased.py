"""
Regression test for a real, reproducible bug found while building
sili_peridot's toy-scale tile-recurrence training validation:
SparseLinearLayer.forward_dense/forward_sparse and DISLDOLayerV.forward
returned a numpy array that was a VIEW into the layer's own reused
internal `output_buf` member, not a copy. Any caller holding a
PREVIOUS call's returned array alive across a SUBSEQUENT call to the
SAME layer object (e.g. comparing before/after a training step, or
ordinary Python code keeping intermediate results around) silently saw
that array's contents change to the NEW call's result -- non-obvious,
easy to hit, and produces WRONG (not crashing) results.
"""

import numpy as np

from sili import _cpu


def _tiny_layer(num_cpus=2):
    layer = _cpu.SparseLinearLayer(4, 4, 16, num_cpus)
    ptrs = np.array([0, 2, 4, 6, 8], dtype=np.int32)
    indices = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32)
    values = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    layer.load_weights(ptrs, indices, values)
    return layer


class TestForwardDenseOutputNotAliased:
    def test_previous_result_survives_a_second_call(self):
        layer = _tiny_layer()
        x_zero = np.zeros((1, 4), dtype=np.float32)
        x1 = np.random.RandomState(10).randn(1, 4).astype(np.float32)

        o_zero = layer.forward_dense(x_zero)
        np.testing.assert_array_equal(o_zero, np.zeros((1, 4), dtype=np.float32))

        o1 = layer.forward_dense(x1)

        # o_zero must NOT have changed after the second call.
        np.testing.assert_array_equal(o_zero, np.zeros((1, 4), dtype=np.float32))
        assert not np.shares_memory(o_zero, o1)

    def test_two_different_inputs_give_different_outputs(self):
        # The bug's most direct symptom: two genuinely different inputs
        # produced IDENTICAL output because the second call's result
        # silently overwrote the first's returned array.
        layer = _tiny_layer()
        x1 = np.random.RandomState(1).randn(1, 4).astype(np.float32)
        x2 = np.random.RandomState(2).randn(1, 4).astype(np.float32)
        o1 = layer.forward_dense(x1)
        o2 = layer.forward_dense(x2)
        assert not np.allclose(o1, o2)

    def test_forward_sparse_output_not_aliased(self):
        layer = _tiny_layer()
        ptrs = np.array([0, 2], dtype=np.int32)
        idx1 = np.array([0, 1], dtype=np.int32)
        vals1 = np.array([1.0, 2.0], dtype=np.float32)
        idx2 = np.array([2, 3], dtype=np.int32)
        vals2 = np.array([3.0, 4.0], dtype=np.float32)

        o1 = layer.forward_sparse(ptrs, idx1, vals1, 1).copy()
        o2 = layer.forward_sparse(ptrs, idx2, vals2, 1)
        assert not np.allclose(o1, o2)
        assert not np.shares_memory(o1, o2)
