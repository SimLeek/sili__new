import numpy as np

from sili.tensor import Tensor, add, mul, log, reduce_sum


class TestBroadcastingBackward:
    def _finite_diff_grad(self, f, x_data, eps=1e-3):
        g = np.zeros_like(x_data)
        it = np.nditer(x_data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            plus = x_data.copy(); plus[idx] += eps
            minus = x_data.copy(); minus[idx] -= eps
            g[idx] = (f(plus) - f(minus)) / (2 * eps)
        return g

    def test_mul_broadcast_shapes_reduce_correctly(self):
        # Regression: mul()'s backward used to accumulate the
        # UNREDUCED elementwise product directly, which only matches
        # the operand's own shape when both operands are already the
        # same shape -- broadcasting a=[3,4] against b=[3,1] (found
        # while building a batched RMSNorm for sili_peridot, scaling
        # each row by its own reciprocal-RMS) silently produced a
        # WRONG-shaped grad for b (or crashed downstream, e.g. inside
        # a following reshape's backward, depending what consumed it).
        rng = np.random.RandomState(1)
        a_data = rng.randn(3, 4).astype(np.float32)
        b_data = rng.randn(3, 1).astype(np.float32)

        a, b = Tensor(a_data.copy()), Tensor(b_data.copy())
        out = mul(a, b)
        out.grad = np.ones_like(out.data)
        out._backward()
        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (3, 1)

        expected_b_grad = self._finite_diff_grad(
            lambda bp: (a_data * bp).sum(), b_data)
        np.testing.assert_allclose(b.grad, expected_b_grad, rtol=1e-2, atol=1e-2)

    def test_add_broadcast_shapes_reduce_correctly(self):
        # Same bug, add()'s side -- a vector bias b=[hidden] added to a
        # batched a=[T,hidden] (e.g. RMSNorm's own learned weight).
        rng = np.random.RandomState(2)
        a_data = rng.randn(3, 4).astype(np.float32)
        b_data = rng.randn(4).astype(np.float32)

        a, b = Tensor(a_data.copy()), Tensor(b_data.copy())
        out = add(a, b)
        out.grad = np.ones_like(out.data)
        out._backward()
        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (4,)
        np.testing.assert_allclose(b.grad, np.full(4, 3.0, dtype=np.float32))


class TestReduceSumBackwardAxis:
    def test_backward_correct_for_every_axis_on_a_2d_tensor(self):
        # Regression: reduce_sum's backward used to just call
        # backend.broadcast_to(out.grad, a.data) directly -- correct by
        # coincidence for axis=0 (the remaining shape happens to already
        # be right-aligned with the original), wrong for any other axis
        # (e.g. axis=-1 drops a NON-trailing... well, drops the trailing
        # dim itself, but broadcasting [T] back to [T,hidden] tries to
        # align it against the wrong (hidden) dimension) -- raised a
        # shape error outright rather than silently computing something
        # wrong. Found while building a batched RMSNorm for sili_peridot.
        for axis in [None, 0, 1, -1]:
            x = Tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
            s = reduce_sum(x, axis=axis)
            s.grad = (np.ones_like(s.data) if s.data.ndim
                       else np.array(1.0, dtype=np.float32))
            s._backward()
            np.testing.assert_array_equal(
                x.grad, np.ones((3, 4), dtype=np.float32),
                err_msg=f"axis={axis!r}")


class TestLog:
    def test_forward_matches_numpy(self):
        x = np.array([0.5, 1.0, 2.0, 10.0], dtype=np.float32)
        out = log(Tensor(x))
        np.testing.assert_allclose(out.data, np.log(x), rtol=1e-6)

    def test_backward_matches_finite_difference(self):
        rng = np.random.RandomState(0)
        x_data = rng.uniform(0.1, 5.0, size=8).astype(np.float32)
        eps = 1e-3

        x = Tensor(x_data.copy())
        out = log(x)
        out.grad = np.ones_like(out.data)
        out._backward()

        for i in range(x_data.shape[0]):
            plus = x_data.copy(); plus[i] += eps
            minus = x_data.copy(); minus[i] -= eps
            numeric = (np.log(plus[i]) - np.log(minus[i])) / (2 * eps)
            assert abs(x.grad[i] - numeric) < 1e-2, (
                f"index {i}: analytic {x.grad[i]} vs finite-diff {numeric}")

    def test_log_exp_roundtrip_gradient_is_identity(self):
        # d(log(exp(x)))/dx == 1 everywhere -- a cheap end-to-end sanity
        # check that log()'s chain rule composes correctly with exp()'s.
        from sili.tensor import exp
        x = Tensor(np.array([-1.0, 0.0, 2.0], dtype=np.float32))
        e = exp(x)
        out = log(e)
        out.grad = np.ones_like(out.data)
        out._backward()
        e._backward()
        np.testing.assert_allclose(x.grad, np.ones(3, dtype=np.float32), rtol=1e-4)
