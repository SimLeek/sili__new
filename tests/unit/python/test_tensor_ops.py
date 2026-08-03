import numpy as np

from sili.tensor import Tensor, log


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
