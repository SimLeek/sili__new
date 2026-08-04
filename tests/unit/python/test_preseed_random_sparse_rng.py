"""
Regression test for a real reproducibility gap found while building
sili_peridot's toy-scale credit-assignment verification: `sparse_rnn.py`'s
`_preseed_random_sparse` called `np.random.default_rng()` unconditionally --
fresh OS entropy every call, completely independent of `np.random.seed()`
(which only controls the legacy global RandomState, not the Generator API).
Every DISLDOLayer/SISLDOLayer's initial sparse wiring has therefore never
actually been reproducible via any "seed" passed by a caller, in this or
any prior test/experiment. Fixed by accepting an optional `rng` (an
`np.random.Generator`) that overrides the default -- direct request: keep
true randomness as the DEFAULT (not seeded-by-default), only make it
OVERRIDABLE for tests that need reproducibility.
"""
import numpy as np

from sili.sparse_rnn import DISLDOLayer, SISLDOLayer, _preseed_random_sparse
from sili import _cpu


class TestPreseedRandomSparseRng:
    def test_default_is_non_reproducible_across_calls(self):
        # No rng passed -- two calls must NOT produce identical wiring
        # (this is the pre-existing, intentional "closer to true RNG"
        # behavior -- confirming it still holds after the fix, not just
        # that the override path works).
        c1 = _cpu.SparseLinearLayer(6, 4, 24, 2)
        c2 = _cpu.SparseLinearLayer(6, 4, 24, 2)
        _preseed_random_sparse(c1, 6, 4, 24)
        _preseed_random_sparse(c2, 6, 4, 24)
        v1 = np.asarray(c1.weights_vals)
        v2 = np.asarray(c2.weights_vals)
        assert not np.allclose(v1, v2), (
            "two unseeded preseed calls produced identical wiring -- "
            "either a real coincidence or the default stopped using fresh entropy")

    def test_explicit_rng_gives_reproducible_wiring(self):
        c1 = _cpu.SparseLinearLayer(6, 4, 24, 2)
        c2 = _cpu.SparseLinearLayer(6, 4, 24, 2)
        _preseed_random_sparse(c1, 6, 4, 24, rng=np.random.default_rng(42))
        _preseed_random_sparse(c2, 6, 4, 24, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(np.asarray(c1.indices), np.asarray(c2.indices))
        np.testing.assert_allclose(np.asarray(c1.weights_vals), np.asarray(c2.weights_vals))

    def test_different_seeds_give_different_wiring(self):
        c1 = _cpu.SparseLinearLayer(6, 4, 24, 2)
        c2 = _cpu.SparseLinearLayer(6, 4, 24, 2)
        _preseed_random_sparse(c1, 6, 4, 24, rng=np.random.default_rng(1))
        _preseed_random_sparse(c2, 6, 4, 24, rng=np.random.default_rng(2))
        assert not np.allclose(np.asarray(c1.weights_vals), np.asarray(c2.weights_vals))

    def test_disldo_layer_accepts_and_uses_rng(self):
        a = DISLDOLayer(6, 4, 24, num_cpus=2, rng=np.random.default_rng(7))
        b = DISLDOLayer(6, 4, 24, num_cpus=2, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(np.asarray(a._c.indices), np.asarray(b._c.indices))
        np.testing.assert_allclose(np.asarray(a._c.weights_vals), np.asarray(b._c.weights_vals))

    def test_sisldo_layer_accepts_and_uses_rng(self):
        a = SISLDOLayer(6, 4, 24, num_cpus=2, rng=np.random.default_rng(7))
        b = SISLDOLayer(6, 4, 24, num_cpus=2, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(np.asarray(a._c.indices), np.asarray(b._c.indices))
        np.testing.assert_allclose(np.asarray(a._c.weights_vals), np.asarray(b._c.weights_vals))

    def test_disldo_layer_default_still_unseeded(self):
        a = DISLDOLayer(6, 4, 24, num_cpus=2)
        b = DISLDOLayer(6, 4, 24, num_cpus=2)
        assert not np.allclose(np.asarray(a._c.weights_vals), np.asarray(b._c.weights_vals))
