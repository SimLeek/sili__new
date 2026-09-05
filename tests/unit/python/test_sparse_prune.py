"""
tests/unit/python/test_sparse_prune.py
────────────────────────────────────────
calibrate_min_abs_param: exact (small-population) behavior plus the
sampling path added for memory reasons (see its own docstring's "Memory"
section) -- materializing every eligible tensor's |weight| as float32
simultaneously peaked at ~15GB RAM for a single call on a real
~1B-parameter checkpoint (MiniCPM5-1B), observed while starting the
sili_peridot conversion. Sampling keeps only one tensor's full float
copy alive at a time instead of all of them concatenated.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pytest

torch = pytest.importorskip("torch")

from sili.conversion.sparse_prune import calibrate_min_abs_param  # noqa: E402


class TestCalibrateMinAbsParamExact:
    """Population at or below max_sample -- sampling never kicks in, so
    these pin down the exact (original) behavior."""

    def test_returns_zero_when_no_2d_tensors(self):
        sd = {"norm.weight": torch.ones(8), "bias": torch.zeros(4)}
        assert calibrate_min_abs_param(sd) == 0.0

    def test_ignores_1d_and_higher_rank_tensors(self):
        big_2d = torch.tensor([[100.0, 100.0], [100.0, 100.0]])
        sd = {
            "matrix": big_2d,
            "vector": torch.tensor([0.001, 0.002, 0.003]),  # would pull threshold way down if counted
            "conv": torch.ones(2, 2, 2) * 0.0001,  # ndim==3, also excluded
        }
        # All entries in `matrix` are 100.0, so any target_sparsity in (0,1)
        # must return exactly 100.0 if the 1-D/3-D tensors are truly ignored.
        assert calibrate_min_abs_param(sd, target_sparsity=0.5) == 100.0

    def test_known_distribution_gives_exact_percentile(self):
        # 0..99, so the 50th-percentile (k=50) value is exactly 49.0.
        t = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        thr = calibrate_min_abs_param({"w": t}, target_sparsity=0.5)
        assert thr == pytest.approx(49.0)

    def test_max_sample_none_matches_default_on_small_population(self):
        t = torch.randn(50, 50)
        exact = calibrate_min_abs_param({"w": t}, target_sparsity=0.6, max_sample=None)
        default = calibrate_min_abs_param({"w": t}, target_sparsity=0.6)
        assert exact == pytest.approx(default)


class TestCalibrateMinAbsParamSampling:
    """Population deliberately larger than a small max_sample, to force
    the sampling path without needing a slow multi-million-element test."""

    def _big_uniform_state_dict(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        # 500 x 400 = 200,000 elements, uniform in [0, 1) -- large enough
        # that a small max_sample forces real subsampling, small enough
        # to keep the test fast.
        return {"w": torch.rand(500, 400, generator=g)}

    def test_sampled_threshold_close_to_exact(self):
        sd = self._big_uniform_state_dict()
        exact = calibrate_min_abs_param(sd, target_sparsity=0.5, max_sample=None)
        sampled = calibrate_min_abs_param(sd, target_sparsity=0.5, max_sample=2000, seed=1)
        # Uniform[0,1) median is 0.5 -- sampling 2000 of 200,000 values
        # should land close; loose absolute tolerance since this is a
        # random estimate, not required to be exact.
        assert sampled == pytest.approx(exact, abs=0.05)

    def test_same_seed_is_deterministic(self):
        sd = self._big_uniform_state_dict()
        a = calibrate_min_abs_param(sd, target_sparsity=0.5, max_sample=1000, seed=7)
        b = calibrate_min_abs_param(sd, target_sparsity=0.5, max_sample=1000, seed=7)
        assert a == b

    def test_max_sample_none_disables_sampling_for_large_population_too(self):
        sd = self._big_uniform_state_dict()
        exact_a = calibrate_min_abs_param(sd, target_sparsity=0.5, max_sample=None)
        exact_b = calibrate_min_abs_param(sd, target_sparsity=0.5, max_sample=None)
        assert exact_a == exact_b  # no RNG involved -- must be bit-identical

    def test_tiny_tensor_still_gets_at_least_one_sample(self):
        # `small` is such a tiny fraction of the combined population that
        # a naive proportional sample size would round to 0 -- the
        # max(1, ...) floor in the sampling code must still include it.
        # Verify via target_sparsity=1.0 (the sampled maximum): small's
        # range [0, 1000) is 1000x wider than large's [0, 1), so if even
        # one sample from `small` survived, the combined max is
        # overwhelmingly likely (999/1000 chance per draw) to come from
        # it rather than from `large`.
        g = torch.Generator().manual_seed(2)
        small = torch.rand(10, 10, generator=g) * 1000.0  # 100 elements, huge values
        large = torch.rand(400, 400, generator=g) * 1.0  # 160,000 elements, small values
        sd = {"small": small, "large": large}
        sampled_max = calibrate_min_abs_param(sd, target_sparsity=1.0, max_sample=5000, seed=3)
        assert sampled_max > 5.0  # large's own max is bounded by 1.0
