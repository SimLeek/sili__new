"""Python-layer verification that _nucleus_top_k_csr reproduces the exact
R(v,k) = sum(v_topk^2)/sum(v^2) >= r_target math -- the C++ kernel already
has its own from-first-principles TDD (test_top_k_csr_nucleus.cpp); this
checks the Python wrapper (scalar/array r_target broadcasting, k_min/k_max
kwarg plumbing) doesn't silently diverge from it."""
import numpy as np
import pytest

from sili.sparse_rnn import _nucleus_top_k_csr


def _captured_ratio(row: np.ndarray, kept: np.ndarray) -> float:
    total = float(np.sum(row.astype(np.float64) ** 2))
    if total == 0.0:
        return 0.0
    return float(np.sum(kept.astype(np.float64) ** 2)) / total


class TestNucleusTopKCSRMathReproduction:
    def test_r_target_invariant_random_rows(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(12, 40)).astype(np.float32)
        r_target = 0.9
        ptrs, indices, values = _nucleus_top_k_csr(x, r_target, num_cpus=4)

        for r in range(x.shape[0]):
            row = x[r]
            kept = values[ptrs[r]:ptrs[r + 1]]
            R = _captured_ratio(row, kept)
            assert R >= r_target - 1e-6, f"row {r}: R={R} < r_target={r_target}"

    def test_per_row_r_target_array(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(6, 24)).astype(np.float32)
        r_targets = np.array([0.1, 0.99, 0.5, 0.75, 0.3, 0.6], dtype=np.float32)
        ptrs, indices, values = _nucleus_top_k_csr(x, r_targets, num_cpus=4)

        for r in range(x.shape[0]):
            row = x[r]
            kept = values[ptrs[r]:ptrs[r + 1]]
            R = _captured_ratio(row, kept)
            assert R >= r_targets[r] - 1e-6, f"row {r}: R={R} < r_target={r_targets[r]}"

    def test_dominant_entry_keeps_only_that_entry(self):
        x = np.array([[100.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        ptrs, indices, values = _nucleus_top_k_csr(x, 0.9, num_cpus=1)
        assert ptrs[1] - ptrs[0] == 1
        assert indices[0] == 0

    def test_k_min_floor_forces_padding(self):
        # r_target=0 alone would keep k=0; k_min must force a floor,
        # pulled from the magnitude-sorted order (top-2), not arbitrary.
        x = np.array([[5.0, -4.0, 1.0, 0.5]], dtype=np.float32)
        ptrs, indices, values = _nucleus_top_k_csr(x, 0.0, num_cpus=1, k_min=2)
        assert ptrs[1] - ptrs[0] == 2
        assert list(indices[ptrs[0]:ptrs[1]]) == [0, 1]

    def test_k_max_ceiling_caps_density(self):
        # r_target=1.0 alone would keep everything; k_max must cap it --
        # exactly the hardware-density-ceiling case from the design note.
        x = np.array([[5.0, -4.0, 1.0, 0.5]], dtype=np.float32)
        ptrs, indices, values = _nucleus_top_k_csr(x, 1.0, num_cpus=1, k_max=2)
        assert ptrs[1] - ptrs[0] == 2

    def test_all_zero_row_stays_empty_even_with_k_min(self):
        x = np.zeros((1, 4), dtype=np.float32)
        ptrs, indices, values = _nucleus_top_k_csr(x, 0.5, num_cpus=1, k_min=3)
        assert ptrs[1] - ptrs[0] == 0

    def test_default_k_min_max_is_noop(self):
        # Same result with and without explicitly passing the defaults --
        # confirms the (0, None->-1) defaults really are no-ops.
        rng = np.random.default_rng(3)
        x = rng.normal(size=(4, 16)).astype(np.float32)
        a = _nucleus_top_k_csr(x, 0.8, num_cpus=2)
        b = _nucleus_top_k_csr(x, 0.8, num_cpus=2, k_min=0, k_max=None)
        for arr_a, arr_b in zip(a, b):
            np.testing.assert_array_equal(arr_a, arr_b)
