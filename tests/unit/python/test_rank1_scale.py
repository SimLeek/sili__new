"""
tests/unit/python/test_rank1_scale.py
────────────────────────────────────────
fit_rank1_scale_envelope + FoldedLayer.from_descriptor(value_scale_mode=
"rank1") -- a per-output-column quantization scale alongside the existing
per-row one, for outputs whose magnitude varies a lot within one row's
fan-out. See sparse_rnn.py's fit_rank1_scale_envelope docstring.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402 -- warnings.filterwarnings above is pre-existing, predates this lint setup
import pytest  # noqa: E402

torch = pytest.importorskip("torch")

from sili.conversion.rnn_fold import FoldedBlockDescriptor  # noqa: E402
from sili.sparse_rnn import FoldedLayer, fit_rank1_scale_envelope  # noqa: E402
from sili.tensor import Tensor  # noqa: E402


def _dense_to_triplets(abs_mat: np.ndarray):
    """Every entry (including zeros) as an explicit triplet -- fine for
    small test matrices, and lets the sparse-triplet function be tested
    against the same fully-dense cases it replaces."""
    n_rows, n_cols = abs_mat.shape
    row_idx, col_idx = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    return row_idx.ravel(), col_idx.ravel(), abs_mat.ravel(), n_rows, n_cols


class TestFitRank1ScaleEnvelope:
    def test_envelope_property_holds_for_every_entry(self):
        rng = np.random.RandomState(0)
        abs_mat = np.abs(rng.randn(20, 15)) * rng.choice([0.01, 1.0, 100.0], size=(20, 15))
        row_idx, col_idx, abs_vals, n_rows, n_cols = _dense_to_triplets(abs_mat)
        row_scale, col_scale = fit_rank1_scale_envelope(row_idx, col_idx, abs_vals, n_rows, n_cols, n_iters=8)
        envelope = row_scale[:, None] * col_scale[None, :]
        # allow float32-relative slack -- the fit is exact by construction,
        # modulo fp rounding (values here range up to ~O(100)).
        assert np.all(envelope >= abs_mat - 1e-5 * np.abs(abs_mat))

    def test_recovers_a_true_rank1_matrix_exactly(self):
        rng = np.random.RandomState(1)
        true_row = np.abs(rng.randn(10)) + 0.1
        true_col = np.abs(rng.randn(8)) + 0.1
        abs_mat = np.outer(true_row, true_col)
        row_idx, col_idx, abs_vals, n_rows, n_cols = _dense_to_triplets(abs_mat)
        row_scale, col_scale = fit_rank1_scale_envelope(row_idx, col_idx, abs_vals, n_rows, n_cols, n_iters=10)
        envelope = row_scale[:, None] * col_scale[None, :]
        assert np.allclose(envelope, abs_mat, rtol=1e-3)

    def test_no_nonzero_entries_does_not_crash(self):
        row_idx = np.array([], dtype=np.int64)
        col_idx = np.array([], dtype=np.int64)
        abs_vals = np.array([], dtype=np.float32)
        row_scale, col_scale = fit_rank1_scale_envelope(row_idx, col_idx, abs_vals, 4, 3, n_iters=4)
        assert row_scale.shape == (4,)
        assert col_scale.shape == (3,)
        assert np.all(row_scale > 0)  # floored at 1e-12, never exactly 0
        assert np.all(col_scale > 0)

    def test_only_nonzero_entries_need_covering(self):
        # A sparse matrix where most entries are structurally absent (not
        # just numerically zero) -- the envelope only needs to bound the
        # entries that actually exist.
        row_idx = np.array([0, 1, 2], dtype=np.int64)
        col_idx = np.array([0, 1, 2], dtype=np.int64)
        abs_vals = np.array([3.0, 0.03, 30.0], dtype=np.float32)
        row_scale, col_scale = fit_rank1_scale_envelope(row_idx, col_idx, abs_vals, 3, 3, n_iters=8)
        envelope = row_scale[:, None] * col_scale[None, :]
        for r, c, v in zip(row_idx, col_idx, abs_vals, strict=False):
            assert envelope[r, c] >= v - 1e-6


def _make_descriptor(w: torch.Tensor, suffix: str = ".w") -> FoldedBlockDescriptor:
    """w: [n_folds*out_dim, in_dim] dense -- wraps it as a 1-fold descriptor
    (n_folds=1 is fine here, tests are about per-row/per-col scale, not
    the fold-sum itself)."""
    csr = w.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
    return FoldedBlockDescriptor(
        n_folds=1,
        block_indices=[0],
        stacked_weights={suffix: csr},
        out_dims={suffix: int(w.shape[0])},
        band_half_widths={suffix: None},
        prefix="model.",
    )


class TestFromDescriptorValueScaleMode:
    def test_invalid_mode_raises(self):
        w = torch.eye(3)
        desc = _make_descriptor(w)
        with pytest.raises(ValueError, match="value_scale_mode"):
            FoldedLayer.from_descriptor(desc, learning_rate=0.01, num_cpus=1, value_scale_mode="bogus")

    def test_per_row_mode_is_still_the_default_and_unchanged(self):
        torch.manual_seed(0)
        w = torch.randn(6, 4)
        desc = _make_descriptor(w)
        layer_default = FoldedLayer.from_descriptor(desc, learning_rate=0.01, num_cpus=1)
        layer_explicit = FoldedLayer.from_descriptor(desc, learning_rate=0.01, num_cpus=1, value_scale_mode="per_row")

        x = np.random.RandomState(2).randn(3, 4).astype(np.float32)
        out_default = np.asarray(layer_default.forward(Tensor(x)).data)
        out_explicit = np.asarray(layer_explicit.forward(Tensor(x)).data)
        assert np.allclose(out_default, out_explicit)

    def test_rank1_mode_sets_a_nontrivial_output_scale(self):
        # Two output rows with VERY different magnitude, same input columns
        # -- per-row-only scaling can't distinguish them (there's only one
        # row here), rank1 should give them genuinely different output_scale.
        w = torch.tensor([[3.0, 0.0], [0.03, 0.0]])  # row0 100x row1's magnitude
        desc = _make_descriptor(w)
        layer = FoldedLayer.from_descriptor(desc, learning_rate=0.01, num_cpus=1, value_scale_mode="rank1")
        raw = layer._sili_layers[".w"]
        out_scale_0 = raw.get_output_scale(0)
        out_scale_1 = raw.get_output_scale(1)
        assert out_scale_0 != pytest.approx(out_scale_1, rel=1e-3)

    def test_rank1_mode_recovers_lower_quantization_error_than_per_row(self):
        # The actual point: real-world per-output magnitude varies a lot
        # WITHIN one row's fan-out -- construct a small analogous case and
        # confirm rank1's reconstruction error is meaningfully lower.
        rng = np.random.RandomState(3)
        n_in, n_out = 6, 40
        # Each output column has a very different "typical" scale, shared
        # loosely across input rows (mimics one folded layer's per-output
        # magnitude spread found in the real investigation).
        col_true_scale = np.abs(rng.randn(n_out)) * rng.choice([0.05, 1.0, 20.0], size=n_out)
        w = (rng.randn(n_in, n_out) * col_true_scale[None, :]).astype(np.float32)
        desc = _make_descriptor(torch.from_numpy(w.T.copy()))  # descriptor expects [out_dim, in_dim]
        layer_per_row = FoldedLayer.from_descriptor(desc, learning_rate=0.01, num_cpus=1, value_scale_mode="per_row")
        layer_rank1 = FoldedLayer.from_descriptor(desc, learning_rate=0.01, num_cpus=1, value_scale_mode="rank1")

        x = np.eye(n_in, dtype=np.float32)  # one-hot rows -> isolates each input's own column
        ref = x @ w  # [n_in, n_out], exact unquantized reference
        out_per_row = np.asarray(layer_per_row.forward(Tensor(x)).data)
        out_rank1 = np.asarray(layer_rank1.forward(Tensor(x)).data)

        err_per_row = np.abs(out_per_row - ref).mean()
        err_rank1 = np.abs(out_rank1 - ref).mean()
        assert err_rank1 < err_per_row


class TestOutputScaleGradientTraining:
    def test_rank1_mode_output_scale_moves_under_backward(self):
        torch.manual_seed(5)
        w = torch.randn(6, 4)
        desc = _make_descriptor(w)
        layer = FoldedLayer.from_descriptor(desc, learning_rate=0.05, num_cpus=1, value_scale_mode="rank1")
        raw = layer._sili_layers[".w"]
        before = raw.get_output_scale(0)

        x = Tensor(np.random.RandomState(1).randn(3, 4).astype(np.float32))
        loss = (layer.forward(x) ** 2).sum()
        loss.backward()

        assert raw.get_output_scale(0) != pytest.approx(before, rel=1e-6)

    def test_per_row_mode_output_scale_stays_fixed_under_backward(self):
        torch.manual_seed(5)
        w = torch.randn(6, 4)
        desc = _make_descriptor(w)
        layer = FoldedLayer.from_descriptor(desc, learning_rate=0.05, num_cpus=1, value_scale_mode="per_row")
        raw = layer._sili_layers[".w"]
        assert raw.get_output_scale(0) == 1.0

        x = Tensor(np.random.RandomState(1).randn(3, 4).astype(np.float32))
        loss = (layer.forward(x) ** 2).sum()
        loss.backward()

        assert raw.get_output_scale(0) == 1.0


class TestScaleImportance:
    def test_rank1_mode_value_and_output_scale_importance_move_off_zero(self):
        # importance is sili's per-parameter optimizer state -- anything
        # gradient-updated (value_scale, now output_scale too) should have
        # its own, same as a per-synapse weight does.
        torch.manual_seed(5)
        w = torch.randn(6, 4)
        desc = _make_descriptor(w)
        layer = FoldedLayer.from_descriptor(desc, learning_rate=0.05, num_cpus=1, value_scale_mode="rank1")
        raw = layer._sili_layers[".w"]
        assert raw.get_value_scale_importance(0) == 0.0
        assert raw.get_output_scale_importance(0) == 0.0

        x = Tensor(np.random.RandomState(1).randn(3, 4).astype(np.float32))
        loss = (layer.forward(x) ** 2).sum()
        loss.backward()

        assert raw.get_value_scale_importance(0) != 0.0
        assert raw.get_output_scale_importance(0) != 0.0

    def test_per_row_mode_value_scale_importance_moves_output_scale_importance_does_not(self):
        torch.manual_seed(5)
        w = torch.randn(6, 4)
        desc = _make_descriptor(w)
        layer = FoldedLayer.from_descriptor(desc, learning_rate=0.05, num_cpus=1, value_scale_mode="per_row")
        raw = layer._sili_layers[".w"]

        x = Tensor(np.random.RandomState(1).randn(3, 4).astype(np.float32))
        loss = (layer.forward(x) ** 2).sum()
        loss.backward()

        assert raw.get_value_scale_importance(0) != 0.0  # value_scale always trainable
        assert raw.get_output_scale_importance(0) == 0.0  # output_scale untouched in this mode

    def test_forward_alone_moves_importance_before_any_backward_pre_existing_failure(self):
        # Per-synapse importance updates in forward_dense (ADSP-style
        # activity correlation), not just backward -- value_scale/
        # output_scale's own importance should too.
        torch.manual_seed(5)
        w = torch.randn(6, 4)
        desc = _make_descriptor(w)
        layer = FoldedLayer.from_descriptor(desc, learning_rate=0.05, num_cpus=1, value_scale_mode="rank1")
        raw = layer._sili_layers[".w"]
        assert raw.get_value_scale_importance(0) == 0.0
        assert raw.get_output_scale_importance(0) == 0.0

        x = Tensor(np.random.RandomState(1).randn(3, 4).astype(np.float32))
        layer.forward(x)  # forward only -- no backward() at all

        assert raw.get_value_scale_importance(0) != 0.0
        assert raw.get_output_scale_importance(0) != 0.0
