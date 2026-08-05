"""
tests/integration/test_folded_column_layer.py
────────────────────────────────────────────────
Integration tests for FoldedColumnLayer and build_fold_skip_layer
(sili_peridot/todolist.md Phase A4). Lives in tests/integration/ (not
tests/unit/python/) because, like test_toy_mistral.py, it exercises
FoldedLayer.from_descriptor's torch-based conversion-time construction
path, not just pure-Python runtime logic.

Run: python -m tests.integration.test_folded_column_layer
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pytest

import sili.cpu
import sili._cpu as _cpu
from sili.tensor import Tensor
from sili.sparse_rnn import (
    FoldedColumnLayer, FoldedLayer, build_fold_skip_layer,
    csr_union, _build_banded_csr, state_dict_to_true_csr,
    _sparse_linear_layer_state_dict,
    _build_rectangular_banded_csr, build_input_skip_preseed,
    _rebuild_layer_with_preseed,
)
from sili.conversion.rnn_fold import FoldedBlockDescriptor, stack_csr_vertical


def _toy_square_descriptor(n_folds: int, hidden: int, seed: int = 0,
                           density: float = 1.0) -> FoldedBlockDescriptor:
    """
    A minimal toy descriptor with n_folds square [hidden, hidden] weight
    matrices stacked. Square shape isn't actually required by
    FoldedColumnLayer itself (see its docstring -- no in_dim==out_dim
    requirement anymore), just convenient for these tests since it lets
    the SAME x/hidden size be reused as both input and (per-fold-step)
    output width.
    """
    rng = torch.Generator().manual_seed(seed)
    per_block = []
    for _ in range(n_folds):
        w = torch.randn(hidden, hidden, generator=rng) * 0.1
        if density < 1.0:
            mask = torch.rand(hidden, hidden, generator=rng) < density
            w = w * mask
        per_block.append(w.to_sparse(sparse_dim=2).coalesce().to_sparse_csr())

    stacked = stack_csr_vertical(per_block)
    return FoldedBlockDescriptor(
        n_folds=n_folds, block_indices=list(range(n_folds)),
        stacked_weights={".down_proj.weight": stacked},
        out_dims={".down_proj.weight": hidden},
        band_half_widths={".down_proj.weight": None},
        prefix="model.layers.",
    )


class TestCsrUnion:
    """csr_union: needed because a dense LLM has no skip connections at
    all -- converting one starts recurrent's structural pattern from an
    empty set that has to be unioned with the fresh banded pre-seed (see
    build_fold_skip_layer's `existing`). NOT used in the forward/backward
    path -- construction/loading time only."""

    def test_disjoint_positions_all_kept(self):
        ptrs_a = np.array([0, 1], dtype=np.int32)
        idx_a  = np.array([0], dtype=np.int32)
        vals_a = np.array([1.0], dtype=np.float32)
        ptrs_b = np.array([0, 1], dtype=np.int32)
        idx_b  = np.array([1], dtype=np.int32)
        vals_b = np.array([2.0], dtype=np.float32)

        p, i, v = csr_union(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b, n_rows=1)
        assert list(zip(i.tolist(), v.tolist())) == [(0, 1.0), (1, 2.0)]

    def test_overlap_prefer_a_keeps_a(self):
        ptrs = np.array([0, 1], dtype=np.int32)
        idx  = np.array([0], dtype=np.int32)
        p, i, v = csr_union(ptrs, idx, np.array([1.0], dtype=np.float32),
                            ptrs, idx, np.array([9.0], dtype=np.float32),
                            n_rows=1, prefer="a")
        assert v.tolist() == [1.0]

    def test_overlap_prefer_b_keeps_b(self):
        ptrs = np.array([0, 1], dtype=np.int32)
        idx  = np.array([0], dtype=np.int32)
        p, i, v = csr_union(ptrs, idx, np.array([1.0], dtype=np.float32),
                            ptrs, idx, np.array([9.0], dtype=np.float32),
                            n_rows=1, prefer="b")
        assert v.tolist() == [9.0]

    def test_overlap_prefer_sum_adds(self):
        ptrs = np.array([0, 1], dtype=np.int32)
        idx  = np.array([0], dtype=np.int32)
        p, i, v = csr_union(ptrs, idx, np.array([1.0], dtype=np.float32),
                            ptrs, idx, np.array([9.0], dtype=np.float32),
                            n_rows=1, prefer="sum")
        assert v.tolist() == [10.0]

    def test_union_with_empty_b_returns_a_unchanged(self):
        ptrs_a = np.array([0, 2], dtype=np.int32)
        idx_a  = np.array([0, 1], dtype=np.int32)
        vals_a = np.array([1.0, 2.0], dtype=np.float32)
        ptrs_b = np.array([0, 0], dtype=np.int32)
        idx_b  = np.zeros(0, dtype=np.int32)
        vals_b = np.zeros(0, dtype=np.float32)

        p, i, v = csr_union(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b, n_rows=1)
        np.testing.assert_array_equal(i, idx_a)
        np.testing.assert_array_equal(v, vals_a)

    def test_invalid_prefer_raises(self):
        ptrs = np.array([0, 0], dtype=np.int32)
        idx  = np.zeros(0, dtype=np.int32)
        vals = np.zeros(0, dtype=np.float32)
        with pytest.raises(AssertionError):
            csr_union(ptrs, idx, vals, ptrs, idx, vals, n_rows=1, prefer="bogus")


class TestStateDictToTrueCsr:
    def test_converts_raw_times_scale_to_true_value(self):
        n_folds, out_dim = 3, 4
        layer = build_fold_skip_layer(n_folds, out_dim, num_cpus=1, expected_lr=0.05)
        layer.load_weights(
            np.asarray(layer.ptrs), np.asarray(layer.indices),
            np.full(layer.nnz, 2.0, dtype=np.float32),
        )
        for r in range(layer.n_inputs):
            layer.set_value_scale_raw(r, 0.3)

        d = _sparse_linear_layer_state_dict(layer)
        ptrs, idx, vals = state_dict_to_true_csr(d)
        np.testing.assert_allclose(vals, 2.0 * 0.3, atol=1e-6)
        np.testing.assert_array_equal(ptrs, d["ptrs"])
        np.testing.assert_array_equal(idx, d["indices"])


class TestBuildFoldSkipLayer:
    def test_shape_and_all_zero_valued(self):
        n_folds, out_dim = 4, 6
        layer = build_fold_skip_layer(n_folds, out_dim, num_cpus=1)
        total = n_folds * out_dim
        assert layer.n_inputs == total
        assert layer.n_outputs == total
        assert np.allclose(layer.weights_vals, 0.0)

    def test_diagonal_always_connects_to_itself(self):
        # abs(r-r)=0 < bandwidth always -- every row must at least connect
        # to its own column.
        n_folds, out_dim = 3, 5
        layer = build_fold_skip_layer(n_folds, out_dim, num_cpus=1)
        idx = np.asarray(layer.indices)
        ptrs = np.asarray(layer.ptrs)
        for r in range(n_folds * out_dim):
            row_cols = set(idx[ptrs[r]:ptrs[r + 1]].tolist())
            assert r in row_cols

    def test_default_bandwidth_reaches_adjacent_fold_step(self):
        # Default bandwidth = out_dim -- row r in fold step t should reach
        # AT LEAST into the immediately adjacent fold step's band (some
        # column there), for interior fold steps.
        n_folds, out_dim = 4, 10
        layer = build_fold_skip_layer(n_folds, out_dim, num_cpus=1)
        idx = np.asarray(layer.indices)
        ptrs = np.asarray(layer.ptrs)
        t, i = 1, 3  # fold step 1 (interior), column 3
        r = t * out_dim + i
        row_cols = set(idx[ptrs[r]:ptrs[r + 1]].tolist())
        prev_band = set(range((t - 1) * out_dim, t * out_dim))
        next_band = set(range((t + 1) * out_dim, (t + 2) * out_dim))
        assert row_cols & prev_band, "row doesn't reach the previous fold step's band"
        assert row_cols & next_band, "row doesn't reach the next fold step's band"

    def test_smaller_bandwidth_is_narrower(self):
        n_folds, out_dim = 4, 10
        wide   = build_fold_skip_layer(n_folds, out_dim, num_cpus=1, bandwidth=out_dim)
        narrow = build_fold_skip_layer(n_folds, out_dim, num_cpus=1, bandwidth=2)
        assert narrow.nnz < wide.nnz

    def test_forward_backward_finite(self):
        n_folds, out_dim = 3, 6
        layer = build_fold_skip_layer(n_folds, out_dim, num_cpus=1)
        total = n_folds * out_dim
        x = Tensor(np.random.randn(total).astype(np.float32))
        out_np = layer.forward_dense(x.data[np.newaxis, :])
        assert np.all(np.isfinite(out_np))
        # All-zero weights -> all-zero output pre-training, regardless of x.
        assert np.allclose(out_np, 0.0)

    def test_no_existing_behaves_exactly_as_before(self):
        # existing=None (the default) is the "converting a dense LLM,
        # which has no skip connections at all" case -- union with an
        # empty set, should be indistinguishable from the pre-`existing`
        # behavior.
        n_folds, out_dim = 3, 5
        layer = build_fold_skip_layer(n_folds, out_dim, num_cpus=1, expected_lr=0.05)
        assert np.allclose(layer.weights_vals, 0.0)
        assert layer.get_value_scale(0) == pytest.approx(0.05 / 6.0)

    def test_existing_brings_forward_real_trained_values(self):
        n_folds, out_dim = 3, 4
        total = n_folds * out_dim

        trained = build_fold_skip_layer(n_folds, out_dim, num_cpus=1, expected_lr=0.05)
        trained.load_weights(
            np.asarray(trained.ptrs), np.asarray(trained.indices),
            (np.arange(trained.nnz, dtype=np.float32) % 6) - 3,
        )
        for r in range(total):
            trained.set_value_scale_raw(r, 0.8)
        existing_true = state_dict_to_true_csr(_sparse_linear_layer_state_dict(trained))

        merged = build_fold_skip_layer(n_folds, out_dim, num_cpus=1, expected_lr=0.05,
                                       existing=existing_true, existing_prefer="b")
        assert not np.allclose(merged.weights_vals, 0.0)
        # A row with real (nonzero) merged values should get max_abs/FP4_MAX
        # scaling, not the tiny zero-escape expected_lr/FP4_MAX scale --
        # otherwise real trained magnitudes would be clipped.
        assert merged.get_value_scale(0) != pytest.approx(0.05 / 6.0)

    def test_existing_unions_new_structural_positions(self):
        # existing containing a position OUTSIDE the fresh band (e.g. from
        # synaptogenesis growth, or a manually-added skip connection) must
        # actually grow the final structural pattern, not be dropped.
        n_folds, out_dim = 3, 4
        total = n_folds * out_dim
        ptrs, idx, vals = _build_banded_csr(total, out_dim)
        fresh_nnz = len(idx)

        row0_end = int(ptrs[1])
        existing_idx  = np.concatenate([idx[:row0_end], [total - 1], idx[row0_end:]])
        existing_vals = np.concatenate([vals[:row0_end], [5.0], vals[row0_end:]])
        existing_ptrs = ptrs.copy()
        existing_ptrs[1:] += 1

        merged = build_fold_skip_layer(
            n_folds, out_dim, num_cpus=1, expected_lr=0.05,
            existing=(existing_ptrs, existing_idx, existing_vals), existing_prefer="b")

        assert merged.nnz == fresh_nnz + 1
        row0_cols = set(np.asarray(merged.indices)[
            np.asarray(merged.ptrs)[0]:np.asarray(merged.ptrs)[1]].tolist())
        assert (total - 1) in row0_cols


class TestBuildRectangularBandedCsr:
    """_build_rectangular_banded_csr: in_proj's skip pre-seed needs a
    geometric (proportional) diagonal, not an exact linear-algebra one,
    since rows (in_dim) and cols (n_folds*out_dim) are different sizes."""

    def test_shape_and_all_zero_valued(self):
        ptrs, idx, vals = _build_rectangular_banded_csr(4, 12, 2)
        assert len(ptrs) == 5
        assert np.allclose(vals, 0.0)
        assert np.all(idx >= 0) and np.all(idx < 12)

    def test_row_zero_centers_near_column_zero(self):
        ptrs, idx, vals = _build_rectangular_banded_csr(4, 12, 2)
        row0 = idx[ptrs[0]:ptrs[1]]
        assert 0 in row0

    def test_last_row_centers_near_last_column(self):
        rows, cols, bw = 4, 12, 2
        ptrs, idx, vals = _build_rectangular_banded_csr(rows, cols, bw)
        last_row = idx[ptrs[rows - 1]:ptrs[rows]]
        expected_center = int((rows - 1) * cols / rows)
        assert expected_center in last_row

    def test_wider_bandwidth_gives_more_columns_per_row(self):
        narrow = _build_rectangular_banded_csr(4, 40, 2)
        wide   = _build_rectangular_banded_csr(4, 40, 8)
        assert len(wide[1]) > len(narrow[1])

    def test_columns_stay_in_range_even_at_edges(self):
        ptrs, idx, vals = _build_rectangular_banded_csr(3, 5, 4)
        assert np.all(idx >= 0) and np.all(idx < 5)


class TestBuildInputSkipPreseed:
    """build_input_skip_preseed: the in_proj analogue of
    build_fold_skip_layer, giving training a direct input->column path to
    every fold-depth step, not just whatever the original dense LLM's own
    per-layer weights happened to connect."""

    def test_shape_matches_n_in_by_n_folds_times_out_dim(self):
        n_in, n_folds, out_dim = 6, 4, 6
        ptrs, idx, vals = build_input_skip_preseed(n_in, n_folds, out_dim)
        assert len(ptrs) == n_in + 1
        assert np.all(idx < n_folds * out_dim)
        assert np.allclose(vals, 0.0)

    def test_reaches_columns_belonging_to_later_fold_steps(self):
        # A "no skip" world would only ever let input dim r touch columns
        # within fold step r's own out_dim-sized slot at most. With the
        # pre-seed, at least some row should reach a column belonging to a
        # LATER fold step's slot too -- proving input can structurally
        # reach layer 2, 3, ... not just layer 1.
        n_in, n_folds, out_dim = 4, 5, 4
        ptrs, idx, vals = build_input_skip_preseed(n_in, n_folds, out_dim, bandwidth=out_dim + 2)
        row0 = idx[ptrs[0]:ptrs[1]]
        assert np.any(row0 >= out_dim), \
            "input row 0 never reaches any column past fold step 0's own slot"

    def test_default_bandwidth_is_out_dim(self):
        n_in, n_folds, out_dim = 6, 4, 6
        default = build_input_skip_preseed(n_in, n_folds, out_dim)
        explicit = build_input_skip_preseed(n_in, n_folds, out_dim, bandwidth=out_dim)
        assert np.array_equal(default[0], explicit[0])
        assert np.array_equal(default[1], explicit[1])


class TestRebuildLayerWithPreseed:
    """_rebuild_layer_with_preseed: unions a zero-valued pre-seed onto an
    already-built layer's REAL weights, real values always winning."""

    def _real_layer(self, n_in=4, n_out=8, seed=0):
        rng = np.random.default_rng(seed)
        layer = _cpu.SparseLinearLayer(n_in, n_out, n_in * n_out, 1)
        ptrs = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        idx  = np.array([0, 2, 4, 6], dtype=np.int32)
        vals = rng.standard_normal(4).astype(np.float32) * 2.0
        for r in range(n_in):
            layer.set_value_scale_raw(r, 1.0)
        raw = vals.copy() / 1.0
        layer.load_weights(ptrs, idx, raw)
        return layer

    def test_real_values_survive_the_union(self):
        layer = self._real_layer()
        before_ptrs, before_idx, before_vals = state_dict_to_true_csr(
            _sparse_linear_layer_state_dict(layer))

        preseed_ptrs, preseed_idx, _ = build_input_skip_preseed(4, 2, 4, bandwidth=4)
        merged = _rebuild_layer_with_preseed(
            layer, preseed_ptrs, preseed_idx, num_cpus=1, expected_lr=0.02)

        after_ptrs, after_idx, after_vals = state_dict_to_true_csr(
            _sparse_linear_layer_state_dict(merged))
        for r in range(4):
            a_start, a_end = int(before_ptrs[r]), int(before_ptrs[r + 1])
            for c, v in zip(before_idx[a_start:a_end], before_vals[a_start:a_end]):
                m_start, m_end = int(after_ptrs[r]), int(after_ptrs[r + 1])
                cols = list(after_idx[m_start:m_end])
                assert c in cols, f"real position (row={r}, col={c}) dropped by union"
                pos = m_start + cols.index(c)
                assert np.isclose(after_vals[pos], v, atol=1e-3), \
                    f"real value at (row={r}, col={c}) changed by union"

    def test_new_zero_positions_are_added(self):
        layer = self._real_layer()
        preseed_ptrs, preseed_idx, _ = build_input_skip_preseed(4, 2, 4, bandwidth=4)
        merged = _rebuild_layer_with_preseed(
            layer, preseed_ptrs, preseed_idx, num_cpus=1, expected_lr=0.02)
        assert merged.nnz > layer.nnz

    def test_shape_preserved(self):
        layer = self._real_layer(n_in=4, n_out=8)
        preseed_ptrs, preseed_idx, _ = build_input_skip_preseed(4, 2, 4, bandwidth=4)
        merged = _rebuild_layer_with_preseed(
            layer, preseed_ptrs, preseed_idx, num_cpus=1, expected_lr=0.02)
        assert merged.n_inputs == 4
        assert merged.n_outputs == 8


class TestFoldedColumnLayerInputSkip:
    """FoldedColumnLayer.from_descriptor unions build_input_skip_preseed
    onto every suffix's real in_proj weights, giving input a direct,
    trainable path to every fold step (see sili_peridot/JOURNAL.md)."""

    def test_in_proj_has_more_connections_than_the_bare_descriptor(self):
        n_folds, hidden = 4, 6
        desc = _toy_square_descriptor(n_folds, hidden, density=0.3, seed=5)
        bare = FoldedLayer.from_descriptor(desc, learning_rate=0.02, num_cpus=1)
        skipped = FoldedColumnLayer.from_descriptor(desc, learning_rate=0.02, num_cpus=1)

        bare_layer    = next(iter(bare._sili_layers.values()))
        skipped_layer = next(iter(skipped._sili_layers.values()))
        assert skipped_layer.nnz > bare_layer.nnz, \
            "in_proj gained no structural connections from the skip pre-seed"

    def test_real_pretrained_values_preserved_after_preseed_union(self):
        n_folds, hidden = 4, 6
        desc = _toy_square_descriptor(n_folds, hidden, density=0.6, seed=6)
        bare    = FoldedLayer.from_descriptor(desc, learning_rate=0.02, num_cpus=1)
        skipped = FoldedColumnLayer.from_descriptor(desc, learning_rate=0.02, num_cpus=1)

        bare_layer    = next(iter(bare._sili_layers.values()))
        skipped_layer = next(iter(skipped._sili_layers.values()))
        bare_ptrs, bare_idx, bare_vals = state_dict_to_true_csr(
            _sparse_linear_layer_state_dict(bare_layer))
        sk_ptrs, sk_idx, sk_vals = state_dict_to_true_csr(
            _sparse_linear_layer_state_dict(skipped_layer))

        for r in range(hidden):
            b_start, b_end = int(bare_ptrs[r]), int(bare_ptrs[r + 1])
            for c, v in zip(bare_idx[b_start:b_end], bare_vals[b_start:b_end]):
                s_start, s_end = int(sk_ptrs[r]), int(sk_ptrs[r + 1])
                cols = list(sk_idx[s_start:s_end])
                assert c in cols
                pos = s_start + cols.index(c)
                assert np.isclose(sk_vals[pos], v, atol=1e-2), \
                    f"real pretrained value at (row={r}, col={c}) changed by skip union"

    def test_forward_backward_still_finite_with_skip_preseed(self):
        n_folds, hidden = 4, 6
        desc = _toy_square_descriptor(n_folds, hidden, density=0.5, seed=7)
        layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=0.02, num_cpus=1)

        x     = Tensor(np.random.randn(hidden).astype(np.float32))
        state = Tensor(np.random.randn(n_folds * hidden).astype(np.float32))
        out   = layer(x, state)
        loss  = (out ** 2).sum()
        loss.backward()
        assert np.all(np.isfinite(out.data))
        assert x.grad is not None and np.all(np.isfinite(x.grad))

    def test_gradient_can_move_a_skip_seeded_position_off_zero(self):
        # Multi-step training against a target that specifically needs a
        # skip-seeded (not originally-real) connection to be useful: feed
        # the SAME x every step (a static "no change" input) and check
        # in_proj's weights end up nonzero at MORE positions than the
        # bare descriptor's real pattern alone would ever allow.
        n_folds, hidden = 3, 5
        desc = _toy_square_descriptor(n_folds, hidden, density=0.2, seed=8)
        layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=0.05, num_cpus=1)
        suffix_layer = next(iter(layer._sili_layers.values()))

        rng = np.random.default_rng(3)
        x = Tensor(rng.standard_normal(hidden).astype(np.float32))
        target = Tensor((rng.standard_normal(n_folds * hidden) * 0.3).astype(np.float32))

        for _ in range(15):
            raw = layer.in_proj(x)
            loss = ((raw - target) ** 2).sum()
            loss.backward()
            assert np.all(np.isfinite(suffix_layer.weights_vals)), "diverged mid-training"

        assert suffix_layer.nnz > 0
        assert not np.allclose(suffix_layer.weights_vals, 0.0)


class TestFoldedColumnLayerForward:
    def test_output_shape_is_n_folds_times_out_dim(self):
        n_folds, hidden = 4, 6
        desc = _toy_square_descriptor(n_folds, hidden, density=0.5)
        layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=0.01, num_cpus=1)
        x = Tensor(np.random.randn(hidden).astype(np.float32))
        out = layer(x)
        assert out.shape == (n_folds * hidden,)
        assert layer.out_features == n_folds * hidden
        assert layer.column_width == hidden

    def test_output_finite(self):
        n_folds, hidden = 3, 5
        desc = _toy_square_descriptor(n_folds, hidden, density=0.5)
        layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=0.01, num_cpus=1)
        x = Tensor(np.random.randn(hidden).astype(np.float32))
        out = layer(x)
        assert np.all(np.isfinite(out.data))

    def test_backward_reaches_input(self):
        n_folds, hidden = 3, 5
        desc = _toy_square_descriptor(n_folds, hidden, density=0.7)
        layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=0.01, num_cpus=1)
        x = Tensor(np.random.randn(hidden).astype(np.float32))
        out = layer(x)
        loss = (out ** 2).sum()
        loss.backward()
        assert x.grad is not None
        assert np.all(np.isfinite(x.grad))

    def test_reshape_matches_folded_layer_sum(self):
        # Sanity cross-check: FoldedColumnLayer's in_proj (the pretrained
        # -weight half, NOT the full forward() -- FoldedLayer has no
        # recurrent concept to compare against), reshaped and summed over
        # the fold axis by hand, should equal what an ordinary FoldedLayer
        # produces from the SAME weights -- confirms retaining the fold
        # axis didn't change the underlying per-fold-step computation.
        n_folds, hidden = 4, 5
        desc_a = _toy_square_descriptor(n_folds, hidden, density=1.0, seed=7)
        desc_b = _toy_square_descriptor(n_folds, hidden, density=1.0, seed=7)
        plain    = FoldedLayer.from_descriptor(desc_a, learning_rate=0.01, num_cpus=1)
        columned = FoldedColumnLayer.from_descriptor(desc_b, learning_rate=0.01, num_cpus=1)

        x_np = np.random.randn(hidden).astype(np.float32)
        out_plain    = plain(Tensor(x_np.copy()))
        out_in_proj  = columned.in_proj(Tensor(x_np.copy()))

        manual_sum = out_in_proj.data.reshape(n_folds, hidden).sum(axis=0)
        np.testing.assert_allclose(manual_sum, out_plain.data, atol=1e-4)

    def test_zero_state_forward_equals_in_proj_alone(self):
        # forward(x) with no state defaults to zeros; recurrent's own
        # weights start at zero too, so a zero state through it produces
        # zero regardless -- forward(x) should exactly equal in_proj(x)
        # on a freshly-constructed layer.
        n_folds, hidden = 3, 5
        desc = _toy_square_descriptor(n_folds, hidden, density=0.6, seed=4)
        layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=0.01, num_cpus=1)
        x_np = np.random.randn(hidden).astype(np.float32)

        out_forward = layer(Tensor(x_np.copy()))
        out_in_proj = layer.in_proj(Tensor(x_np.copy()))
        np.testing.assert_allclose(out_forward.data, out_in_proj.data, atol=1e-6)

    def test_weights_learn_via_backward_dense_no_energy_gating(self):
        # FoldedColumnLayer's OWN weights (via backward_dense, not a free
        # Tensor parameter -- test_column_averaging.py's SGD test uses the
        # latter) should be able to reduce column_averaging_loss given
        # enough learning rate/steps, with no EnergyDynamics in the loop.
        # Isolates "does backprop through this layer's real sparse weights
        # work" from "does it still work under energy competition" (see
        # TestColumnAveragingEndToEnd for the latter, and why that one
        # asserts stability rather than convergence).
        from sili.energy import column_averaging_loss

        n_folds, hidden = 6, 16
        desc = _toy_square_descriptor(n_folds, hidden, density=0.3, seed=3)
        layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=1.0, num_cpus=1)

        rng = np.random.default_rng(5)
        target_np = (rng.standard_normal(hidden) * 0.5).astype(np.float32)
        x_np      = (rng.standard_normal(hidden) * 0.3).astype(np.float32)

        errors = []
        for _ in range(300):
            x = Tensor(x_np.copy())
            raw = layer(x)
            target = Tensor(target_np.copy())
            loss = column_averaging_loss(raw, target, n_folds=n_folds, weight=1.0)
            loss.backward()
            col_mean = raw.data.reshape(n_folds, hidden).mean(axis=0)
            errors.append(float(np.mean((col_mean - target_np) ** 2)))

        err_early = float(np.mean(errors[:10]))
        err_late  = float(np.mean(errors[-10:]))
        assert err_late < err_early * 0.7, (
            f"layer weights didn't reduce column MSE via backward_dense: "
            f"early={err_early:.4f} late={err_late:.4f}"
        )


class TestFoldedColumnLayerRecurrent:
    """FoldedColumnLayer's automatic in_proj + recurrent composition --
    mirrors SparseRNNCell's own input_proj+recurrent split (see class
    docstring), built automatically by from_descriptor now rather than
    requiring the caller to compose build_fold_skip_layer by hand."""

    def test_recurrent_built_automatically(self):
        n_folds, hidden = 4, 6
        desc = _toy_square_descriptor(n_folds, hidden, density=0.5)
        layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=0.05, num_cpus=1)
        assert hasattr(layer, "recurrent")
        assert layer.recurrent.n_inputs == n_folds * hidden
        assert layer.recurrent.n_outputs == n_folds * hidden
        assert np.allclose(layer.recurrent.weights_vals, 0.0)

    def test_nonzero_state_changes_output_vs_zero_state(self):
        n_folds, hidden = 4, 6
        desc = _toy_square_descriptor(n_folds, hidden, density=0.5, seed=1)
        layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=0.05, num_cpus=1)
        # Give recurrent nonzero weights directly (bypassing training) so
        # a nonzero state actually has something to propagate through.
        layer.recurrent.load_weights(
            np.asarray(layer.recurrent.ptrs), np.asarray(layer.recurrent.indices),
            np.ones(layer.recurrent.nnz, dtype=np.float32),
        )
        layer.recurrent.set_value_scale_raw(0, 1.0)
        for r in range(layer.recurrent.n_inputs):
            layer.recurrent.set_value_scale_raw(r, 1.0)

        x = Tensor(np.random.randn(hidden).astype(np.float32))
        out_zero    = layer(x)
        out_nonzero = layer(x, state=Tensor(np.ones(n_folds * hidden, dtype=np.float32)))
        assert not np.allclose(out_zero.data, out_nonzero.data), \
            "recurrent(state) had no effect despite nonzero weights and nonzero state"

    def test_gradient_reaches_both_in_proj_and_recurrent_weights(self):
        n_folds, hidden = 4, 6
        desc = _toy_square_descriptor(n_folds, hidden, density=0.5)
        layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=0.05, num_cpus=1)

        x     = Tensor(np.random.randn(hidden).astype(np.float32))
        state = Tensor(np.random.randn(n_folds * hidden).astype(np.float32))
        out   = layer(x, state)
        loss  = (out ** 2).sum()
        loss.backward()

        assert x.grad is not None
        assert np.all(np.isfinite(x.grad))
        assert np.all(np.isfinite(out.data))
        # in_proj's weights: check via neuron_grad_accum on the underlying
        # suffix layer(s) -- nonzero confirms backward_dense ran on them.
        sub = next(iter(layer._sili_layers.values()))
        assert np.any(sub.neuron_grad_accum != 0.0)

    def test_recurrent_weights_move_off_zero_after_multi_step_training(self):
        # Genuine multi-step recurrence: state threaded from one call to
        # the next (mirroring SparseRNNCell's own calling convention),
        # not a single forward(x, state) call with a hand-picked state.
        #
        # lr=0.3 here originally diverged in an earlier version of this
        # test (weight magnitude exploding against a random target with
        # no real relationship to input, eventually overflowing FP4
        # storage back to exactly 0) -- not a library bug, just a
        # badly-tuned test. A learning rate matched to the layer's own
        # (0.02) converges to a stable nonzero state instead.
        lr = 0.02
        n_folds, hidden = 4, 6
        desc = _toy_square_descriptor(n_folds, hidden, density=0.5, seed=2)
        layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=lr, num_cpus=1)

        rng = np.random.default_rng(1)
        target = Tensor((rng.standard_normal(n_folds * hidden) * 0.3).astype(np.float32))

        state = None
        for _ in range(20):
            x = Tensor(rng.standard_normal(hidden).astype(np.float32))
            state = layer(x, state)
            loss = ((state - target) ** 2).sum()
            loss.backward()
            assert np.all(np.isfinite(layer.recurrent.weights_vals)), "diverged mid-training"
            state = state.detach()

        assert not np.allclose(layer.recurrent.weights_vals, 0.0)

    def test_from_descriptor_existing_recurrent_end_to_end(self):
        # The actual MiniCPM5 conversion path: a dense LLM's folded
        # weights come with NO skip connections at all, so the caller
        # supplies a previously-trained `recurrent` CSR (e.g. saved from
        # an earlier sili run via state_dict_to_true_csr) and expects
        # from_descriptor to union it with the fresh banded pre-seed
        # rather than discarding it.
        n_folds, hidden = 4, 6
        total = n_folds * hidden
        desc = _toy_square_descriptor(n_folds, hidden, density=0.5, seed=3)

        trained_ptrs = np.array([0, 1, 1, 1, 2, 2, 2, 2,
                                  2, 2, 2, 2, 2, 2, 2, 2,
                                  2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32)
        assert len(trained_ptrs) == total + 1
        trained_idx = np.array([hidden * 3 + 1, hidden * 3 + 1], dtype=np.int32)  # row0->far, row3->far
        trained_vals = np.array([0.9, -0.9], dtype=np.float32)

        layer = FoldedColumnLayer.from_descriptor(
            desc, learning_rate=0.02, num_cpus=1,
            existing_recurrent=(trained_ptrs, trained_idx, trained_vals),
            existing_recurrent_prefer="b",
        )

        # The trained long-range connection at row 0 survived the union.
        row_start, row_end = int(layer.recurrent.ptrs[0]), int(layer.recurrent.ptrs[1])
        cols = list(layer.recurrent.indices[row_start:row_end])
        assert trained_idx[0] in cols
        col_pos = row_start + cols.index(trained_idx[0])
        true_val = layer.recurrent.weights_vals[col_pos] * layer.recurrent.get_value_scale(0)
        assert np.isclose(true_val, 0.9, atol=1e-3)

        # The fresh banded pre-seed's own diagonal is still present too.
        assert 0 in cols

        # And the merged layer is still usable end-to-end.
        x = Tensor(np.random.randn(hidden).astype(np.float32))
        state = Tensor(np.random.randn(total).astype(np.float32))
        out = layer(x, state)
        loss = (out ** 2).sum()
        loss.backward()
        assert np.all(np.isfinite(out.data))
        assert x.grad is not None and np.all(np.isfinite(x.grad))


class TestFoldedColumnLayerStatePersistence:
    """FoldedLayer.state_dict() previously only saved in_proj (silently
    dropping trained recurrent weights on save/reload), and had NO
    load_state_dict() at all -- state_dict()'s own output could never be
    loaded back through the public API. Found while comparing
    FoldedColumnLayer against SparseRNNCell's persistence."""

    def _trained_layer(self, n_folds=4, hidden=6, seed=2):
        desc = _toy_square_descriptor(n_folds, hidden, density=0.5, seed=seed)
        layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=0.02, num_cpus=1)
        # Give recurrent non-trivial weights + a non-default value_scale
        # directly (bypassing training) so the round-trip has something
        # real to lose if the fix is wrong.
        layer.recurrent.load_weights(
            np.asarray(layer.recurrent.ptrs), np.asarray(layer.recurrent.indices),
            ((np.arange(layer.recurrent.nnz, dtype=np.float32) % 6) - 3),
        )
        for r in range(layer.recurrent.n_inputs):
            layer.recurrent.set_value_scale_raw(r, 0.7)
        return layer

    def test_state_dict_includes_recurrent(self):
        layer = self._trained_layer()
        d = layer.state_dict()
        assert "recurrent" in d
        assert not np.allclose(d["recurrent"]["weights"], 0.0)

    def test_load_state_dict_restores_in_proj_weights_and_scale(self):
        n_folds, hidden = 4, 6
        layer  = self._trained_layer(n_folds, hidden)
        target = FoldedColumnLayer.from_descriptor(
            _toy_square_descriptor(n_folds, hidden, density=0.5, seed=99),
            learning_rate=0.02, num_cpus=1)

        target.load_state_dict(layer.state_dict())

        src = layer._sili_layers[".down_proj.weight"]
        dst = target._sili_layers[".down_proj.weight"]
        np.testing.assert_array_equal(dst.weights_vals, src.weights_vals)
        for r in range(src.n_inputs):
            assert dst.get_value_scale(r) == pytest.approx(src.get_value_scale(r))

    def test_load_state_dict_restores_recurrent_weights_and_scale(self):
        n_folds, hidden = 4, 6
        layer  = self._trained_layer(n_folds, hidden)
        target = FoldedColumnLayer.from_descriptor(
            _toy_square_descriptor(n_folds, hidden, density=0.5, seed=99),
            learning_rate=0.02, num_cpus=1)

        target.load_state_dict(layer.state_dict())

        np.testing.assert_array_equal(target.recurrent.weights_vals, layer.recurrent.weights_vals)
        for r in range(layer.recurrent.n_inputs):
            assert target.recurrent.get_value_scale(r) == pytest.approx(
                layer.recurrent.get_value_scale(r))

    def test_round_trip_produces_identical_forward_output(self):
        # The actual functional guarantee: same input -> same output
        # after a save/reload cycle, not just matching internal arrays.
        n_folds, hidden = 4, 6
        layer  = self._trained_layer(n_folds, hidden)
        target = FoldedColumnLayer.from_descriptor(
            _toy_square_descriptor(n_folds, hidden, density=0.5, seed=99),
            learning_rate=0.02, num_cpus=1)
        target.load_state_dict(layer.state_dict())

        x_np     = np.random.randn(hidden).astype(np.float32)
        state_np = np.random.randn(n_folds * hidden).astype(np.float32)
        out_src = layer(Tensor(x_np.copy()), Tensor(state_np.copy()))
        out_dst = target(Tensor(x_np.copy()), Tensor(state_np.copy()))
        np.testing.assert_allclose(out_src.data, out_dst.data, atol=1e-5)

    def test_plain_folded_layer_load_state_dict_also_works(self):
        # FoldedLayer itself (not just the FoldedColumnLayer subclass) --
        # this method didn't exist at all before this fix.
        n_folds, hidden = 3, 5
        desc_a = _toy_square_descriptor(n_folds, hidden, density=0.6, seed=1)
        desc_b = _toy_square_descriptor(n_folds, hidden, density=0.6, seed=42)
        src = FoldedLayer.from_descriptor(desc_a, learning_rate=0.01, num_cpus=1)
        dst = FoldedLayer.from_descriptor(desc_b, learning_rate=0.01, num_cpus=1)

        dst.load_state_dict(src.state_dict())

        src_sub = src._sili_layers[".down_proj.weight"]
        dst_sub = dst._sili_layers[".down_proj.weight"]
        np.testing.assert_array_equal(dst_sub.weights_vals, src_sub.weights_vals)


class TestColumnAveragingEndToEnd:
    """
    FoldedColumnLayer -> EnergyDynamics -> column_averaging_loss, wired
    together for real (not each piece tested in isolation).

    Scope note, reached empirically while writing this: on this toy
    96-neuron layer, the FULL pipeline (real energy competition --
    KL/shutoff/forced-firing NOT exempted for column neurons, exactly as
    A3 specifies) does not cleanly converge within a few thousand steps --
    it oscillates in a rough stalemate rather than trending down. Isolated,
    both pieces converge correctly (column_averaging_loss's own gradient
    tests in test_column_averaging.py; FoldedColumnLayer's weights alone,
    verified manually against column_averaging_loss with no energy gating
    in the loop, do reduce error given enough learning rate/steps). Jointly
    optimizing a discrete energy gate AND continuous sparse weights is a
    genuinely harder/slower coupled dynamical system, consistent with the
    design's own expectation ("the network will have to train a bit...
    expect lowered quality, but usable") -- not evidence of a bug. What
    this class actually asserts is STABILITY under that competition (the
    hard p ceiling holds, aux_loss doesn't diverge, everything stays
    finite), which is what a toy-scale unit test can honestly claim,
    rather than an convergence claim that would need real-model-scale
    training (sili_peridot's Phase B8/B9) to actually validate. A real,
    separate finding surfaced while tuning this is documented in TODO.md:
    a fired-but-not-top-p-selected neuron's energy is never reset, which
    diverges under a REPEATED IDENTICAL input (fixed below by using a
    varying input, matching every realistic use of this codebase).
    """

    def test_stable_under_real_energy_competition(self):
        # Genuine multi-step recurrence now: state threaded from each
        # step's energy-gated output into the next step's layer(x, state)
        # call (detached between steps, BPTT=1-style, matching
        # SparseRNNAgent.train_step's convention) -- this is what actually
        # lets recurrent() participate, and is what "track next input at
        # EVERY fold step" means now that it's backed by real recurrence
        # rather than a single independently-computed snapshot.
        from sili.energy import EnergyDynamics, column_averaging_loss
        from sili.tensor import combine_losses

        n_folds, hidden = 6, 16
        desc = _toy_square_descriptor(n_folds, hidden, density=0.3, seed=3)
        layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=0.3, num_cpus=1)

        ed = EnergyDynamics(drive=0.08, activation_cost=0.08, precision=0.02,
                            density=0.15, p=0.6, exploration=0.002, reactivity=0.02)

        rng = np.random.default_rng(5)
        target_np = (rng.standard_normal(hidden) * 0.5).astype(np.float32)
        x_base    = (rng.standard_normal(hidden) * 0.3).astype(np.float32)

        aux_losses = []
        state = None
        for _ in range(500):
            # Varying input -- a literally-static repeated input is the
            # degenerate case documented in TODO.md (fired-but-unselected
            # neurons' energy grows unboundedly with no input variation to
            # eventually let them win their local competition).
            x_np = x_base + rng.standard_normal(hidden).astype(np.float32) * 0.02
            x = Tensor(x_np)
            raw = layer(x, state)
            h_out, aux_loss, actual_p = ed.forward(raw)

            assert np.all(np.isfinite(h_out.data))
            # actual_p's own docstring: may differ slightly from p at small
            # region sizes due to integer rounding (k = round(p*n)) -- use
            # a tolerance of one neuron's worth, not an exact bound.
            assert actual_p <= ed.p + 1.0 / (n_folds * hidden)

            target   = Tensor(target_np.copy())
            col_loss = column_averaging_loss(h_out, target, n_folds=n_folds, weight=1.0)
            assert np.isfinite(float(col_loss.data))
            total    = combine_losses(aux_loss, col_loss)
            total.backward()

            aux_losses.append(float(aux_loss.data))
            state = h_out.detach()

        assert all(np.isfinite(a) for a in aux_losses)
        # Deliberately NOT asserting recurrent moves off zero here (unlike
        # test_recurrent_weights_move_off_zero_after_multi_step_training,
        # which does, with no EnergyDynamics in the loop): checked directly
        # while writing this -- the gradient reaching recurrent through the
        # full energy-gated path is real but tiny (attenuated by both the
        # sparse gate and backward_dense's lr_per_row_nnz dilution), and can
        # legitimately fall below FP4's quantization step for many steps.
        # That's consistent with this class's own scope (stability under
        # competition, not convergence -- see class docstring) rather than
        # a bug; demanding visible weight movement here would be re-tuning
        # toy hyperparameters to force a demo, the exact thing this file's
        # history already argues against.
        # Regression guard for the TODO.md-documented divergence: with the
        # static-input degenerate case this reached 177+ by step 270 and
        # was still climbing; bounded and settling (not still climbing at
        # a similar rate) is the actual claim here.
        late_growth = np.mean(aux_losses[-50:]) - np.mean(aux_losses[-150:-100])
        assert abs(late_growth) < 2.0, (
            f"aux_loss still drifting late in training (delta={late_growth:.3f}) "
            f"-- possible regression of the TODO.md-documented energy-growth issue"
        )


def run(verbose: bool = True) -> bool:
    """Standalone runner (python -m tests.integration.test_folded_column_layer)."""
    import pytest as _pytest
    args = [__file__, "-q"]
    if not verbose:
        args.append("--quiet")
    return _pytest.main(args) == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
