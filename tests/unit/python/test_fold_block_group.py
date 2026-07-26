"""
tests/unit/python/test_fold_block_group.py
─────────────────────────────────────────────
fold_block_group's dict-entry handling: a sparse_prune.py payload's
"raw" key is used for ANY dense-stored tensor (not only true
scalars/vectors -- _keep_dense_reason falls back to it for any 2-D
matrix that stayed dense), and used to be silently treated as
"non-stackable, skip" -- worse, fold_sparse_payload's removal step
deletes a block's keys by (prefix, index) alone, so a skipped suffix's
real weights were deleted from the payload with no trace at all, not
just left unfolded. Found converting a real checkpoint whose per-role
pruning thresholds deliberately leave most 2-D suffixes dense.

Fix: a "raw" dict entry is now treated identically to a plain
(non-dict) dense tensor -- same reshape rules, same stacking, no
special-casing -- for consistent behavior regardless of whether the
caller wrapped the tensor in a dict.

Note: fold_sparse_payload's own top-level entry point does its OWN
separate flat-tensor unwrapping before calling fold_block_group, so it
never actually exercises fold_block_group's dict-handling branch at
all -- these dict-entry tests call fold_block_group directly (the
pattern sili_peridot's per-suffix folding needs, since
fold_sparse_payload bundles every suffix sharing a block range into one
descriptor, which is wrong when suffixes have different out_dim).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import pytest

from sili.conversion.rnn_fold import fold_block_group, fold_sparse_payload, FoldedBlockDescriptor, SiliBlock


def _block_group_state_dict(n_layers=3, hidden=4, second_suffix="raw"):
    """Two suffixes per layer under a shared "model.layers." prefix:
    q_proj always CSR, down_proj varies by `second_suffix`
    ("raw" dense 2-D, "csr", "scalar", or "empty")."""
    torch.manual_seed(0)
    sd = {}
    for i in range(n_layers):
        w_sparse = torch.randn(hidden, hidden)
        mask = torch.rand(hidden, hidden) < 0.5
        csr = (w_sparse * mask).to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = {"csr": csr, "shape": (hidden, hidden)}

        if second_suffix == "empty":
            sd[f"model.layers.{i}.empty_param"] = {"raw": torch.zeros(0), "shape": (0,)}
        elif second_suffix == "raw":
            w_dense = torch.randn(hidden, hidden)
            sd[f"model.layers.{i}.mlp.down_proj.weight"] = {"raw": w_dense, "shape": (hidden, hidden)}
        elif second_suffix == "csr":
            w_sparse2 = torch.randn(hidden, hidden)
            csr2 = (w_sparse2 * mask).to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
            sd[f"model.layers.{i}.mlp.down_proj.weight"] = {"csr": csr2, "shape": (hidden, hidden)}
        elif second_suffix == "scalar":
            sd[f"model.layers.{i}.some_scale"] = {"raw": torch.tensor(1.0), "shape": ()}
        else:
            raise ValueError(second_suffix)
    return sd


class TestFoldBlockGroupRawEntries:
    def test_raw_dense_2d_suffix_gets_folded_not_skipped(self):
        sd = _block_group_state_dict(n_layers=3, second_suffix="raw")
        desc = fold_block_group(list(range(3)), sd, "model.layers.")
        assert ".mlp.down_proj.weight" in desc.stacked_weights
        assert desc.out_dims[".mlp.down_proj.weight"] == 4

    def test_raw_and_csr_suffix_produce_the_same_stacking(self):
        # Same underlying data, one stored as "raw", one as "csr" -- the
        # folded result must be numerically identical either way.
        torch.manual_seed(1)
        hidden, n_layers = 4, 3
        weights = [torch.randn(hidden, hidden) for _ in range(n_layers)]

        sd_raw = {}
        sd_csr = {}
        for i, w in enumerate(weights):
            sd_raw[f"model.layers.{i}.mlp.down_proj.weight"] = {"raw": w.clone(), "shape": (hidden, hidden)}
            csr = w.clone().to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
            sd_csr[f"model.layers.{i}.mlp.down_proj.weight"] = {"csr": csr, "shape": (hidden, hidden)}

        desc_raw = fold_block_group(list(range(n_layers)), sd_raw, "model.layers.")
        desc_csr = fold_block_group(list(range(n_layers)), sd_csr, "model.layers.")

        dense_raw = desc_raw.stacked_weights[".mlp.down_proj.weight"].to_dense()
        dense_csr = desc_csr.stacked_weights[".mlp.down_proj.weight"].to_dense()
        assert torch.allclose(dense_raw, dense_csr)

    def test_no_nnz_lost_for_raw_suffix(self):
        sd = _block_group_state_dict(n_layers=3, second_suffix="raw")
        desc = fold_block_group(list(range(3)), sd, "model.layers.")
        stacked = desc.stacked_weights[".mlp.down_proj.weight"]
        original_nnz = sum(
            int((sd[f"model.layers.{i}.mlp.down_proj.weight"]["raw"] != 0).sum())
            for i in range(3)
        )
        assert int(stacked.values().numel()) == original_nnz

    def test_true_scalar_stacks_like_a_bare_tensor_scalar_would(self):
        sd = _block_group_state_dict(n_layers=3, second_suffix="scalar")
        desc = fold_block_group(list(range(3)), sd, "model.layers.")
        assert ".some_scale" in desc.stacked_weights
        assert desc.stacked_weights[".some_scale"].shape == (3, 1)

    def test_empty_tensor_stacks_into_a_degenerate_csr_without_crashing(self):
        # Matches the plain-dense-tensor branch's own (pre-existing,
        # unchanged) behavior -- neither path special-cases "empty" into
        # a skip, both just stack whatever they're given.
        sd = _block_group_state_dict(n_layers=3, second_suffix="empty")
        desc = fold_block_group(list(range(3)), sd, "model.layers.")
        assert ".empty_param" in desc.stacked_weights
        assert desc.stacked_weights[".empty_param"].shape == (3, 0)

    def test_dict_entry_with_neither_key_raises(self):
        sd = _block_group_state_dict(n_layers=2, second_suffix="csr")
        sd["model.layers.0.mlp.down_proj.weight"] = {"shape": (4, 4)}   # no csr, no raw
        with pytest.raises(ValueError, match="neither 'csr' nor 'raw'"):
            fold_block_group([0, 1], sd, "model.layers.")


class TestFoldSparsePayloadRemovalMatchesFoldedSuffixes:
    def test_raw_suffix_is_removed_after_successful_folding(self):
        sd = _block_group_state_dict(n_layers=3, second_suffix="raw")
        payload = {"sparse_state_dict": sd, "min_abs_param": 0.0, "meta": {}}
        out = fold_sparse_payload(payload)
        remaining = [k for k in out["sparse_state_dict"] if "layers." in k]
        assert remaining == []   # both suffixes folded -- nothing left over


class TestSiliBlockNoDensifyTranspose:
    """SiliBlock.__init__ had the same to_dense().t() bug FoldedLayer.
    from_descriptor did (see sparse_rnn.py) -- fixed the same way
    (csr.t().to_sparse_csr(), never densifying the whole stacked
    matrix). No prior test coverage existed for SiliBlock at all."""

    def _descriptor(self, n_folds=3, out_dim=5, in_dim=4, seed=0):
        torch.manual_seed(seed)
        w = torch.randn(n_folds * out_dim, in_dim)
        w = w * (torch.rand(n_folds * out_dim, in_dim) < 0.6)
        csr = w.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
        desc = FoldedBlockDescriptor(
            n_folds=n_folds, block_indices=list(range(n_folds)),
            stacked_weights={".w": csr}, out_dims={".w": out_dim},
            band_half_widths={".w": None}, prefix="model.",
        )
        return desc, w

    def test_constructs_with_correct_shapes(self):
        desc, w = self._descriptor()
        block = SiliBlock(desc, learning_rate=0.01, num_cpus=1)
        layer = block._layers[".w"]
        assert layer.n_inputs == w.shape[1]
        assert layer.n_outputs == w.shape[0]

    def test_nnz_matches_source_matrix(self):
        desc, w = self._descriptor()
        block = SiliBlock(desc, learning_rate=0.01, num_cpus=1)
        layer = block._layers[".w"]
        assert layer.nnz == int((w != 0).sum())
