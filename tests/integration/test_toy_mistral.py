"""
Integration test: gen_toy_mistral -> sparse_prune -> per-suffix FoldedLayer.

The fold_block_group function folds ALL suffixes in a transformer block into
one FoldedBlockDescriptor (including 1D layernorm weights). FoldedLayer.forward
sums outputs from all suffixes, which only works if all suffixes share the same
output dimension. To avoid shape mismatches, fold ONE SUFFIX AT A TIME.

Pipeline:
  1. Build toy Mistral state dict (Mistral-shaped, KB-scale, 70 layers)
  2. Prune to sparse (threshold = default_min_abs_param)
  3. For each 2D suffix (q_proj, k_proj, v_proj, o_proj, gate/up/down_proj):
     a. Collect that suffix across all 70 blocks
     b. Stack into one big CSR matrix (70*out_dim, in_dim)
     c. Create FoldedLayer -- behaves as one large sparse projection
  4. Test a forward pass through each suffix layer: shape and range checks.
  5. Test backward through one suffix layer (gradient flows to sparse W).

Run: python -m tests.integration.test_toy_mistral
"""

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'unit', 'python'))
warnings.filterwarnings('ignore')

import torch
import numpy as np
import sili.cpu
from sili.tensor import Tensor
from sili.sparse_rnn import FoldedLayer
from sili.conversion.rnn_fold import (
    FoldedBlockDescriptor, detect_repeated_block_groups,
    stack_csr_vertical, _is_attention_param,
)

from gen_toy_mistral import build_toy_mistral_state_dict, N_LAYERS, HIDDEN, N_HEADS, N_KV_HEADS, HEAD_DIM, INTERMEDIATE


def prune_state_dict(sd: dict, threshold: float) -> dict:
    """Zero weights below threshold and convert 2D tensors to CSR."""
    out = {}
    for k, v in sd.items():
        if v.ndim >= 2:
            mask = v.abs() >= threshold
            out[k] = (v * mask).to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
        else:
            out[k] = v  # keep 1D (layernorm) as dense
    return out


def fold_one_suffix(suffix: str, sd_sparse: dict, prefix: str,
                    n_layers: int) -> FoldedLayer:
    """
    Fold a single parameter suffix across all n_layers blocks into one FoldedLayer.
    E.g. suffix='.self_attn.q_proj.weight' folds all 70 q_proj matrices.
    """
    per_block = []
    for i in range(n_layers):
        name = f"{prefix}{i}{suffix}"
        if name not in sd_sparse:
            raise KeyError(f"Missing: {name}")
        t = sd_sparse[name]
        if not t.is_sparse_csr:
            t = t.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
        per_block.append(t)

    stacked = stack_csr_vertical(per_block)
    out_dim  = int(per_block[0].shape[0])

    desc = FoldedBlockDescriptor(
        n_folds          = n_layers,
        block_indices    = list(range(n_layers)),
        stacked_weights  = {suffix: stacked},
        out_dims         = {suffix: out_dim},
        band_half_widths = {suffix: None},
        prefix           = prefix,
    )
    return FoldedLayer.from_descriptor(desc, learning_rate=0.001, num_cpus=1)


def run(verbose: bool = True) -> bool:
    """Returns True if all checks pass."""
    from sili.conversion.sparse_prune import default_min_abs_param

    if verbose:
        print("\n=== test_toy_mistral: gen -> prune -> per-suffix FoldedLayer ===")

    sd, intended_zf = build_toy_mistral_state_dict(seed=1234)
    if verbose:
        total = sum(t.numel() for t in sd.values())
        print(f"  toy Mistral: {N_LAYERS} layers, {total:,} params, hidden={HIDDEN}")

    threshold = default_min_abs_param()
    sd_sparse = prune_state_dict(sd, threshold)
    if verbose:
        print(f"  pruned with threshold={threshold:.5f}")

    prefix = 'model.layers.'
    # Only 2D suffixes: each has its own out_dim
    SUFFIXES_SHAPES = {
        '.self_attn.q_proj.weight': (N_HEADS * HEAD_DIM, HIDDEN),
        '.self_attn.k_proj.weight': (N_KV_HEADS * HEAD_DIM, HIDDEN),
        '.self_attn.v_proj.weight': (N_KV_HEADS * HEAD_DIM, HIDDEN),
        '.self_attn.o_proj.weight': (HIDDEN, N_HEADS * HEAD_DIM),
        '.mlp.gate_proj.weight':    (INTERMEDIATE, HIDDEN),
        '.mlp.up_proj.weight':      (INTERMEDIATE, HIDDEN),
        '.mlp.down_proj.weight':    (HIDDEN, INTERMEDIATE),
    }

    all_pass = True
    for suffix, (out_dim_per_block, in_dim) in SUFFIXES_SHAPES.items():
        layer = fold_one_suffix(suffix, sd_sparse, prefix, N_LAYERS)
        exp_n_in  = in_dim
        exp_n_out = out_dim_per_block   # FoldedLayer sums over folds -> out_dim

        assert layer.in_features == exp_n_in, \
            f"{suffix}: in={layer.in_features}, expected {exp_n_in}"
        assert layer.out_features == exp_n_out, \
            f"{suffix}: out={layer.out_features}, expected {exp_n_out}"

        # Forward pass
        x = Tensor(np.random.randn(in_dim).astype(np.float32) * 0.1)
        h = layer(x)
        assert h.data.shape == (exp_n_out,), \
            f"{suffix}: output shape {h.data.shape}, expected ({exp_n_out},)"
        assert np.isfinite(h.data).all(), f"{suffix}: non-finite output"

        # Backward pass (gradient should reach x)
        h.grad = np.ones_like(h.data)
        h.backward()
        assert x.grad is not None, f"{suffix}: no gradient at input"
        assert np.isfinite(x.grad).all(), f"{suffix}: non-finite gradient"

        nnz_per_fold = layer.nnz_total() // N_LAYERS
        if verbose:
            print(f"  {suffix.split('.')[-2]:12s}: "
                  f"in={in_dim} out={out_dim_per_block} "
                  f"nnz/fold={nnz_per_fold}  [forward+backward OK]")

    if verbose:
        print(f"\n  All {len(SUFFIXES_SHAPES)} suffix layers: PASS" if all_pass else "  FAIL")
    return all_pass


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--quiet', action='store_true')
    ok = run(verbose=not ap.parse_args().quiet)
    sys.exit(0 if ok else 1)
