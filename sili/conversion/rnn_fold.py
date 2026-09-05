#!/usr/bin/env python3
"""
rnn_fold.py -- fold N consecutive identical transformer blocks in a
(sparse-pruned) model into a single recurrent block whose weights are one
stacked sparse CSR matrix. See docs/research/rnn_fold.rst:
rnn_fold.module_overview for the conceptual transform, weight-layout
diagram, sparse_prune.py integration, and full CLI usage.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    # Package-qualified import ONLY -- see docs/research/rnn_fold.rst:
    # rnn_fold.package_qualified_import_bug for the pybind11 double-
    # registration crash a bare `import _cpu` used to cause.
    from sili import _cpu as _sili_cpu  # SparseLinearLayer, hoyer_score, dense_to_csr

    _SILI_AVAILABLE = True
except ImportError:
    _sili_cpu = None
    _SILI_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  Block detection
# ══════════════════════════════════════════════════════════════════════════════

# Matches any integer block / layer index embedded in a parameter name.
# Captures: (prefix_before_index, block_index_str, suffix_after_index)
# Example: "model.layers.3.self_attn.q_proj.weight"
#          → ("model.layers.", "3", ".self_attn.q_proj.weight")
_BLOCK_RE = re.compile(r"^(.*?\.)(\d+)(\..+)$")


def _parse_block_key(name: str) -> tuple[str, int, str] | None:
    """
    Split a parameter name into (prefix, block_index, suffix).

    Only the *first* integer segment that looks like a block index (i.e., the
    one closest to the root of the name) is used.  Weights like
    "transformer.h.2.mlp.fc1.weight" → ("transformer.h.", 2, ".mlp.fc1.weight").

    Returns None if no integer segment is found.
    """
    m = _BLOCK_RE.match(name)
    if m is None:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def detect_repeated_block_groups(
    state_dict: dict[str, torch.Tensor],
    min_group_size: int = 2,
) -> list[tuple[str, list[int]]]:
    """
    Find runs of consecutive blocks that share identical parameter structure.

    Two blocks are structurally identical if every parameter suffix maps to
    the same tensor shape in both, AND they share the same prefix.

    Parameters
    ----------
    state_dict      : flat {name: tensor} dict (dense or sparse payload)
    min_group_size  : minimum consecutive identical blocks to constitute a group

    Returns
    -------
    List of (prefix, indices) pairs, e.g. blocks 0-23 under "model.layers."
    all identical -> [("model.layers.", [0,1,...,23])]. Non-repeated or
    structurally inconsistent blocks are excluded. Keyed by (prefix, index),
    NOT bare index -- see docs/research/rnn_fold.rst:
    detect_repeated_block_groups.prefix_index_keying_bug for why (a VLM's
    language/vision towers sharing an index range will silently merge under
    bare-index keying).
    """
    # Gather {(prefix, block_index): {suffix: shape}}
    block_params: dict[tuple[str, int], dict[str, torch.Size]] = defaultdict(dict)
    for name, tensor in state_dict.items():
        parsed = _parse_block_key(name)
        if parsed is None:
            continue
        prefix, idx, suffix = parsed
        shape = tensor.shape if isinstance(tensor, torch.Tensor) else None
        if shape is not None:
            block_params[(prefix, idx)][suffix] = shape

    if not block_params:
        return []

    # Process each prefix's index range independently -- two families must
    # never merge into one group even if their index ranges overlap.
    prefixes = sorted({p for p, _ in block_params})
    groups: list[tuple[str, list[int]]] = []

    for prefix in prefixes:
        indices = sorted(i for p, i in block_params if p == prefix)
        current_group: list[int] = [indices[0]]

        for prev_i, cur_i in itertools.pairwise(indices):
            consecutive = cur_i == prev_i + 1
            same_shape = block_params[(prefix, prev_i)] == block_params[(prefix, cur_i)]
            if consecutive and same_shape:
                current_group.append(cur_i)
            else:
                if len(current_group) >= min_group_size:
                    groups.append((prefix, current_group))
                current_group = [cur_i]

        if len(current_group) >= min_group_size:
            groups.append((prefix, current_group))

    return groups


def report_block_groups(
    state_dict: dict[str, torch.Tensor],
    groups: list[tuple[str, list[int]]],
) -> None:
    """Print a human-readable summary of detected fold groups."""
    if not groups:
        print("[rnn_fold]  No repeated block groups detected.")
        return

    print(f"[rnn_fold]  Detected {len(groups)} foldable group(s):\n")
    for g_idx, (prefix, group) in enumerate(groups):
        n = len(group)
        sample_idx = group[0]
        # Match prefix AND index -- see detect_repeated_block_groups.prefix_index_keying_bug.
        suffixes = sorted(
            suffix
            for name in state_dict
            if (parsed := _parse_block_key(name)) is not None and parsed[0] == prefix and parsed[1] == sample_idx
            for suffix in [parsed[2]]
        )
        # Find Q/K/V projections
        attn_params = [
            s
            for s in suffixes
            if any(t in s for t in ("q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn", "in_proj"))
        ]

        print(f"  Group {g_idx}: prefix='{prefix}'  blocks {group[0]}–{group[-1]}  ({n} blocks to fold into 1)")
        print(f"    Parameters per block : {len(suffixes)}")
        print(f"    Attention projections: {attn_params or '(none detected)'}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  CSR vertical stacking
# ══════════════════════════════════════════════════════════════════════════════


def stack_csr_vertical(csr_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Vertically stack N CSR matrices of shape [out_i, in] into one [sum(out_i), in].

    Inputs need not share row count but must share column count (in_dim).
    Pure index arithmetic: values/col_indices concatenate directly;
    crow_indices are offset by preceding blocks' total nonzeros then
    appended (skipping each subsequent block's leading 0).
    """
    if not csr_list:
        raise ValueError("csr_list is empty")
    if len(csr_list) == 1:
        return csr_list[0]

    all_values = []
    all_col_indices = []
    crow_parts = []
    nnz_offset = 0
    total_rows = 0

    for csr in csr_list:
        vals = csr.values()
        cols = csr.col_indices()
        crows = csr.crow_indices()  # length = n_rows + 1

        all_values.append(vals)
        all_col_indices.append(cols)

        if total_rows == 0:
            crow_parts.append(crows)  # include the leading 0
        else:
            # Skip the leading 0; offset all entries by nnz accumulated so far
            crow_parts.append(crows[1:] + nnz_offset)

        nnz_offset += vals.numel()
        total_rows += crows.numel() - 1  # n_rows = len(crow_indices) - 1

    stacked_values = torch.cat(all_values)
    stacked_col_indices = torch.cat(all_col_indices)
    stacked_crow = torch.cat(crow_parts)

    n_cols = csr_list[0].shape[1]

    return torch.sparse_csr_tensor(
        stacked_crow,
        stacked_col_indices,
        stacked_values,
        size=(total_rows, n_cols),
        dtype=csr_list[0].dtype,
    )


def slice_csr_rows(csr: torch.Tensor, row_start: int, row_end: int) -> torch.Tensor:
    """
    Extract rows [row_start, row_end) from a CSR matrix as a dense tensor.

    O(nnz_in_slice) -- only materialises the needed rows. Returns float32
    dense [row_end - row_start, n_cols].
    """
    crow = csr.crow_indices()
    cols = csr.col_indices()
    vals = csr.values()
    n_col = csr.shape[1]

    nnz_start = int(crow[row_start].item())
    nnz_end = int(crow[row_end].item())

    slice_vals = vals[nnz_start:nnz_end]
    slice_cols = cols[nnz_start:nnz_end]
    # New crow_indices for the slice
    slice_crow = crow[row_start : row_end + 1] - nnz_start

    n_rows = row_end - row_start
    csr_slice = torch.sparse_csr_tensor(
        slice_crow,
        slice_cols,
        slice_vals,
        size=(n_rows, n_col),
        dtype=csr.dtype,
    )
    # Densify: for a weight slice we need a regular matrix for matmul
    return csr_slice.to_dense()


# ══════════════════════════════════════════════════════════════════════════════
#  Attention banding
# ══════════════════════════════════════════════════════════════════════════════


def make_banded_attention_mask(
    seq_len: int,
    band_half_width: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build an additive attention mask restricting each query position to keys
    within +-band_half_width positions: 0 where allowed, -inf where blocked.
    Shape [seq_len, seq_len]. See docs/research/rnn_fold.rst:
    make_banded_attention_mask.locality_rationale for why folding needs this
    at all (state carries cross-fold-step context that must stay local).
    """
    q_pos = torch.arange(seq_len, device=device).unsqueeze(1)  # [L, 1]
    k_pos = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, L]
    blocked = (q_pos - k_pos).abs() > band_half_width
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    mask.masked_fill_(blocked, float("-inf"))
    return mask


def infer_seq_len_from_attn_weight(
    weight_shape: torch.Size,
) -> int:
    """
    Guess the sequence length / attention band-width from a Q/K/V weight
    shape. Convention: Q/K/V weights are [d_model, d_k] or [d_model,
    d_model]; the row dimension equals the sequence length used for
    position embeddings in most fixed-position transformers. Rotary/ALiBi
    architectures should override band_half_width explicitly in
    fold_sparse_payload() instead of trusting this.
    """
    return int(weight_shape[0])


# ══════════════════════════════════════════════════════════════════════════════
#  Folded block descriptor
# ══════════════════════════════════════════════════════════════════════════════


class FoldedBlockDescriptor:
    """
    All metadata needed to execute one folded block at inference time.

    Attributes
    ----------
    n_folds                 : number of original blocks folded together
    block_indices           : original block indices (e.g. [0,1,2,...,23])
    stacked_weights         : {param_suffix: stacked_csr_tensor}
    out_dims                : {param_suffix: int}  -- rows per fold step in stacked matrix
    band_half_widths        : {param_suffix: int | None}  -- attention band, None = no mask
    prefix                  : common name prefix before the block index
    skip_connection_outputs : if True, average per-step outputs instead of
                              final accumulated state -- see docs/research/rnn_fold.rst:
                              RNNFoldedBlock.average_vs_sum_rationale
    """

    def __init__(
        self,
        n_folds: int,
        block_indices: list[int],
        stacked_weights: dict[str, torch.Tensor],
        out_dims: dict[str, int],
        band_half_widths: dict[str, int | None],
        prefix: str,
        skip_connection_outputs: bool = False,
    ):
        self.n_folds = n_folds
        self.block_indices = block_indices
        self.stacked_weights = stacked_weights
        self.out_dims = out_dims
        self.band_half_widths = band_half_widths
        self.prefix = prefix
        self.skip_connection_outputs = skip_connection_outputs

    def fold_weight(self, suffix: str, fold_step: int) -> torch.Tensor:
        """Dense weight slice for `suffix` at `fold_step` (rows [fold_step*out_dim:(fold_step+1)*out_dim], densified)."""
        csr = self.stacked_weights[suffix]
        out_dim = self.out_dims[suffix]
        r_start = fold_step * out_dim
        r_end = r_start + out_dim
        return slice_csr_rows(csr, r_start, r_end)

    def fold_weight_csr(self, suffix: str, fold_step: int) -> torch.Tensor:
        """
        CSR weight slice for `suffix` at `fold_step` WITHOUT densifying --
        use when passing weights to a SparseLinearLayer via load_weights()
        (densifying first throws away the sparsity structure it expects).
        Column indices are re-sorted per row via a COO coalesce round-trip
        since the delta-CSR kernels require ascending column order, which
        the stacked CSR does not guarantee.
        """
        csr = self.stacked_weights[suffix]
        out_dim = self.out_dims[suffix]
        r_start = fold_step * out_dim
        r_end = r_start + out_dim
        crow = csr.crow_indices()
        cols = csr.col_indices()
        vals = csr.values()
        n_col = csr.shape[1]
        nnz_start = int(crow[r_start].item())
        nnz_end = int(crow[r_end].item())
        slice_crow = crow[r_start : r_end + 1] - nnz_start
        csr_slice = torch.sparse_csr_tensor(
            slice_crow, cols[nnz_start:nnz_end], vals[nnz_start:nnz_end], size=(out_dim, n_col), dtype=csr.dtype
        )
        return csr_slice.to_sparse().coalesce().to_sparse_csr()  # sort col indices via COO round-trip

    def attention_mask(
        self,
        suffix: str,
        seq_len: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor | None:
        """Banded attention mask for a Q/K/V parameter, or None if not an attention projection / banding disabled."""
        bw = self.band_half_widths.get(suffix)
        if bw is None:
            return None
        return make_banded_attention_mask(seq_len, bw, device=device)

    def summary(self) -> str:
        skip_str = "averaged (skip-conn)" if self.skip_connection_outputs else "final accumulated state"
        lines = [
            f"FoldedBlock: {self.n_folds} folds  prefix='{self.prefix}'  output={skip_str}",
            f"  Parameters : {len(self.stacked_weights)}",
        ]
        for suffix, csr in self.stacked_weights.items():
            od = self.out_dims[suffix]
            bw = self.band_half_widths.get(suffix)
            bw_str = f"  band±{bw}" if bw is not None else ""
            nnz = int(csr.values().numel())
            lines.append(f"  {suffix:<45} stacked={tuple(csr.shape)}  nnz={nnz:,}  fold_out={od}{bw_str}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Reference inference module
# ══════════════════════════════════════════════════════════════════════════════


class RNNFoldedBlock(nn.Module):
    """
    Reference inference module for a single folded block descriptor.

    Architecture-agnostic skeleton showing the canonical RNN fold execution
    pattern. Real use requires either (a) subclassing and overriding
    _apply_block(), or (b) using FoldedBlockDescriptor directly inside an
    architecture-specific forward() method.

    Two output modes, controlled by descriptor.skip_connection_outputs: the
    standard mode returns the final accumulated state; the skip-connection
    mode returns the mean of all per-step outputs instead. See
    docs/research/rnn_fold.rst:RNNFoldedBlock.average_vs_sum_rationale for
    why averaging (not summing) and RNNFoldedBlock.no_gating_rationale for
    why no LSTM/GRU-style gates.
    """

    def __init__(self, descriptor: FoldedBlockDescriptor):
        super().__init__()
        self.desc = descriptor

    def _apply_block(
        self,
        x: torch.Tensor,  # [batch, seq, d_model]
        weights: dict[str, torch.Tensor],  # {suffix: dense weight}
        masks: dict[str, torch.Tensor | None],  # {suffix: attn mask | None}
    ) -> torch.Tensor:
        """Override in architecture-specific subclasses; base implementation raises NotImplementedError."""
        raise NotImplementedError(
            "RNNFoldedBlock._apply_block() must be overridden for your "
            "specific transformer architecture.  Use FoldedBlockDescriptor "
            "directly if you are integrating into an existing forward() method."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        seq_len = x.shape[1] if x.ndim == 3 else x.shape[0]

        state = torch.zeros_like(x)
        suffixes = list(self.desc.stacked_weights.keys())

        # State update is identical in both modes; only the return value differs.
        step_outputs: list[torch.Tensor] = []

        for fold_step in range(self.desc.n_folds):
            weights = {s: self.desc.fold_weight(s, fold_step).to(device) for s in suffixes}
            masks = {s: self.desc.attention_mask(s, seq_len, device) for s in suffixes}
            out = self._apply_block(x + state, weights, masks)
            state = state + out

            if self.desc.skip_connection_outputs:
                step_outputs.append(out)

        if self.desc.skip_connection_outputs:
            return torch.stack(step_outputs, dim=0).mean(dim=0)  # [n_folds, ...] -> mean

        return state


# ══════════════════════════════════════════════════════════════════════════════
#  SiliBlock: SparseLinearLayer-backed folded block with hoyer_score dispatch
# ══════════════════════════════════════════════════════════════════════════════

# hoyer_score > 0.8 means roughly <20% of activations are active -> routes
# to forward_sparse instead of forward_dense. Fixed for now; could become
# adaptive (benchmarked) later -- add a TODO.md entry if/when adapting.
_HOYER_SPARSE_THRESHOLD = 0.8


class SiliBlock(RNNFoldedBlock):
    """
    SparseLinearLayer-backed version of RNNFoldedBlock. Requires _sili_cpu
    (_cpu module) importable; raises ImportError at construction otherwise.

    Each fold step's SparseLinearLayer is pre-built at construction from the
    CSR slice (not re-loaded per call). Dispatch (Python-level, not in the
    C++ hot path): hoyer_score(x) > threshold -> forward_sparse, else
    forward_dense; same threshold applies to the gradient on backward.
    Per-row importance/value-scale normalization (lr / nnz_this_row) is
    always enabled on backward (lr_per_row_nnz=True).
    """

    # Per-parameter-suffix threshold can be overridden per instance
    hoyer_threshold: float = _HOYER_SPARSE_THRESHOLD

    def __init__(
        self,
        descriptor: FoldedBlockDescriptor,
        learning_rate: float = 0.01,
        num_cpus: int = 4,
    ):
        if not _SILI_AVAILABLE:
            raise ImportError(
                "SiliBlock requires the _cpu extension module (sili/cpu_backend.cpp). "
                "Build it with setup.py or cmake before using SiliBlock."
            )
        super().__init__(descriptor)
        self._lr = learning_rate
        self._num_cpus = num_cpus

        # ONE SparseLinearLayer per suffix, loaded with the FULL stacked
        # weight matrix (not per-fold-step slices) -- see docs/research/
        # rnn_fold.rst:SiliBlock.single_call_stacked_layer_design for why,
        # and for the weight-orientation transpose below (SparseLinearLayer
        # is [n_inputs x n_outputs]; stacked_weights[suffix] is
        # [n_folds*out_dim x in_dim]).
        import numpy as np

        # FP4 table largest magnitude -- scale each row so its max maps here.
        _FP4_MAX = 6.0
        self._layers: dict[str, object] = {}
        for suffix, csr in descriptor.stacked_weights.items():
            # csr.t() then .to_sparse_csr(): metadata-only transpose + real
            # but nnz-proportional (never dense) reorg. NEVER .to_dense()
            # here -- see SiliBlock.single_call_stacked_layer_design.
            csr_t = csr.t().to_sparse_csr()
            n_in = int(csr_t.shape[0])  # in_dim
            n_out = int(csr_t.shape[1])  # n_folds * out_dim
            nnz = int(csr_t.values().numel())
            layer = _sili_cpu.SparseLinearLayer(n_in, n_out, int(nnz * 1.2) + 64, num_cpus)
            ptrs = csr_t.crow_indices().numpy().astype(np.int32)
            idx = csr_t.col_indices().numpy().astype(np.int32)
            vals = csr_t.values().float().numpy().copy()

            # Per-row FP4 scaling -- see docs/research/rnn_fold.rst:
            # SiliBlock.per_row_fp4_scaling. DO NOT call rescale_value_row()
            # after this pre-scaled load -- it would re-encode already-scaled values.
            row_scales = np.ones(n_in, dtype=np.float32)
            for r in range(n_in):
                start, end = int(ptrs[r]), int(ptrs[r + 1])
                if end > start:
                    max_abs = float(np.abs(vals[start:end]).max())
                    if max_abs > 0.0:
                        row_scales[r] = max_abs / _FP4_MAX
                        vals[start:end] /= row_scales[r]

            layer.load_weights(ptrs, idx, vals)

            # Set the per-row scale metadata (stored values are already scaled)
            for r in range(n_in):
                if row_scales[r] != 1.0:
                    layer.set_value_scale_raw(r, row_scales[r])

            self._layers[suffix] = layer

    def _forward_one_suffix(
        self,
        layer: object,  # SparseLinearLayer
        x: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        """Forward with hoyer_score dispatch. x=[batch,n_in] -> [batch,n_folds*out_dim]."""
        import numpy as np

        x2d = x.reshape(1, -1).astype(np.float32)
        if _sili_cpu.hoyer_score(x2d)["hoyer_score"] > self.hoyer_threshold:
            ptrs, idx, vals = _sili_cpu.dense_to_csr(x2d, 0.0)
            return layer.forward_sparse(ptrs, idx, vals, batch=1, learning_rate=lr)
        return layer.forward_dense(x2d, lr)

    def _backward_one_suffix(
        self,
        layer: object,  # SparseLinearLayer
        dy: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        """
        Backward through one SparseLinearLayer. lr_per_row_nnz always True.
        Sparse gradient path (hoyer_score > threshold) uses backward_sparse
        with last_input (exposed as a Python property); falls back to
        backward_dense if no forward pass has run yet or the gradient is dense.
        """
        import numpy as np

        dy2d = dy.reshape(1, -1).astype(np.float32)
        if _sili_cpu.hoyer_score(dy2d)["hoyer_score"] > self.hoyer_threshold:
            x_last = layer.last_input  # None if no forward run yet
            if x_last is not None:
                ptrs, idx, vals = _sili_cpu.dense_to_csr(dy2d, 0.0)
                return layer.backward_sparse(x_last, ptrs, idx, vals, batch=1, learning_rate=lr, lr_per_row_nnz=True)
        return layer.backward_dense(dy2d, lr, lr_per_row_nnz=True)

    def backward_sili(
        self,
        dy: torch.Tensor,
        lr: float = 0.0,
    ) -> torch.Tensor:
        """
        Backward pass through the entire folded RNN block. Reverses both
        levels of sum in forward_sili: (1) fold sum -- dy[batch, out_dim]
        broadcasts to dy_raw[batch, n_folds, out_dim] (gradient of a sum is
        1 per summand, so every fold slot gets the same dy, not split
        across folds); (2) suffix sum -- each suffix layer gets the same
        dy_raw, dx = sum of per-suffix dx. Weight updates happen inside
        _backward_one_suffix via backward_dense/backward_sparse with
        lr_per_row_nnz=True. dy: [batch, out_dim], dx: [batch, in_dim].
        """
        import numpy as np

        device = dy.device
        dy_np = dy.detach().cpu().float().numpy()
        batch = dy_np.shape[0]
        out_dim = next(iter(self.desc.out_dims.values()))

        # (1) Broadcast dy through the fold reshape+sum (reverse of forward's
        # raw[batch, n_folds*out_dim].sum(axis=1) -> [batch, out_dim]).
        dy_raw = (
            np.tile(dy_np.reshape(batch, 1, out_dim), (1, self.desc.n_folds, 1))
            .reshape(batch, self.desc.n_folds * out_dim)
            .astype(np.float32)
        )

        # (2) Each suffix sees the same dy_raw; accumulate dx across suffixes.
        dx_parts = [self._backward_one_suffix(layer, dy_raw, lr) for layer in self._layers.values()]
        dx_np = sum(dx_parts).reshape(batch, -1)

        return torch.from_numpy(dx_np).to(device)

    def forward_sili(
        self,
        x: torch.Tensor,
        lr: float = 0.0,
    ) -> torch.Tensor:
        """
        Single forward pass through the entire folded RNN block --
        _forward_one_suffix runs EXACTLY ONCE per suffix (not N times); the
        full stacked weight matrix handles all N fold steps in one shot.
        Shape contract matches RNNFoldedBlock.forward: [batch, hidden_dim]
        in and out. Internally each SparseLinearLayer produces
        [batch, n_folds*out_dim], mapped back to [batch, out_dim] by
        reshaping and summing across the fold dimension -- at
        initialization this approximates the sequential composition of all
        N original blocks (linear regime, sum ~= chain); after
        synaptogenesis only the sparse connections that contribute
        meaningfully survive.
        """
        device = x.device
        x_np = x.detach().cpu().float().numpy()

        out_parts = [self._forward_one_suffix(layer, x_np, lr) for layer in self._layers.values()]
        raw_np = sum(out_parts)  # sum suffix contributions (Q, K, V, MLP etc. all write to state)

        # out_dim (hidden_dim of original per-layer weights) agrees across suffixes for compatible blocks.
        out_dim = next(iter(self.desc.out_dims.values()))
        batch = raw_np.shape[0]
        summed = raw_np.reshape(batch, self.desc.n_folds, out_dim).sum(axis=1)

        return torch.from_numpy(summed).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-mode drop-in: routes through forward_sili(lr=0)."""
        return self.forward_sili(x, lr=0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Core folding logic
# ══════════════════════════════════════════════════════════════════════════════

# Parameter suffix patterns that identify attention projections (Q, K, V).
# The attention band mask is applied only to these.
_ATTN_SUFFIXES = re.compile(r"(q_proj|k_proj|v_proj|query|key|value|c_attn|in_proj)\.(weight|bias)$")


def _is_attention_param(suffix: str) -> bool:
    return bool(_ATTN_SUFFIXES.search(suffix))


def fold_block_group(
    group: list[int],
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    band_half_width_override: int | None = None,
    skip_connection_outputs: bool = False,
) -> FoldedBlockDescriptor:
    """
    Fold a group of consecutive identical blocks into a FoldedBlockDescriptor.
    For each parameter suffix shared by all blocks: collect CSR tensors
    (converting dense to CSR if needed), vertically stack into one CSR
    matrix, record out_dim (rows per fold step) and attention band width.

    band_half_width_override overrides the inferred attention band half-width.
    """
    n_folds = len(group)
    sample_idx = group[0]

    # Match prefix AND index -- see detect_repeated_block_groups.prefix_index_keying_bug.
    suffixes: list[str] = sorted(
        parsed[2]
        for name in state_dict
        if (parsed := _parse_block_key(name)) is not None and parsed[0] == prefix and parsed[1] == sample_idx
    )

    stacked_weights: dict[str, torch.Tensor] = {}
    out_dims: dict[str, int] = {}
    band_half_widths: dict[str, int | None] = {}

    for suffix in suffixes:
        per_block_csr: list[torch.Tensor] = []

        for block_idx in group:
            param_name = f"{prefix}{block_idx}{suffix}"
            raw = state_dict.get(param_name)
            if raw is None:
                raise KeyError(f"Expected parameter '{param_name}' not found in state_dict")

            # Accept a raw CSR tensor or a sparse_prune.py dict entry. See
            # docs/research/rnn_fold.rst:fold_block_group.dense_raw_entry_bug
            # for why the "raw" branch below must be treated identically to
            # the plain-dense-tensor branch, not skipped as a scalar.
            if isinstance(raw, dict):
                if "csr" in raw:
                    tensor = raw["csr"]
                elif "raw" in raw:
                    t = raw["raw"].detach().float()
                    if t.ndim == 0:
                        t = t.reshape(1, 1)
                    elif t.ndim == 1:
                        t = t.unsqueeze(0)
                    elif t.ndim > 2:
                        t = t.reshape(t.shape[0], -1)
                    tensor = t.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
                else:
                    raise ValueError(
                        f"dict entry for '{param_name}' has neither 'csr' nor 'raw' key: {sorted(raw.keys())}"
                    )
            elif isinstance(raw, torch.Tensor) and raw.layout == torch.sparse_csr:
                tensor = raw
            elif isinstance(raw, torch.Tensor):
                t = raw.detach().float()  # dense tensor -> CSR (no pruning, just layout)
                if t.ndim == 0:
                    t = t.reshape(1, 1)
                elif t.ndim == 1:
                    t = t.unsqueeze(0)
                elif t.ndim > 2:
                    t = t.reshape(t.shape[0], -1)
                tensor = t.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
            else:
                raise TypeError(f"Unexpected type for '{param_name}': {type(raw)}")

            per_block_csr.append(tensor)

        stacked = stack_csr_vertical(per_block_csr)
        out_dim = int(per_block_csr[0].shape[0])  # rows of a single block's matrix

        stacked_weights[suffix] = stacked
        out_dims[suffix] = out_dim

        if _is_attention_param(suffix):
            if band_half_width_override is not None:
                bw = band_half_width_override
            else:
                bw = infer_seq_len_from_attn_weight(per_block_csr[0].shape)
            band_half_widths[suffix] = bw
        else:
            band_half_widths[suffix] = None

    return FoldedBlockDescriptor(
        n_folds=n_folds,
        block_indices=group,
        stacked_weights=stacked_weights,
        out_dims=out_dims,
        band_half_widths=band_half_widths,
        prefix=prefix,
        skip_connection_outputs=skip_connection_outputs,
    )


def fold_sparse_payload(
    payload: dict,
    min_group_size: int = 2,
    band_half_width_override: int | None = None,
    skip_connection_outputs: bool = False,
) -> dict:
    """
    Top-level function: take a sparse_prune.py payload (the dict saved by
    sparse_prune.sparsify_model(), with a "sparse_state_dict" of
    {name: {"csr": tensor, "shape": ...} | {"raw": tensor}}) and fold
    repeated blocks: extract a flat tensor view, find repeated block
    groups, build a FoldedBlockDescriptor per group, and return a new
    payload with "folded_blocks" (serialisable descriptors) and
    "rnn_fold_meta" (summary stats) added, folded parameters removed from
    sparse_state_dict. skip_connection_outputs is passed straight through
    to each descriptor -- see RNNFoldedBlock.average_vs_sum_rationale.
    """
    ssd = payload.get("sparse_state_dict", payload)

    flat: dict[str, torch.Tensor] = {}
    for name, entry in ssd.items():
        if isinstance(entry, dict):
            csr = entry.get("csr")
            if csr is not None and isinstance(csr, torch.Tensor):
                flat[name] = csr
            elif "raw" in entry:
                flat[name] = entry["raw"]
        elif isinstance(entry, torch.Tensor):
            flat[name] = entry

    groups = detect_repeated_block_groups(flat, min_group_size=min_group_size)

    if not groups:
        print("[rnn_fold]  No repeated block groups found.  Payload unchanged.")
        return payload

    folded_descriptors: list[dict] = []
    removed_names: set = set()

    for prefix, group in groups:
        print(f"[rnn_fold]  Folding blocks {group[0]}–{group[-1]} (n={len(group)}, prefix='{prefix}')")

        desc = fold_block_group(
            group,
            flat,
            prefix,
            band_half_width_override=band_half_width_override,
            skip_connection_outputs=skip_connection_outputs,
        )

        print(desc.summary())
        print()

        # Match prefix AND index, AND a suffix actually in desc.stacked_weights --
        # see fold_block_group.dense_raw_entry_bug for why the latter matters.
        folded_suffixes = set(desc.stacked_weights.keys())
        for block_idx in group:
            for name in list(flat.keys()):
                parsed = _parse_block_key(name)
                if parsed and parsed[0] == prefix and parsed[1] == block_idx and parsed[2] in folded_suffixes:
                    removed_names.add(name)

        folded_descriptors.append(
            {
                "n_folds": desc.n_folds,
                "block_indices": desc.block_indices,
                "prefix": desc.prefix,
                "stacked_weights": desc.stacked_weights,
                "out_dims": desc.out_dims,
                "band_half_widths": desc.band_half_widths,
                "skip_connection_outputs": desc.skip_connection_outputs,
            }
        )

    cleaned_ssd = {k: v for k, v in ssd.items() if k not in removed_names}

    # _real_nnz counts real nonzeros regardless of CSR vs dense/strided layout --
    # see docs/research/rnn_fold.rst:fold_sparse_payload.nnz_accounting_bug.
    def _real_nnz(t: torch.Tensor) -> int:
        if t.layout == torch.sparse_csr:
            return int(t.values().numel())
        return int((t != 0).sum().item())  # dense/strided: count real nonzeros

    original_nnz = sum(_real_nnz(flat[n]) for n in removed_names if n in flat)
    folded_nnz = sum(int(csr.values().numel()) for fd in folded_descriptors for csr in fd["stacked_weights"].values())

    result = dict(payload)
    result["sparse_state_dict"] = cleaned_ssd
    result["folded_blocks"] = folded_descriptors
    result["rnn_fold_meta"] = {
        "n_groups": len(groups),
        "total_blocks_folded": sum(len(g) for g in groups),
        "params_removed": len(removed_names),
        "original_nnz": original_nnz,
        "folded_nnz": folded_nnz,
        "nnz_unchanged": original_nnz == folded_nnz,  # stacking is lossless
        "skip_connection_outputs": skip_connection_outputs,
    }

    _print_fold_summary(result["rnn_fold_meta"])
    return result


def _print_fold_summary(meta: dict) -> None:
    print("─" * 60)
    print("[rnn_fold summary]")
    print(f"  Groups folded          : {meta['n_groups']}")
    print(f"  Blocks → 1 each        : {meta['total_blocks_folded']}")
    print(f"  Parameters consolidated: {meta['params_removed']}")
    print(f"  Nonzeros (pre-fold)    : {meta['original_nnz']:,}")
    print(f"  Nonzeros (post-fold)   : {meta['folded_nnz']:,}")
    print(f"  Lossless stacking      : {meta['nnz_unchanged']}")
    print("─" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  RNN-all: per-layer recurrent extension (zero-init, no new synapses)
# ══════════════════════════════════════════════════════════════════════════════
# See docs/research/rnn_fold.rst:rnn_all.zero_init_transform_overview for
# the full per-layer recurrent transform, zero-init rationale, and conv handling.

# Parameter name fragments that indicate non-projection weights to skip.
_RNN_ALL_SKIP_SUFFIXES = re.compile(
    r"\.(bias|weight_g|weight_v|running_mean|running_var|num_batches_tracked"
    r"|ln_[0-9]|layer_norm|layernorm|norm|embed|pos_emb|position_embed"
    r"|wpe|wte)$",
    re.IGNORECASE,
)

# Embedding weight names often contain these substrings regardless of suffix.
_RNN_ALL_SKIP_NAMES = re.compile(
    r"embed|positional|lm_head|cls_token|patch_embed",
    re.IGNORECASE,
)


def _is_rnn_all_eligible(name: str, shape: torch.Size) -> bool:
    """True if extendable with a recurrent block: 2-D/4-D, no dim==1, name not a known non-projection pattern."""
    if _RNN_ALL_SKIP_SUFFIXES.search(name):
        return False
    if _RNN_ALL_SKIP_NAMES.search(name):
        return False
    if len(shape) == 2:
        return shape[0] > 1 and shape[1] > 1
    if len(shape) == 4:  # conv: [out_c, in_c, kH, kW]
        return shape[0] > 1 and shape[1] > 1
    return False


def _extend_csr_columns(csr: torch.Tensor, extra_cols: int) -> torch.Tensor:
    """
    Extend CSR [rows, cols] to [rows, cols+extra_cols]; new columns hold no
    nonzeros (zero-sparse init) -- pure metadata op, values/indices unchanged.
    """
    new_cols = csr.shape[1] + extra_cols
    return torch.sparse_csr_tensor(
        csr.crow_indices(),
        csr.col_indices(),
        csr.values(),
        size=(csr.shape[0], new_cols),
        dtype=csr.dtype,
    )


def _extend_dense_columns(t: torch.Tensor, extra_cols: int) -> torch.Tensor:
    """Extend dense [rows, cols] to [rows, cols+extra_cols] by appending zero columns (non-sparse weights)."""
    zeros = torch.zeros(t.shape[0], extra_cols, dtype=t.dtype, device=t.device)
    return torch.cat([t, zeros], dim=1)


def _extend_dense_conv(t: torch.Tensor, extra_in_channels: int) -> torch.Tensor:
    """Extend conv [out_c, in_c, kH, kW] to [out_c, in_c+extra, kH, kW] by appending zero input-channel slices."""
    kH, kW = t.shape[2], t.shape[3]
    zeros = torch.zeros(t.shape[0], extra_in_channels, kH, kW, dtype=t.dtype, device=t.device)
    return torch.cat([t, zeros], dim=1)


class RNNAllLayerInfo:
    """
    Metadata for a single per-layer recurrent extension.

    Attributes
    ----------
    name          : parameter name in the state dict
    original_shape: shape before extension
    ff_dim        : original input dimension (cols 0..ff_dim-1 are feedforward)
    rec_dim       : recurrent dimension = output dim (out for linear, out_c for conv)
    is_conv       : True if a conv weight (4-D), False for linear (2-D)
    """

    def __init__(self, name: str, original_shape: torch.Size, ff_dim: int, rec_dim: int, is_conv: bool):
        self.name = name
        self.original_shape = original_shape
        self.ff_dim = ff_dim
        self.rec_dim = rec_dim
        self.is_conv = is_conv

    def state_shape(self, batch: int = 1, seq: int = 1, spatial_h: int = 1, spatial_w: int = 1) -> tuple:
        """Runtime state shape: linear -> (batch, seq, rec_dim); conv -> (batch, rec_dim, spatial_h, spatial_w)."""
        if self.is_conv:
            return (batch, self.rec_dim, spatial_h, spatial_w)
        return (batch, seq, self.rec_dim)

    def __repr__(self) -> str:
        kind = "conv" if self.is_conv else "linear"
        return (
            f"RNNAllLayerInfo({kind}  {self.name}  {tuple(self.original_shape)} → ff={self.ff_dim} rec={self.rec_dim})"
        )


def apply_rnn_all_to_payload(payload: dict) -> dict:
    """
    Extend every eligible linear/conv weight in the payload with a
    zero-sparse recurrent input block, converting each op into a per-layer
    RNN cell -- see docs/research/rnn_fold.rst:
    rnn_all.zero_init_transform_overview for the transform and rationale.
    Returns payload with "sparse_state_dict" replaced by extended versions,
    "rnn_all_layers" (list of RNNAllLayerInfo dicts), "rnn_all_meta" (summary counts).
    """
    ssd = payload.get("sparse_state_dict", payload)

    layer_infos: list[RNNAllLayerInfo] = []
    new_ssd = {}

    n_extended = 0
    n_skipped = 0
    extra_cols_total = 0

    COL = 56
    header = f"  {'Parameter':<{COL}} {'Original':>16} {'Extended':>16}  rec_dim"
    sep = "─" * len(header)
    print("\n[rnn_all]  Extending layers to per-layer RNN cells …")
    print(f"\n{header}")
    print(sep)

    for name, entry in ssd.items():
        # ── Unwrap entry format from sparse_prune.py ─────────────────────────
        is_sparse_entry = isinstance(entry, dict)
        if is_sparse_entry:
            csr = entry.get("csr")
            raw = entry.get("raw")
            shape = entry.get("shape", csr.shape if csr is not None else (raw.shape if raw is not None else None))
        elif isinstance(entry, torch.Tensor):
            csr, raw, shape = (
                (entry, None, entry.shape) if entry.layout == torch.sparse_csr else (None, entry, entry.shape)
            )
        else:
            new_ssd[name] = entry
            n_skipped += 1
            continue

        if shape is None:
            new_ssd[name] = entry
            n_skipped += 1
            continue

        shape_ts = torch.Size(shape) if not isinstance(shape, torch.Size) else shape

        if not _is_rnn_all_eligible(name, shape_ts):
            new_ssd[name] = entry
            n_skipped += 1
            continue

        is_conv = len(shape_ts) == 4
        out_dim = int(shape_ts[0])  # output channels / rows
        in_dim = int(shape_ts[1])  # input channels / cols
        rec_dim = out_dim  # recurrent dim = output dim

        if is_conv:
            # Conv: extended shape [out_c, in_c+rec, kH, kW]
            new_shape = torch.Size([out_dim, in_dim + rec_dim, shape_ts[2], shape_ts[3]])
        else:
            # Linear: extended shape [out, in+rec]
            new_shape = torch.Size([out_dim, in_dim + rec_dim])

        # ── Perform the extension ─────────────────────────────────────────────
        if csr is not None and csr.layout == torch.sparse_csr and not is_conv:
            # Sparse linear: widen column count, no new nonzeros
            new_tensor = _extend_csr_columns(csr, rec_dim)
            new_entry = {"csr": new_tensor, "shape": new_shape}
            extra_cols_total += rec_dim
        elif raw is not None and not is_conv:
            # Dense linear (inside sparse_prune entry or plain tensor)
            new_tensor = _extend_dense_columns(raw.float(), rec_dim)
            new_entry = {"csr": None, "shape": new_shape, "raw": new_tensor}
        elif raw is not None and is_conv:
            new_tensor = _extend_dense_conv(raw.float(), rec_dim)
            new_entry = {"csr": None, "shape": new_shape, "raw": new_tensor}
        elif csr is not None and is_conv:
            # Conv stored as 2-D CSR (rows=out_c, cols=in_c*kH*kW): treat as
            # linear with in_dim = in_c*kH*kW, extend that
            new_tensor = _extend_csr_columns(csr, rec_dim)
            new_entry = {"csr": new_tensor, "shape": new_shape}
            extra_cols_total += rec_dim
        else:
            new_ssd[name] = entry
            n_skipped += 1
            continue

        if not is_sparse_entry and isinstance(entry, torch.Tensor):
            # Plain tensor input — return plain tensor
            _raw = new_entry.get("raw")
            new_entry = _raw if _raw is not None else new_entry.get("csr")

        new_ssd[name] = new_entry

        info = RNNAllLayerInfo(
            name=name,
            original_shape=shape_ts,
            ff_dim=in_dim,
            rec_dim=rec_dim,
            is_conv=is_conv,
        )
        layer_infos.append(info)
        n_extended += 1

        print(f"  {name:<{COL}} {tuple(shape_ts)!s:>16} {tuple(new_shape)!s:>16}  {rec_dim}")

    print(sep)
    print(f"\n[rnn_all]  {n_extended} layers extended,  {n_skipped} skipped")
    print(f"           extra recurrent columns (sparse, zero-init): {extra_cols_total:,}")
    print("           new nonzero count added: 0  (synaptogenesis grows these)")

    result = dict(payload)
    result["sparse_state_dict"] = new_ssd
    result["rnn_all_layers"] = [
        {
            "name": li.name,
            "original_shape": list(li.original_shape),
            "ff_dim": li.ff_dim,
            "rec_dim": li.rec_dim,
            "is_conv": li.is_conv,
        }
        for li in layer_infos
    ]
    result["rnn_all_meta"] = {
        "n_extended": n_extended,
        "n_skipped": n_skipped,
        "extra_rec_cols": extra_cols_total,
        "new_nonzeros_added": 0,
    }
    return result


class RNNAllStateBuffer:
    """
    Zero-initialized runtime state buffer for the per-layer RNN cells.

    Holds one state tensor per extended layer, zero-initialized.  The buffer
    lives entirely in memory and is never stored in the weight file.

    Usage
    ─────
        buf = RNNAllStateBuffer(rnn_all_layers_meta)
        # At each forward call, for each linear layer:
        x_aug = torch.cat([x, buf.get(name, x)], dim=-1)   # linear
        y     = F.linear(x_aug, weight)
        buf.update(name, y)

        # For conv:
        x_aug = torch.cat([x, buf.get_spatial(name, x)], dim=1)
        y     = F.conv2d(x_aug, weight, ...)
        buf.update(name, y)

        # Reset between independent sequences:
        buf.reset()
    """

    def __init__(self, layer_metas: list[dict], device: torch.device = torch.device("cpu")):
        self._states: dict[str, torch.Tensor | None] = {m["name"]: None for m in layer_metas}
        self._infos: dict[str, dict] = {m["name"]: m for m in layer_metas}
        self.device = device

    def reset(self) -> None:
        """Zero all states (call between independent sequences)."""
        for name in self._states:
            self._states[name] = None

    def get(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """
        Return the current state for a linear layer, broadcasting to match x.

        x is expected to be [..., in_features].  The state has shape
        [..., rec_dim] (last dim).  If not yet initialized, returns zeros.
        """
        state = self._states.get(name)
        rec_dim = self._infos[name]["rec_dim"]
        if state is None:
            shape = list(x.shape)
            shape[-1] = rec_dim
            return torch.zeros(shape, dtype=x.dtype, device=x.device)
        return state.to(dtype=x.dtype, device=x.device)

    def get_spatial(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """
        Return the current state for a conv layer, matching spatial dims of x.

        x is expected to be [batch, in_c, H, W].  State has shape
        [batch, rec_dim, H, W].  If not yet initialized or spatial size
        changed, returns zeros.
        """
        state = self._states.get(name)
        rec_dim = self._infos[name]["rec_dim"]
        batch, _, H, W = x.shape
        target_shape = (batch, rec_dim, H, W)
        if state is None or state.shape != target_shape:
            return torch.zeros(target_shape, dtype=x.dtype, device=x.device)
        return state.to(dtype=x.dtype, device=x.device)

    def update(self, name: str, y: torch.Tensor) -> None:
        """Store y as the new state for layer `name`."""
        self._states[name] = y.detach()

    def __repr__(self) -> str:
        initialized = sum(1 for v in self._states.values() if v is not None)
        return f"RNNAllStateBuffer({len(self._states)} layers, {initialized} initialized, device={self.device})"


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone entry point (folds an already-pruned sparse payload)
# ══════════════════════════════════════════════════════════════════════════════


def _load_payload(path: str) -> dict:
    """Load either a sparse_prune.py payload or a raw state dict."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"File not found: {path}")

    if p.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError:
            sys.exit("pip install safetensors")
        sd = load_file(str(p))
        return {"sparse_state_dict": dict(sd.items())}

    obj = torch.load(str(p), map_location="cpu", weights_only=False)

    # Recognise sparse_prune.py payload
    if isinstance(obj, dict) and "sparse_state_dict" in obj:
        return obj

    # Plain state dict or checkpoint
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj)
        return {"sparse_state_dict": dict(sd.items())}

    if hasattr(obj, "state_dict"):
        sd = dict(obj.state_dict())
        return {"sparse_state_dict": dict(sd.items())}

    sys.exit(f"Cannot interpret loaded object of type {type(obj)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="RNN-fold repeated transformer blocks in a sparse model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", nargs="?", help="sparse_prune.py output (.pt) or raw model")
    p.add_argument("-o", "--output", default=None, help="Output .pt path  (default: <input>_folded.pt)")
    p.add_argument("--show-groups", action="store_true", help="Detect and print block groups, then exit")
    p.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        metavar="N",
        help="Minimum consecutive identical blocks to fold  (default: 2)",
    )
    p.add_argument(
        "--band-width",
        type=int,
        default=None,
        metavar="N",
        help="Override attention band half-width  (default: infer from weight shape)",
    )
    p.add_argument(
        "--also-prune", action="store_true", help="Run sparse_prune.sparsify_model() on input before folding"
    )
    p.add_argument(
        "--bits",
        type=int,
        default=32,
        choices=[32, 16, 8, 4, 2],
        help="Quantization bits for pruning threshold (used with --also-prune)",
    )
    p.add_argument(
        "--skip-outputs",
        action="store_true",
        help="Average per-fold-step outputs instead of returning final "
        "accumulated state (skip-connection outputs / pyramidal pooling)",
    )
    p.add_argument(
        "--rnn-all",
        action="store_true",
        help="Extend every linear/conv weight with a zero-sparse recurrent "
        "input block, making each op a per-layer RNN cell. "
        "No new nonzeros at conversion time; synaptogenesis fills them.",
    )
    args = p.parse_args()

    if args.input is None:
        p.print_help()
        sys.exit(1)

    payload = _load_payload(args.input)

    if args.also_prune:
        try:
            import sparse_prune
        except ImportError:
            sys.exit("sparse_prune.py must be in the same directory or on PYTHONPATH")
        print("[rnn_fold]  Running sparse_prune first …")
        # Reuse sparsify_model internals: get a flat state dict, prune, repack
        flat_sd: dict[str, torch.Tensor] = {}
        for k, v in payload["sparse_state_dict"].items():
            if isinstance(v, torch.Tensor):
                flat_sd[k] = v
            elif isinstance(v, dict) and "raw" in v:
                flat_sd[k] = v["raw"]
        threshold = sparse_prune.default_min_abs_param(bits=args.bits)
        pruned_ssd = {}
        for name, tensor in flat_sd.items():
            csr, shape = sparse_prune.to_sparse_csr(tensor, threshold)
            pruned_ssd[name] = {"csr": csr, "shape": shape}
        payload["sparse_state_dict"] = pruned_ssd
        print(f"[rnn_fold]  Pruned {len(pruned_ssd)} tensors  (threshold={threshold:.4e})\n")

    if args.show_groups:
        flat: dict[str, torch.Tensor] = {}
        for name, entry in payload["sparse_state_dict"].items():
            if isinstance(entry, dict):
                t = entry.get("csr")
                if t is None:
                    t = entry.get("raw")
                if t is not None:
                    flat[name] = t
            elif isinstance(entry, torch.Tensor):
                flat[name] = entry
        groups = detect_repeated_block_groups(flat, min_group_size=args.min_group_size)
        report_block_groups(flat, groups)
        return

    working = payload

    # Fold unless --rnn-all is the only flag passed -- see docs/research/
    # rnn_fold.rst:main.rnn_fold_flag_bug.
    if not args.rnn_all:
        working = fold_sparse_payload(
            working,
            min_group_size=args.min_group_size,
            band_half_width_override=args.band_width,
            skip_connection_outputs=args.skip_outputs,
        )

    if args.rnn_all:
        working = apply_rnn_all_to_payload(working)

    out_suffix = ""
    if not args.rnn_all:
        out_suffix += "_folded"
    if args.rnn_all:
        out_suffix += "_rnnall"

    out_path = args.output or str(Path(args.input).parent / (Path(args.input).stem + out_suffix + ".pt"))
    torch.save(working, out_path)
    print(f"\n[saved]  {out_path}")


if __name__ == "__main__":
    main()
