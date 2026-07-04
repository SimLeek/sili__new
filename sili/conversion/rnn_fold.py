#!/usr/bin/env python3
"""
rnn_fold.py
───────────
Detect consecutive identical transformer blocks in a (sparse-pruned) model
and fold them into a single recurrent block whose weights are stored as one
stacked sparse CSR matrix.

Conceptual transform
────────────────────
Original:  in ──► [Block_0] ──► [Block_1] ──► ··· ──► [Block_{N-1}] ──► out
           N identical-structure blocks, each with different trained weights.

Folded:    ┌─────────────────────────────────────┐
           │  state = 0                           │
           │  for i in 0..N-1:                    │
           │    state += Block_folded(x + state)  │
           └─────────────────────────────────────-┘
           One block, N steps, state accumulates all fold outputs.

Weight layout in the stacked CSR matrix
────────────────────────────────────────
If each block has a weight of shape [out, in], the stacked matrix is
[N·out, in].  Visually, the nonzeros form horizontal bands:

  row 0         ┌──────────────────────────────┐
  …             │  Block_0 nonzeros  (band 0)  │
  row out-1     └──────────────────────────────┘
  row out        ┌─────────────────────────────┐
  …              │  Block_1 nonzeros  (band 1) │
  row 2·out-1   └─────────────────────────────┘
        …                    …
  row (N-1)·out  ┌────────────────────────────┐
  …              │ Block_{N-1} nonzeros (band N-1) │
  row N·out-1   └────────────────────────────┘

At fold step i the executing code slices rows [i·out : (i+1)·out] from the
CSR matrix and uses them as the current weight, so the original per-block
computation is faithfully reproduced.

Attention banding for Q / K / V projections
────────────────────────────────────────────
In each block the attention score matrix is [seq_len × seq_len].
After folding, the state fed back into the block carries information from
all previous fold steps. To prevent a fold-step i from "seeing" context that
was more than one fold step away — matching the locality of the original per-
block computation — an additive band mask is applied to the attention scores:

    mask[q, k] = 0       if |q − k| ≤ band_half_width
               = −∞      otherwise

where band_half_width defaults to seq_len (the row or column dimension of
the original Q/K/V projection weight).  With band_half_width = seq_len every
position in the current fold step is allowed to attend to every position in
the immediately adjacent state context, but nothing further back, bounding
the effective receptive field per fold step to match the original block.

Integration with sparse_prune.py
─────────────────────────────────
  from sparse_prune import load_state_dict
  from rnn_fold import fold_sparse_payload

  payload = torch.load("model_sparse.pt")           # from sparse_prune.py
  folded  = fold_sparse_payload(payload)
  torch.save(folded, "model_folded.pt")

  # Or as part of the sparse_prune pipeline:
  python sparse_prune.py model.pt --rnn-fold

CLI
───
  python rnn_fold.py model_sparse.pt                     # auto-detect & fold
  python rnn_fold.py model_sparse.pt -o model_folded.pt
  python rnn_fold.py model_sparse.pt --show-groups        # detect only, no fold
  python rnn_fold.py model.pt        --also-prune         # prune then fold
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    import _cpu as _sili_cpu   # SparseLinearLayer, hoyer_score, dense_to_csr
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
_BLOCK_RE = re.compile(r'^(.*?\.)(\d+)(\..+)$')


def _parse_block_key(name: str) -> Optional[Tuple[str, int, str]]:
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
    state_dict: Dict[str, torch.Tensor],
    min_group_size: int = 2,
) -> List[List[int]]:
    """
    Find runs of consecutive blocks that share identical parameter structure.

    Two blocks are considered structurally identical if every parameter suffix
    maps to the same tensor shape in both.

    Parameters
    ----------
    state_dict      : flat {name: tensor} dict (dense or sparse payload)
    min_group_size  : minimum consecutive identical blocks to constitute a group

    Returns
    -------
    List of groups.  Each group is a sorted list of block indices that can be
    folded together.  Non-repeated or structurally inconsistent blocks are
    excluded.

    Example
    -------
    Blocks 0–23 identical → [[0,1,2,...,23]]
    Blocks 0–11 one shape, 12–23 another → [[0..11], [12..23]]
    """
    # Gather {block_index: {suffix: shape}}
    block_params: Dict[int, Dict[str, torch.Size]] = defaultdict(dict)
    for name, tensor in state_dict.items():
        parsed = _parse_block_key(name)
        if parsed is None:
            continue
        _, idx, suffix = parsed
        shape = tensor.shape if isinstance(tensor, torch.Tensor) else None
        if shape is not None:
            block_params[idx][suffix] = shape

    if not block_params:
        return []

    sorted_indices = sorted(block_params.keys())

    # Walk through sorted indices and group consecutive structurally-equal blocks
    groups: List[List[int]] = []
    current_group: List[int] = [sorted_indices[0]]

    for prev_i, cur_i in zip(sorted_indices, sorted_indices[1:]):
        # Consecutive index check (no gaps)
        if cur_i != prev_i + 1:
            if len(current_group) >= min_group_size:
                groups.append(current_group)
            current_group = [cur_i]
            continue

        if block_params[prev_i] == block_params[cur_i]:
            current_group.append(cur_i)
        else:
            if len(current_group) >= min_group_size:
                groups.append(current_group)
            current_group = [cur_i]

    if len(current_group) >= min_group_size:
        groups.append(current_group)

    return groups


def report_block_groups(
    state_dict: Dict[str, torch.Tensor],
    groups: List[List[int]],
) -> None:
    """Print a human-readable summary of detected fold groups."""
    if not groups:
        print("[rnn_fold]  No repeated block groups detected.")
        return

    print(f"[rnn_fold]  Detected {len(groups)} foldable group(s):\n")
    for g_idx, group in enumerate(groups):
        n = len(group)
        sample_idx = group[0]
        # Collect parameter suffixes for this block
        suffixes = sorted(
            suffix for name in state_dict
            if (parsed := _parse_block_key(name)) is not None
            and parsed[1] == sample_idx
            for suffix in [parsed[2]]
        )
        # Find Q/K/V projections
        attn_params = [s for s in suffixes if any(t in s for t in
                       ('q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value',
                        'c_attn', 'in_proj'))]

        print(f"  Group {g_idx}: blocks {group[0]}–{group[-1]}  ({n} blocks to fold into 1)")
        print(f"    Parameters per block : {len(suffixes)}")
        print(f"    Attention projections: {attn_params or '(none detected)'}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  CSR vertical stacking
# ══════════════════════════════════════════════════════════════════════════════

def stack_csr_vertical(csr_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Vertically stack N CSR matrices of shape [out_i, in] into one [Σout_i, in].

    Each input matrix is expected to have shape [out_i, in] (2-D CSR).
    They need not have the same number of rows, but must share the same
    number of columns (in_dim).

    The stacking is pure index arithmetic on the three CSR arrays:
      values       : concatenate directly
      col_indices  : concatenate directly (column positions are unchanged)
      crow_indices : for each successive block, offset by the total nonzeros
                     from all preceding blocks, then append (skipping the
                     leading 0 of each subsequent block's crow_indices)

    Returns
    -------
    A single coalesced CSR tensor of shape [total_rows, in_dim].
    """
    if not csr_list:
        raise ValueError("csr_list is empty")
    if len(csr_list) == 1:
        return csr_list[0]

    all_values      = []
    all_col_indices = []
    crow_parts      = []
    nnz_offset      = 0
    total_rows      = 0

    for csr in csr_list:
        vals  = csr.values()
        cols  = csr.col_indices()
        crows = csr.crow_indices()          # length = n_rows + 1

        all_values.append(vals)
        all_col_indices.append(cols)

        if total_rows == 0:
            crow_parts.append(crows)        # include the leading 0
        else:
            # Skip the leading 0; offset all entries by nnz accumulated so far
            crow_parts.append(crows[1:] + nnz_offset)

        nnz_offset += vals.numel()
        total_rows += crows.numel() - 1     # n_rows = len(crow_indices) - 1

    stacked_values      = torch.cat(all_values)
    stacked_col_indices = torch.cat(all_col_indices)
    stacked_crow        = torch.cat(crow_parts)

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

    CSR row slicing is O(nnz_in_slice) — we only materialise the needed rows.
    Returns a float32 dense tensor of shape [row_end - row_start, n_cols].
    """
    crow  = csr.crow_indices()
    cols  = csr.col_indices()
    vals  = csr.values()
    n_col = csr.shape[1]

    nnz_start = int(crow[row_start].item())
    nnz_end   = int(crow[row_end].item())

    slice_vals = vals[nnz_start:nnz_end]
    slice_cols = cols[nnz_start:nnz_end]
    # New crow_indices for the slice
    slice_crow = crow[row_start : row_end + 1] - nnz_start

    n_rows = row_end - row_start
    csr_slice = torch.sparse_csr_tensor(
        slice_crow, slice_cols, slice_vals,
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
    dtype: torch.dtype   = torch.float32,
) -> torch.Tensor:
    """
    Build an additive attention mask that restricts each query position to
    only attend to keys within ±band_half_width positions.

    Shape: [seq_len, seq_len]
    Values:
      0        where |q_pos − k_pos| ≤ band_half_width  (allowed)
      -inf     otherwise                                  (blocked)

    Why this is needed after folding
    ──────────────────────────────────
    In the original N-block transformer each block computes full attention
    over the sequence.  After folding into one RNN block that runs N times,
    the state carried between steps mixes context from previous fold steps.
    To prevent step i from attending to "stale" context that is more than one
    fold step away — which would not have been possible in the original
    sequential layout — the attention window is capped at band_half_width.

    Setting band_half_width = seq_len (the default in fold_sparse_payload)
    means every query sees the full current sequence plus exactly one step of
    prior state, matching the per-block locality of the original model.

    For a pure same-sequence self-attention (no cross-step state in the keys)
    band_half_width = seq_len covers the entire matrix and the mask is all-0.
    The mask becomes non-trivial only when key/value context is extended with
    state from previous fold steps.

    Parameters
    ----------
    seq_len          : sequence length (and query/key count)
    band_half_width  : half-width of the allowed attention band
    device / dtype   : target device and float dtype

    Returns
    -------
    Additive bias tensor of shape [seq_len, seq_len].
    """
    q_pos = torch.arange(seq_len, device=device).unsqueeze(1)   # [L, 1]
    k_pos = torch.arange(seq_len, device=device).unsqueeze(0)   # [1, L]
    blocked = (q_pos - k_pos).abs() > band_half_width
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    mask.masked_fill_(blocked, float("-inf"))
    return mask


def infer_seq_len_from_attn_weight(
    weight_shape: torch.Size,
) -> int:
    """
    Guess the sequence length / attention band-width from a Q/K/V weight shape.

    Convention: Q/K/V projection weights are [d_model, d_k] or [d_model, d_model].
    The row dimension (d_model = weight_shape[0]) is the embedding dimension and
    also, in most fixed-position transformers, equals the sequence length used
    for position embeddings.  We return that as the band half-width.

    For architectures with rotary or ALiBi embeddings, override band_half_width
    explicitly in fold_sparse_payload().
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
    out_dims                : {param_suffix: int}  — rows per fold step in stacked matrix
    band_half_widths        : {param_suffix: int | None}  — attention band, None = no mask
    prefix                  : common name prefix before the block index
    skip_connection_outputs : if True, average per-step outputs instead of returning
                              final accumulated state (see RNNFoldedBlock.forward)
    """
    def __init__(
        self,
        n_folds:                 int,
        block_indices:           List[int],
        stacked_weights:         Dict[str, torch.Tensor],
        out_dims:                Dict[str, int],
        band_half_widths:        Dict[str, Optional[int]],
        prefix:                  str,
        skip_connection_outputs: bool = False,
    ):
        self.n_folds                 = n_folds
        self.block_indices           = block_indices
        self.stacked_weights         = stacked_weights
        self.out_dims                = out_dims
        self.band_half_widths        = band_half_widths
        self.prefix                  = prefix
        self.skip_connection_outputs = skip_connection_outputs

    def fold_weight(self, suffix: str, fold_step: int) -> torch.Tensor:
        """
        Return the dense weight slice for parameter `suffix` at `fold_step`.

        Slices rows [fold_step * out_dim : (fold_step+1) * out_dim] from the
        stacked CSR matrix and densifies them.
        """
        csr     = self.stacked_weights[suffix]
        out_dim = self.out_dims[suffix]
        r_start = fold_step * out_dim
        r_end   = r_start + out_dim
        return slice_csr_rows(csr, r_start, r_end)

    def fold_weight_csr(self, suffix: str, fold_step: int) -> torch.Tensor:
        """
        Return the CSR weight slice for parameter `suffix` at `fold_step`
        WITHOUT densifying. Use this when passing weights to a SparseLinearLayer
        via load_weights() -- the layer handles the sparse format natively and
        densifying before handing it over throws away the sparsity structure.

        Note: column indices are sorted within each row (via coalesce on a COO
        round-trip) since SparseLinearLayer's delta-CSR kernels require
        ascending column order within rows -- the stacked CSR may have unsorted
        column indices depending on how the original weights were stored.
        """
        csr     = self.stacked_weights[suffix]
        out_dim = self.out_dims[suffix]
        r_start = fold_step * out_dim
        r_end   = r_start + out_dim
        crow    = csr.crow_indices()
        cols    = csr.col_indices()
        vals    = csr.values()
        n_col   = csr.shape[1]
        nnz_start = int(crow[r_start].item())
        nnz_end   = int(crow[r_end].item())
        slice_crow = crow[r_start : r_end + 1] - nnz_start
        csr_slice = torch.sparse_csr_tensor(
            slice_crow, cols[nnz_start:nnz_end], vals[nnz_start:nnz_end],
            size=(out_dim, n_col), dtype=csr.dtype)
        # Ensure sorted column indices (required by delta-CSR kernels).
        # Round-trip through COO with coalesce() is the portable PyTorch way.
        return csr_slice.to_sparse().coalesce().to_sparse_csr()

    def attention_mask(
        self,
        suffix: str,
        seq_len: int,
        device:  torch.device = torch.device("cpu"),
    ) -> Optional[torch.Tensor]:
        """
        Return the banded attention mask for a Q/K/V parameter, or None if
        this parameter is not an attention projection or banding is disabled.
        """
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
            od  = self.out_dims[suffix]
            bw  = self.band_half_widths.get(suffix)
            bw_str = f"  band±{bw}" if bw is not None else ""
            nnz = int(csr.values().numel())
            lines.append(
                f"  {suffix:<45} stacked={tuple(csr.shape)}  "
                f"nnz={nnz:,}  fold_out={od}{bw_str}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Reference inference module
# ══════════════════════════════════════════════════════════════════════════════

class RNNFoldedBlock(nn.Module):
    """
    Reference inference module for a single folded block descriptor.

    This is an architecture-agnostic skeleton that shows the canonical RNN
    fold execution pattern.  Real use requires either:
      (a) subclassing and overriding _apply_block(), or
      (b) using FoldedBlockDescriptor directly inside an architecture-specific
          forward() method.

    Two output modes (controlled by descriptor.skip_connection_outputs)
    ────────────────────────────────────────────────────────────────────
    Standard (skip_connection_outputs=False):
        state = 0
        for i in range(n_folds):
            out    = block(x + state)
            state += out
        return state                    # final accumulated state

    Skip-connection outputs (skip_connection_outputs=True):
        state   = 0
        outputs = []
        for i in range(n_folds):
            out    = block(x + state)
            state += out
            outputs.append(out)
        return mean(outputs)            # average of all per-step outputs

    Why average and not sum?
    ────────────────────────
    Each fold step computes at a different temporal scale: early steps see
    only local context (narrow banded attention over x), later steps see
    richer integrated context (x + accumulated state from prior steps).
    Averaging lets every scale contribute equally to the final output,
    regardless of how many fold steps there are — analogous to how pyramidal
    neurons in cortex receive both fast/local signals on basal dendrites and
    slow/long-range signals on apical dendrites, with the soma averaging
    across both.

    Mathematically, with windowed attention of half-width W, fold step i has
    an effective receptive field of (i+1)·W tokens.  Averaging the outputs
    produces a weighted sum over all receptive-field sizes 1W…N·W with equal
    weight 1/N — equivalent to an ensemble over scales.  A plain sum would
    make the last step dominate (it has seen the most context); averaging
    removes that bias.

    Why no LSTM/GRU?
    ────────────────
    Gates add parameters and complexity.  The original transformer already
    has residual connections, layer norms, and attention — these collectively
    perform the role of gating.  A plain additive state lets the folded block
    leverage those existing mechanisms without duplicating them.  Attention
    banding enforces locality between fold steps, the main stability
    mechanism that an LSTM gate would otherwise provide.
    """

    def __init__(self, descriptor: FoldedBlockDescriptor):
        super().__init__()
        self.desc = descriptor

    def _apply_block(
        self,
        x:        torch.Tensor,                     # [batch, seq, d_model]
        weights:  Dict[str, torch.Tensor],           # {suffix: dense weight}
        masks:    Dict[str, Optional[torch.Tensor]], # {suffix: attn mask | None}
    ) -> torch.Tensor:
        """
        Override this in architecture-specific subclasses.

        The base implementation raises NotImplementedError.  See module
        docstring for the expected pattern.
        """
        raise NotImplementedError(
            "RNNFoldedBlock._apply_block() must be overridden for your "
            "specific transformer architecture.  Use FoldedBlockDescriptor "
            "directly if you are integrating into an existing forward() method."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device  = x.device
        seq_len = x.shape[1] if x.ndim == 3 else x.shape[0]

        state    = torch.zeros_like(x)
        suffixes = list(self.desc.stacked_weights.keys())

        # Collect per-step outputs when skip_connection_outputs is enabled.
        # The state update is identical in both modes — only the return value
        # differs (final state vs. mean of all per-step outputs).
        step_outputs: List[torch.Tensor] = []

        for fold_step in range(self.desc.n_folds):
            weights = {s: self.desc.fold_weight(s, fold_step).to(device)
                       for s in suffixes}
            masks   = {s: self.desc.attention_mask(s, seq_len, device)
                       for s in suffixes}
            out    = self._apply_block(x + state, weights, masks)
            state  = state + out

            if self.desc.skip_connection_outputs:
                step_outputs.append(out)

        if self.desc.skip_connection_outputs:
            # Stack along a new leading dimension → [n_folds, ...] then mean
            return torch.stack(step_outputs, dim=0).mean(dim=0)

        return state


# ══════════════════════════════════════════════════════════════════════════════
#  SiliBlock: SparseLinearLayer-backed folded block with hoyer_score dispatch
# ══════════════════════════════════════════════════════════════════════════════

# Threshold for routing to forward_sparse instead of forward_dense.
# hoyer_score > 0.8 means roughly <20% of activations are active.
# This fixed constant could eventually become adaptive (benchmarked) --
# note: also add that TODO to TODO.md if/when adapting.
_HOYER_SPARSE_THRESHOLD = 0.8


class SiliBlock(RNNFoldedBlock):
    """
    SparseLinearLayer-backed version of RNNFoldedBlock.

    Requires _sili_cpu (_cpu module) to be importable. Each fold step has its
    own pre-built SparseLinearLayer (built at construction from the CSR slice,
    not re-loaded per call), so the forward loop is just a layer call rather
    than a weight-load + matmul.

    Dispatch (Python-level, not in the C++ hot path):
      hoyer_score(x) > 0.8  ->  forward_sparse (activations are sparse)
      otherwise              ->  forward_dense

    For the backward pass the same threshold is applied to the gradient.
    Per-row importance and value scale normalization (lr / nnz_this_row) is
    always enabled on backward (lr_per_row_nnz=True).

    Raises ImportError at construction if _cpu is not available.
    """

    # Per-parameter-suffix threshold can be overridden per instance
    hoyer_threshold: float = _HOYER_SPARSE_THRESHOLD

    def __init__(
        self,
        descriptor:   "FoldedBlockDescriptor",
        learning_rate: float = 0.01,
        num_cpus:      int   = 4,
    ):
        if not _SILI_AVAILABLE:
            raise ImportError(
                "SiliBlock requires the _cpu extension module (sili/cpu_backend.cpp). "
                "Build it with setup.py or cmake before using SiliBlock."
            )
        super().__init__(descriptor)
        self._lr       = learning_rate
        self._num_cpus = num_cpus

        # ONE SparseLinearLayer per suffix -- loaded with the FULL stacked
        # weight matrix, not individual fold-step slices.
        #
        # Design (per conversation): loading all N fold steps into ONE layer
        # means forward_sili calls _forward_one_suffix EXACTLY ONCE, not N
        # times. At initialisation the result is equivalent to running the N
        # original transformer blocks sequentially. After synaptogenesis,
        # redundant connections across fold steps are pruned -- only what is
        # genuinely needed survives. Dense stacking would be an N-wide layer
        # with no benefit; sparsity is what makes the single-call design
        # viable, not a micro-optimisation.
        #
        # Weight orientation: SparseLinearLayer is [n_inputs x n_outputs].
        # stacked_weights[suffix] is [n_folds*out_dim x in_dim] (standard
        # weight-matrix orientation) -- transpose before loading so the layer
        # has n_inputs=in_dim, n_outputs=n_folds*out_dim.
        import numpy as np
        self._layers: Dict[str, object] = {}
        for suffix, csr in descriptor.stacked_weights.items():
            csr_t = csr.to_dense().t().to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
            n_in  = int(csr_t.shape[0])   # in_dim
            n_out = int(csr_t.shape[1])   # n_folds * out_dim
            nnz   = int(csr_t.values().numel())
            layer = _sili_cpu.SparseLinearLayer(
                n_in, n_out, int(nnz * 1.2) + 64, num_cpus)
            ptrs = csr_t.crow_indices().numpy().astype(np.int32)
            idx  = csr_t.col_indices().numpy().astype(np.int32)
            vals = csr_t.values().float().numpy()
            layer.load_weights(ptrs, idx, vals)
            self._layers[suffix] = layer

    def _forward_one_suffix(
        self,
        layer:  object,   # SparseLinearLayer
        x:      "np.ndarray",
        lr:     float,
    ) -> "np.ndarray":
        """Forward with hoyer_score dispatch. x=[batch,n_in] -> [batch,n_folds*out_dim]."""
        import numpy as np
        x2d = x.reshape(1, -1).astype(np.float32)
        if _sili_cpu.hoyer_score(x2d)["hoyer_score"] > self.hoyer_threshold:
            ptrs, idx, vals = _sili_cpu.dense_to_csr(x2d, 0.0)
            return layer.forward_sparse(ptrs, idx, vals, batch=1, learning_rate=lr)
        return layer.forward_dense(x2d, lr)

    def _backward_one_suffix(
        self,
        layer:  object,   # SparseLinearLayer
        dy:     "np.ndarray",
        lr:     float,
    ) -> "np.ndarray":
        """Backward with hoyer_score dispatch on gradient. lr_per_row_nnz always True."""
        import numpy as np
        dy2d = dy.reshape(1, -1).astype(np.float32)
        if _sili_cpu.hoyer_score(dy2d)["hoyer_score"] > self.hoyer_threshold:
            ptrs, idx, vals = _sili_cpu.dense_to_csr(dy2d, 0.0)
            return layer.backward_sparse(
                layer._last_input if hasattr(layer, "_last_input") else dy2d,
                ptrs, idx, vals, batch=1, learning_rate=lr, lr_per_row_nnz=True)
        return layer.backward_dense(dy2d, lr, lr_per_row_nnz=True)

    def forward_sili(
        self,
        x:   "torch.Tensor",
        lr:  float = 0.0,
    ) -> "torch.Tensor":
        """
        Single forward pass through the entire folded RNN block.

        _forward_one_suffix is called EXACTLY ONCE per suffix (not N times).
        The full stacked weight matrix handles all N fold steps in one shot.
        Output shape: [batch, n_folds * out_dim].
        """
        import numpy as np
        device = x.device
        x_np   = x.detach().cpu().float().numpy()
        out_parts = [self._forward_one_suffix(layer, x_np, lr)
                     for layer in self._layers.values()]
        out_np = sum(out_parts)
        return torch.from_numpy(out_np).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-mode drop-in: routes through forward_sili(lr=0)."""
        return self.forward_sili(x, lr=0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Core folding logic
# ══════════════════════════════════════════════════════════════════════════════

# Parameter suffix patterns that identify attention projections (Q, K, V).
# The attention band mask is applied only to these.
_ATTN_SUFFIXES = re.compile(
    r'(q_proj|k_proj|v_proj|query|key|value|c_attn|in_proj)\.(weight|bias)$'
)


def _is_attention_param(suffix: str) -> bool:
    return bool(_ATTN_SUFFIXES.search(suffix))


def fold_block_group(
    group:                    List[int],
    state_dict:               Dict[str, torch.Tensor],
    prefix:                   str,
    band_half_width_override: Optional[int] = None,
    skip_connection_outputs:  bool = False,
) -> FoldedBlockDescriptor:
    """
    Fold a group of consecutive identical blocks into a FoldedBlockDescriptor.

    For each parameter suffix shared by all blocks in the group:
      1. Collect the CSR tensors (or convert dense to CSR if needed).
      2. Vertically stack them into one large CSR matrix.
      3. Record out_dim (rows per fold step) and attention band width.

    Parameters
    ----------
    group                    : sorted list of block indices to fold
    state_dict               : the (sparse) state dict
    prefix                   : common name prefix, e.g. "model.layers."
    band_half_width_override : override the inferred attention band half-width

    Returns
    -------
    FoldedBlockDescriptor ready for use at inference time.
    """
    n_folds = len(group)
    sample_idx = group[0]

    # Collect all parameter suffixes present in the sample block
    suffixes: List[str] = sorted(
        parsed[2]
        for name in state_dict
        if (parsed := _parse_block_key(name)) is not None
        and parsed[1] == sample_idx
    )

    stacked_weights:  Dict[str, torch.Tensor]       = {}
    out_dims:         Dict[str, int]                 = {}
    band_half_widths: Dict[str, Optional[int]]       = {}

    for suffix in suffixes:
        per_block_csr: List[torch.Tensor] = []

        for block_idx in group:
            param_name = f"{prefix}{block_idx}{suffix}"
            raw = state_dict.get(param_name)
            if raw is None:
                raise KeyError(
                    f"Expected parameter '{param_name}' not found in state_dict"
                )

            # Accept either a raw CSR tensor or a dict entry from sparse_prune.py
            if isinstance(raw, dict):
                # sparse_prune.py format: {"csr": csr_tensor, "shape": ...}
                csr_entry = raw.get("csr")
                if csr_entry is None:
                    # Scalar / empty tensor — skip stacking, keep as-is
                    per_block_csr = None
                    break
                tensor = csr_entry
            elif isinstance(raw, torch.Tensor) and raw.layout == torch.sparse_csr:
                tensor = raw
            elif isinstance(raw, torch.Tensor):
                # Dense tensor — convert to CSR (no pruning, just layout)
                t = raw.detach().float()
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

        if per_block_csr is None:
            # Non-stackable entry (scalar) — skip
            continue

        stacked = stack_csr_vertical(per_block_csr)
        out_dim = int(per_block_csr[0].shape[0])   # rows of a single block's matrix

        stacked_weights[suffix] = stacked
        out_dims[suffix]        = out_dim

        # Determine attention band half-width
        if _is_attention_param(suffix):
            if band_half_width_override is not None:
                bw = band_half_width_override
            else:
                bw = infer_seq_len_from_attn_weight(per_block_csr[0].shape)
            band_half_widths[suffix] = bw
        else:
            band_half_widths[suffix] = None

    return FoldedBlockDescriptor(
        n_folds                 = n_folds,
        block_indices           = group,
        stacked_weights         = stacked_weights,
        out_dims                = out_dims,
        band_half_widths        = band_half_widths,
        prefix                  = prefix,
        skip_connection_outputs = skip_connection_outputs,
    )


def fold_sparse_payload(
    payload:                  dict,
    min_group_size:           int = 2,
    band_half_width_override: Optional[int] = None,
    skip_connection_outputs:  bool = False,
) -> dict:
    """
    Top-level function: take a sparse_prune.py payload and fold repeated blocks.

    The input payload is the dict saved by sparse_prune.sparsify_model():
    {
        "sparse_state_dict": { name: {"csr": tensor, "shape": ...} | {"raw": tensor} },
        "min_abs_param":     float,
        "meta":              { ... },
    }

    This function:
      1. Extracts the flat tensor view for block detection.
      2. Finds repeated block groups.
      3. For each group, builds a FoldedBlockDescriptor.
      4. Returns a new payload with folded_blocks added and folded parameters
         removed from sparse_state_dict.

    Parameters
    ----------
    payload                  : output of sparse_prune.sparsify_model()
    min_group_size           : minimum consecutive identical blocks to fold
    band_half_width_override : override inferred attention band half-width
    skip_connection_outputs  : if True, each fold step's output is collected
                               and the mean is returned instead of the final
                               accumulated state (see RNNFoldedBlock docstring)

    Returns
    -------
    Modified payload dict with extra keys:
      "folded_blocks": list of serialisable FoldedBlockDescriptor dicts
      "rnn_fold_meta": summary statistics
    """
    ssd = payload.get("sparse_state_dict", payload)

    # Build a flat {name: tensor} view for detection
    flat: Dict[str, torch.Tensor] = {}
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

    # Determine the prefix for each group from the first block's first param
    def _group_prefix(group: List[int]) -> str:
        idx = group[0]
        for name in flat:
            parsed = _parse_block_key(name)
            if parsed and parsed[1] == idx:
                return parsed[0]
        return ""

    folded_descriptors: List[dict] = []
    removed_names: set = set()

    for group in groups:
        prefix = _group_prefix(group)
        print(f"[rnn_fold]  Folding blocks {group[0]}–{group[-1]} "
              f"(n={len(group)}, prefix='{prefix}')")

        desc = fold_block_group(
            group, flat, prefix,
            band_half_width_override=band_half_width_override,
            skip_connection_outputs=skip_connection_outputs,
        )

        print(desc.summary())
        print()

        # Record which param names to remove from the flat state dict
        for block_idx in group:
            for name in list(flat.keys()):
                parsed = _parse_block_key(name)
                if parsed and parsed[1] == block_idx:
                    removed_names.add(name)

        # Serialise descriptor for the payload
        folded_descriptors.append({
            "n_folds":                 desc.n_folds,
            "block_indices":           desc.block_indices,
            "prefix":                  desc.prefix,
            "stacked_weights":         desc.stacked_weights,
            "out_dims":                desc.out_dims,
            "band_half_widths":        desc.band_half_widths,
            "skip_connection_outputs": desc.skip_connection_outputs,
        })

    # Build cleaned sparse_state_dict (without folded params)
    cleaned_ssd = {k: v for k, v in ssd.items() if k not in removed_names}

    # Compute fold statistics
    #
    # original_nnz must count ACTUAL nonzero content regardless of whether a
    # given tensor started as CSR or dense/strided layout -- the old
    # `flat[n].layout == torch.sparse_csr` filter silently EXCLUDED
    # dense-format entries (LayerNorm weights, and anything that fell back to
    # dense via min_sparsity/max_sparse_ratio) from this count entirely,
    # while folded_nnz correctly counts EVERYTHING post-stacking (fold_block_group
    # converts dense entries to CSR too, see line ~654). That asymmetry made
    # "lossless stacking" read False whenever any dense-format layer
    # participated in a fold group -- not because stacking actually lost or
    # duplicated data, but because the "before" count was silently smaller
    # than the true content it was supposed to represent.
    def _real_nnz(t: torch.Tensor) -> int:
        if t.layout == torch.sparse_csr:
            return int(t.values().numel())
        return int((t != 0).sum().item())   # dense/strided: count real nonzeros

    original_nnz = sum(
        _real_nnz(flat[n]) for n in removed_names if n in flat
    )
    folded_nnz = sum(
        int(csr.values().numel())
        for fd in folded_descriptors
        for csr in fd["stacked_weights"].values()
    )

    result = dict(payload)
    result["sparse_state_dict"] = cleaned_ssd
    result["folded_blocks"]     = folded_descriptors
    result["rnn_fold_meta"] = {
        "n_groups":               len(groups),
        "total_blocks_folded":    sum(len(g) for g in groups),
        "params_removed":         len(removed_names),
        "original_nnz":           original_nnz,
        "folded_nnz":             folded_nnz,
        "nnz_unchanged":          original_nnz == folded_nnz,  # stacking is lossless
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

"""
Per-layer recurrent transform
──────────────────────────────
For every linear weight W of shape [out, in] (or conv weight [out, in_c, *k])
we extend the input dimension so the layer now accepts cat(x, state) as input:

  W_extended : [out,  in + out]       # linear
  W_extended : [out, in_c + out, *k]  # conv (out channels appended to in_c)

  Layout of extended weight columns:
    cols  0 … in-1          : original feedforward weights (unchanged)
    cols  in … in+out-1     : recurrent block (zero-sparse at conversion time)

  Recurrence per call:
    y     = W_ff @ x  +  W_rec @ state   # W_ff = cols[:in], W_rec = cols[in:]
    state = y                             # update for next call

  Zero init:
    state starts as all-zeros.  The recurrent block also starts empty (no
    nonzero CSR entries).  This means the first forward pass is bit-identical
    to the original model.  Synaptogenesis (a separate process) grows
    nonzeros into the recurrent block over time, using the zero-init
    trajectory as a supervised signal.

    Because no new synapses exist at conversion time, file size does not
    increase (new CSR columns are free — they just extend the shape metadata).

  Conv handling:
    For a conv kernel [out_c, in_c, kH, kW] the recurrent state is a feature
    map [out_c, H, W].  The kernel is extended to [out_c, in_c+out_c, kH, kW].
    At call time the state is spatially aligned with x and concatenated along
    the channel dimension before convolution.  The new in_c+out_c input
    channels map cleanly to the extended kernel.

  Skipped layers:
    Embeddings, biases, layer-norm parameters, and 1-D weights are not
    extended — they are not linear projections in the matmul sense.
"""

# Parameter name fragments that indicate non-projection weights to skip.
_RNN_ALL_SKIP_SUFFIXES = re.compile(
    r'\.(bias|weight_g|weight_v|running_mean|running_var|num_batches_tracked'
    r'|ln_[0-9]|layer_norm|layernorm|norm|embed|pos_emb|position_embed'
    r'|wpe|wte)$',
    re.IGNORECASE,
)

# Embedding weight names often contain these substrings regardless of suffix.
_RNN_ALL_SKIP_NAMES = re.compile(
    r'embed|positional|lm_head|cls_token|patch_embed',
    re.IGNORECASE,
)


def _is_rnn_all_eligible(name: str, shape: torch.Size) -> bool:
    """
    Return True if this parameter should be extended with a recurrent block.

    Eligibility rules:
      1. Must be 2-D (linear) or 4-D (conv).
      2. Neither dimension may be 1 (no bias-shaped weights, no single-channel).
      3. Name must not match known non-projection patterns.
    """
    if _RNN_ALL_SKIP_SUFFIXES.search(name):
        return False
    if _RNN_ALL_SKIP_NAMES.search(name):
        return False
    if len(shape) == 2:
        return shape[0] > 1 and shape[1] > 1
    if len(shape) == 4:           # conv: [out_c, in_c, kH, kW]
        return shape[0] > 1 and shape[1] > 1
    return False


def _extend_csr_columns(csr: torch.Tensor, extra_cols: int) -> torch.Tensor:
    """
    Extend a CSR matrix of shape [rows, cols] to [rows, cols + extra_cols].

    The new columns contain no nonzero entries (zero-sparse init).
    This is a pure metadata operation: values and indices are unchanged;
    only the stored shape is widened.

    Parameters
    ----------
    csr        : existing CSR tensor  [rows, cols]
    extra_cols : number of new (empty) columns to append

    Returns
    -------
    New CSR tensor of shape [rows, cols + extra_cols] with identical nonzeros.
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
    """
    Extend a dense 2-D tensor of shape [rows, cols] to [rows, cols + extra_cols]
    by appending zero columns.  Used for non-sparse weights.
    """
    zeros = torch.zeros(t.shape[0], extra_cols, dtype=t.dtype, device=t.device)
    return torch.cat([t, zeros], dim=1)


def _extend_dense_conv(t: torch.Tensor, extra_in_channels: int) -> torch.Tensor:
    """
    Extend a conv weight [out_c, in_c, kH, kW] to [out_c, in_c+extra, kH, kW]
    by appending zero input-channel slices.
    """
    kH, kW = t.shape[2], t.shape[3]
    zeros = torch.zeros(t.shape[0], extra_in_channels, kH, kW,
                        dtype=t.dtype, device=t.device)
    return torch.cat([t, zeros], dim=1)


class RNNAllLayerInfo:
    """
    Metadata for a single per-layer recurrent extension.

    Attributes
    ----------
    name          : parameter name in the state dict
    original_shape: shape before extension
    ff_dim        : original input dimension (cols 0…ff_dim-1 are feedforward)
    rec_dim       : recurrent dimension = output dim (= out for linear, out_c for conv)
    is_conv       : True if this is a conv weight (4-D), False for linear (2-D)
    """
    def __init__(self, name: str, original_shape: torch.Size,
                 ff_dim: int, rec_dim: int, is_conv: bool):
        self.name           = name
        self.original_shape = original_shape
        self.ff_dim         = ff_dim
        self.rec_dim        = rec_dim
        self.is_conv        = is_conv

    def state_shape(self, batch: int = 1, seq: int = 1,
                    spatial_h: int = 1, spatial_w: int = 1) -> tuple:
        """
        Runtime state shape for this layer.

        Linear:  (batch, seq, rec_dim)  — broadcasts over sequence dim
        Conv:    (batch, rec_dim, spatial_h, spatial_w)
        """
        if self.is_conv:
            return (batch, self.rec_dim, spatial_h, spatial_w)
        return (batch, seq, self.rec_dim)

    def __repr__(self) -> str:
        kind = "conv" if self.is_conv else "linear"
        return (f"RNNAllLayerInfo({kind}  {self.name}  "
                f"{tuple(self.original_shape)} → ff={self.ff_dim} rec={self.rec_dim})")


def apply_rnn_all_to_payload(payload: dict) -> dict:
    """
    Extend every eligible linear / conv weight in the payload with a zero-sparse
    recurrent input block, converting each op into a per-layer RNN cell.

    Transform per eligible weight W
    ─────────────────────────────────
    Linear  [out, in]           →  [out, in + out]
    Conv    [out_c, in_c, k, k] →  [out_c, in_c + out_c, k, k]

    The appended columns / channels are empty (no nonzero entries for sparse
    tensors, zero-padded for dense tensors).  File size increases only by the
    additional shape metadata, not by new weight values.

    Returns
    -------
    Modified payload with:
      "sparse_state_dict"  : weights replaced by extended versions
      "rnn_all_layers"     : list of RNNAllLayerInfo dicts (for runtime use)
      "rnn_all_meta"       : summary counts
    """
    ssd = payload.get("sparse_state_dict", payload)

    layer_infos: List[RNNAllLayerInfo] = []
    new_ssd = {}

    n_extended = 0
    n_skipped  = 0
    extra_cols_total = 0

    COL = 56
    header = f"  {'Parameter':<{COL}} {'Original':>16} {'Extended':>16}  rec_dim"
    sep    = "─" * len(header)
    print(f"\n[rnn_all]  Extending layers to per-layer RNN cells …")
    print(f"\n{header}")
    print(sep)

    for name, entry in ssd.items():
        # ── Unwrap entry format from sparse_prune.py ─────────────────────────
        is_sparse_entry = isinstance(entry, dict)
        if is_sparse_entry:
            csr      = entry.get("csr")
            raw      = entry.get("raw")
            shape    = entry.get("shape", csr.shape if csr is not None
                                  else (raw.shape if raw is not None else None))
        elif isinstance(entry, torch.Tensor):
            csr, raw, shape = (entry, None, entry.shape)                 if entry.layout == torch.sparse_csr                 else (None, entry, entry.shape)
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

        is_conv  = len(shape_ts) == 4
        out_dim  = int(shape_ts[0])           # output channels / rows
        in_dim   = int(shape_ts[1])           # input channels / cols
        rec_dim  = out_dim                    # recurrent dim = output dim

        if is_conv:
            # Conv: extended shape [out_c, in_c+rec, kH, kW]
            new_shape = torch.Size([out_dim, in_dim + rec_dim,
                                    shape_ts[2], shape_ts[3]])
        else:
            # Linear: extended shape [out, in+rec]
            new_shape = torch.Size([out_dim, in_dim + rec_dim])

        # ── Perform the extension ─────────────────────────────────────────────
        if csr is not None and csr.layout == torch.sparse_csr and not is_conv:
            # Sparse linear: widen column count, no new nonzeros
            new_tensor = _extend_csr_columns(csr, rec_dim)
            new_entry  = {"csr": new_tensor, "shape": new_shape}
            extra_cols_total += rec_dim
        elif raw is not None and not is_conv:
            # Dense linear (inside sparse_prune entry or plain tensor)
            new_tensor = _extend_dense_columns(raw.float(), rec_dim)
            new_entry  = {"csr": None, "shape": new_shape, "raw": new_tensor}
        elif raw is not None and is_conv:
            new_tensor = _extend_dense_conv(raw.float(), rec_dim)
            new_entry  = {"csr": None, "shape": new_shape, "raw": new_tensor}
        elif csr is not None and is_conv:
            # Conv stored as 2-D CSR (rows=out_c, cols=in_c*kH*kW): treat as
            # linear with in_dim = in_c*kH*kW, extend that
            new_tensor = _extend_csr_columns(csr, rec_dim)
            new_entry  = {"csr": new_tensor, "shape": new_shape}
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

        print(
            f"  {name:<{COL}} {str(tuple(shape_ts)):>16} "
            f"{str(tuple(new_shape)):>16}  {rec_dim}"
        )

    print(sep)
    print(f"\n[rnn_all]  {n_extended} layers extended,  {n_skipped} skipped")
    print(f"           extra recurrent columns (sparse, zero-init): {extra_cols_total:,}")
    print(f"           new nonzero count added: 0  (synaptogenesis grows these)")

    result = dict(payload)
    result["sparse_state_dict"] = new_ssd
    result["rnn_all_layers"] = [
        {
            "name":           li.name,
            "original_shape": list(li.original_shape),
            "ff_dim":         li.ff_dim,
            "rec_dim":        li.rec_dim,
            "is_conv":        li.is_conv,
        }
        for li in layer_infos
    ]
    result["rnn_all_meta"] = {
        "n_extended":         n_extended,
        "n_skipped":          n_skipped,
        "extra_rec_cols":     extra_cols_total,
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

    def __init__(self, layer_metas: List[dict], device: torch.device = torch.device("cpu")):
        self._states: Dict[str, Optional[torch.Tensor]] = {
            m["name"]: None for m in layer_metas
        }
        self._infos: Dict[str, dict] = {m["name"]: m for m in layer_metas}
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
        return (f"RNNAllStateBuffer({len(self._states)} layers, "
                f"{initialized} initialized, device={self.device})")

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
        return {"sparse_state_dict": {k: v for k, v in sd.items()}}

    obj = torch.load(str(p), map_location="cpu", weights_only=False)

    # Recognise sparse_prune.py payload
    if isinstance(obj, dict) and "sparse_state_dict" in obj:
        return obj

    # Plain state dict or checkpoint
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj)
        return {"sparse_state_dict": {k: v for k, v in sd.items()}}

    if hasattr(obj, "state_dict"):
        sd = dict(obj.state_dict())
        return {"sparse_state_dict": {k: v for k, v in sd.items()}}

    sys.exit(f"Cannot interpret loaded object of type {type(obj)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="RNN-fold repeated transformer blocks in a sparse model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input",  nargs="?",  help="sparse_prune.py output (.pt) or raw model")
    p.add_argument("-o", "--output",     default=None,
                   help="Output .pt path  (default: <input>_folded.pt)")
    p.add_argument("--show-groups",      action="store_true",
                   help="Detect and print block groups, then exit")
    p.add_argument("--min-group-size",   type=int, default=2, metavar="N",
                   help="Minimum consecutive identical blocks to fold  (default: 2)")
    p.add_argument("--band-width",       type=int, default=None, metavar="N",
                   help="Override attention band half-width  (default: infer from weight shape)")
    p.add_argument("--also-prune",       action="store_true",
                   help="Run sparse_prune.sparsify_model() on input before folding")
    p.add_argument("--bits",             type=int, default=32, choices=[32,16,8,4,2],
                   help="Quantization bits for pruning threshold (used with --also-prune)")
    p.add_argument("--skip-outputs",     action="store_true",
                   help="Average per-fold-step outputs instead of returning final "
                        "accumulated state (skip-connection outputs / pyramidal pooling)")
    p.add_argument("--rnn-all",          action="store_true",
                   help="Extend every linear/conv weight with a zero-sparse recurrent "
                        "input block, making each op a per-layer RNN cell. "
                        "No new nonzeros at conversion time; synaptogenesis fills them.")
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
        # Re-use sparsify_model internals: get a flat state dict, prune, repack
        flat_sd: Dict[str, torch.Tensor] = {}
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
        flat: Dict[str, torch.Tensor] = {}
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

    # fold is the default operation; --rnn-all is the opt-in extension.
    # --rnn-fold was never added to the CLI parser (pre-existing bug) so
    # the original `if args.rnn_fold or not args.rnn_all:` would crash with
    # AttributeError. The corrected intent: fold unless --rnn-all was the
    # ONLY flag passed (meaning the user explicitly opted out of folding).
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

    out_path = args.output or str(
        Path(args.input).parent / (Path(args.input).stem + out_suffix + ".pt")
    )
    torch.save(working, out_path)
    print(f"\n[saved]  {out_path}")


if __name__ == "__main__":
    main()
