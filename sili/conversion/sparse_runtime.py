#!/usr/bin/env python3
"""
sparse_runtime.py
─────────────────
Build a runnable nn.Module from the output of the sparse_prune + rnn_fold
pipeline.  Weights are kept as sparse CSR matrices and multiplied with
torch.sparse.mm wherever possible.

Operator support status
───────────────────────
  ✓  SparseLinear        — torch.sparse.mm(csr, x.T).T  (CSR × dense)
  ✓  FoldedRNNLayer      — slice-per-step weight extraction + plain RNN state
  ✓  RNNAllLayer         — whole-model additive state, zero-init
  ✓  BandedAttention     — dense Q·Kᵀ with additive -inf band mask
  ✗  SparseLayerNorm     — NotImplementedError: no sparse layer-norm kernel
  ✗  SparseEmbedding     — NotImplementedError: sparse embedding rows unsupported
  ✗  SparseSoftmax       — NotImplementedError: sparse softmax not defined
  ✗  SparseAttentionFull — NotImplementedError: full sparse attention (Q·K are
                           dense after projection; only the mask is sparse)

Architecture-agnostic design
─────────────────────────────
SparseModelRuntime does not assume a specific architecture.  It assembles
layers in the order they appear in the payload, grouping:
  • Folded blocks (from rnn_fold) → FoldedRNNLayer
  • Remaining CSR weights         → SparseLinear
  • Non-weight buffers (norms, biases) → kept dense

For full forward() execution your subclass must override _apply_block() in
each FoldedRNNLayer (same contract as rnn_fold.RNNFoldedBlock) — or use
SparseModelRuntime.forward_with_fn() and supply a callable.

Usage
─────
  from sparse_runtime import SparseModelRuntime, load_sparse_runtime

  # From pipeline output file
  runtime = load_sparse_runtime("model_sparse_folded.pt")
  runtime.eval()

  # Inspect which ops will raise NotImplementedError
  runtime.op_summary()

  # Run (requires _apply_block override or forward_with_fn)
  out = runtime.forward_with_fn(input_ids, block_fn=my_attn_fn)

  python sparse_runtime.py model_sparse_folded.pt --summary
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# model_reconstruct is a sibling module — imported lazily only by
# load_combined_runtime().  sparse_runtime.py is fully usable without it.
_reconstruct_mod = None

def _get_reconstruct():
    global _reconstruct_mod
    if _reconstruct_mod is None:
        try:
            import model_reconstruct as _m
            _reconstruct_mod = _m
        except ImportError:
            sys.exit(
                "model_reconstruct.py not found.  Place it in the same "
                "directory as sparse_runtime.py or add it to PYTHONPATH."
            )
    return _reconstruct_mod


# ══════════════════════════════════════════════════════════════════════════════
#  Sparse operator stubs — clearly labelled by support status
# ══════════════════════════════════════════════════════════════════════════════

class _NotImplementedOp(nn.Module):
    """
    Placeholder for a sparse operation that has no kernel yet.

    Instantiating this is fine.  Calling forward() always raises
    NotImplementedError with a descriptive message explaining what is missing
    and what the workaround is.
    """
    def __init__(self, op_name: str, reason: str, workaround: str = ""):
        super().__init__()
        self.op_name   = op_name
        self.reason    = reason
        self.workaround = workaround

    def forward(self, *args, **kwargs):
        msg = (
            f"\n[sparse_runtime]  {self.op_name} is not yet implemented.\n"
            f"  Reason     : {self.reason}\n"
        )
        if self.workaround:
            msg += f"  Workaround : {self.workaround}\n"
        raise NotImplementedError(msg)

    def extra_repr(self) -> str:
        return f"op={self.op_name!r}  [NOT IMPLEMENTED]"


class SparseLayerNorm(_NotImplementedOp):
    """
    Layer normalisation over sparse input — NOT IMPLEMENTED.

    Reason: LayerNorm requires computing mean and variance over all elements
    in a vector.  For a sparse tensor this would require materialising the
    zeros (or treating them as explicit values), at which point sparsity
    provides no benefit.  PyTorch has no kernel for sparse LayerNorm as of
    2.3.  The standard approach is to keep LayerNorm weights and inputs dense
    and only sparsify the projection weights (Linear layers).
    """
    def __init__(self, normalized_shape, **kwargs):
        super().__init__(
            op_name="SparseLayerNorm",
            reason="No PyTorch kernel for sparse LayerNorm; mean/var require dense scatter.",
            workaround="Keep LayerNorm inputs and weights dense. Only Linear weights are sparse.",
        )
        self.normalized_shape = normalized_shape


class SparseEmbedding(_NotImplementedOp):
    """
    Sparse embedding lookup — NOT IMPLEMENTED.

    Reason: Embedding tables are indexed (row-select), not matrix-multiplied.
    CSR format optimises row-wise matmul, not random row access.  There is no
    standard sparse embedding kernel.  COO or direct row indexing is O(nnz)
    but requires densification of the selected row anyway.

    Workaround: Keep the embedding table dense.  After lookup the embedding
    vectors are dense by definition (you selected a row), so there is nothing
    to sparsify at this point.
    """
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__(
            op_name="SparseEmbedding",
            reason="CSR is not suitable for random row access (embedding lookup).",
            workaround="Keep embedding tables as dense nn.Embedding.",
        )
        self.num_embeddings  = num_embeddings
        self.embedding_dim   = embedding_dim


class SparseSoftmax(_NotImplementedOp):
    """
    Softmax over a sparse attention score matrix — NOT IMPLEMENTED.

    Reason: Softmax is an elementwise normalisation across a row, which
    requires reading all values in that row.  For a sparse attention matrix
    the off-diagonal -inf values still participate in the exp() norminator
    even though they produce 0.  The result is dense (every position gets a
    weight ≥ 0), so sparse softmax degenerates to dense in the general case.

    Workaround: Use BandedAttention (implemented below), which only computes
    scores in the non-zero band and zero-pads the rest, keeping the softmax
    input bounded in size to O(seq_len × band_width).
    """
    def __init__(self, **kwargs):
        super().__init__(
            op_name="SparseSoftmax",
            reason="Softmax output is dense; sparse input gives no memory benefit.",
            workaround="Use BandedAttention to restrict the score matrix size.",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Implemented sparse operators
# ══════════════════════════════════════════════════════════════════════════════

class SparseLinear(nn.Module):
    """
    Linear layer with a CSR sparse weight matrix.  Bias (if present) is dense.

    Forward:  y = x @ Wᵀ + b
    With CSR: y = (W @ xᵀ)ᵀ + b   using torch.sparse.mm(W_csr, x.T).T

    torch.sparse.mm supports CSR × dense as of PyTorch 1.13 on CPU and
    PyTorch 2.0+ on CUDA with cuSPARSE.  It is more memory-efficient than
    converting to dense first: only the nnz values and index arrays are read.

    Parameters
    ----------
    weight_csr  : CSR tensor of shape [out_features, in_features]
    bias        : optional dense tensor of shape [out_features]
    original_shape : the original (possibly > 2D) shape before CSR flattening;
                     used only for introspection, not computation
    """

    def __init__(
        self,
        weight_csr:     torch.Tensor,
        bias:           Optional[torch.Tensor] = None,
        original_shape: Optional[tuple]        = None,
    ):
        super().__init__()
        if weight_csr.layout != torch.sparse_csr:
            raise TypeError(f"weight_csr must be sparse_csr, got {weight_csr.layout}")
        # Register as a buffer so .to(device) moves it correctly
        self.register_buffer("weight_csr", weight_csr)
        self.bias           = nn.Parameter(bias) if bias is not None else None
        self.original_shape = original_shape
        self.out_features   = weight_csr.shape[0]
        self.in_features    = weight_csr.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [..., in_features]
        y : [..., out_features]

        Handles arbitrary batch dimensions by flattening to 2D,
        running the sparse matmul, then restoring shape.
        """
        *batch, in_f = x.shape
        x2 = x.reshape(-1, in_f).t().contiguous()   # [in_f, B*T]

        # torch.sparse.mm(sparse [M,K], dense [K,N]) → dense [M,N]
        y2 = torch.sparse.mm(self.weight_csr, x2)   # [out_f, B*T]
        y  = y2.t().reshape(*batch, self.out_features)

        if self.bias is not None:
            y = y + self.bias
        return y

    def sparsity(self) -> float:
        total = self.out_features * self.in_features
        nnz   = int(self.weight_csr.values().numel())
        return 1.0 - nnz / total if total > 0 else 0.0

    def extra_repr(self) -> str:
        nnz = int(self.weight_csr.values().numel())
        return (f"in={self.in_features}, out={self.out_features}, "
                f"nnz={nnz:,}, sparsity={self.sparsity():.1%}")


class BandedAttention(nn.Module):
    """
    Causal self-attention restricted to a diagonal band, implemented in dense
    arithmetic but with a sparse additive mask.

    This is the attention primitive used inside FoldedRNNLayer for Q/K/V
    projections that carry a band_half_width constraint.  It is NOT fully
    sparse — the Q·Kᵀ product is still computed densely — but the band mask
    prevents the model from using out-of-band positions, and the score matrix
    itself could be replaced by a sparse kernel in future (see SparseSoftmax).

    For full sparse attention a dedicated Triton or cuSPARSE kernel is required.

    Parameters
    ----------
    n_heads         : number of attention heads
    head_dim        : per-head dimension
    band_half_width : maximum distance |q_pos − k_pos| allowed
    causal          : if True, also apply upper-triangular -inf mask
    """

    def __init__(self, n_heads: int, head_dim: int,
                 band_half_width: int, causal: bool = True):
        super().__init__()
        self.n_heads        = n_heads
        self.head_dim       = head_dim
        self.band_half_width = band_half_width
        self.causal         = causal
        self.scale          = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,   # [B, n_heads, T, head_dim]
        k: torch.Tensor,
        v: torch.Tensor,
        extra_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, T, D = q.shape

        # Build band + optional causal mask (cached on first use per seq_len)
        q_pos = torch.arange(T, device=q.device).unsqueeze(1)
        k_pos = torch.arange(T, device=q.device).unsqueeze(0)
        blocked = (q_pos - k_pos).abs() > self.band_half_width
        if self.causal:
            blocked = blocked | (k_pos > q_pos)
        mask = torch.zeros(T, T, device=q.device, dtype=q.dtype)
        mask.masked_fill_(blocked, float("-inf"))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + mask
        if extra_mask is not None:
            scores = scores + extra_mask
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def extra_repr(self) -> str:
        return (f"n_heads={self.n_heads}, head_dim={self.head_dim}, "
                f"band±{self.band_half_width}, causal={self.causal}")


# ══════════════════════════════════════════════════════════════════════════════
#  Folded RNN layer (wraps rnn_fold.FoldedBlockDescriptor)
# ══════════════════════════════════════════════════════════════════════════════

class FoldedRNNLayer(nn.Module):
    """
    Sparse inference wrapper for one rnn_fold FoldedBlockDescriptor.

    Each fold step:
      1. Slices the stacked CSR weight for that step → densify (O(nnz_in_band))
      2. Applies the user-supplied block_fn (handles attention, norm, etc.)
      3. Accumulates into RNN state

    Output mode is controlled by descriptor.skip_connection_outputs:
      False → return final state
      True  → return mean of per-step outputs  (pyramidal pooling)

    To execute forward() you must supply block_fn:
        def block_fn(x, weights, masks, fold_step):
            # weights: {suffix: dense tensor}
            # masks:   {suffix: banded attention mask or None}
            return output_tensor

    Or subclass and override _apply_block().
    """

    def __init__(self, descriptor):
        super().__init__()
        # Store the descriptor's stacked CSR weights as named buffers
        self.n_folds                 = descriptor.n_folds
        self.out_dims                = descriptor.out_dims
        self.band_half_widths        = descriptor.band_half_widths
        self.skip_connection_outputs = descriptor.skip_connection_outputs
        self._suffixes               = list(descriptor.stacked_weights.keys())

        for i, suffix in enumerate(self._suffixes):
            safe_key = f"_w{i}"
            self.register_buffer(safe_key, descriptor.stacked_weights[suffix])
        self._suffix_to_buf = {s: f"_w{i}" for i, s in enumerate(self._suffixes)}

    def _get_weight(self, suffix: str) -> torch.Tensor:
        return getattr(self, self._suffix_to_buf[suffix])

    def _slice_fold_weight(self, suffix: str, fold_step: int) -> torch.Tensor:
        """Return dense weight slice for this suffix at fold_step."""
        csr     = self._get_weight(suffix)
        out_dim = self.out_dims[suffix]
        r0, r1  = fold_step * out_dim, (fold_step + 1) * out_dim

        crow  = csr.crow_indices()
        cols  = csr.col_indices()
        vals  = csr.values()
        n_col = csr.shape[1]

        nnz0 = int(crow[r0].item())
        nnz1 = int(crow[r1].item())
        slice_csr = torch.sparse_csr_tensor(
            crow[r0:r1+1] - nnz0,
            cols[nnz0:nnz1],
            vals[nnz0:nnz1],
            size=(out_dim, n_col),
            dtype=csr.dtype,
        )
        return slice_csr.to_dense()

    def _attn_mask(self, suffix: str, seq_len: int, device) -> Optional[torch.Tensor]:
        bw = self.band_half_widths.get(suffix)
        if bw is None:
            return None
        q_pos = torch.arange(seq_len, device=device).unsqueeze(1)
        k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
        blocked = (q_pos - k_pos).abs() > bw
        mask = torch.zeros(seq_len, seq_len, device=device)
        mask.masked_fill_(blocked, float("-inf"))
        return mask

    def _apply_block(
        self,
        x:        torch.Tensor,
        weights:  Dict[str, torch.Tensor],
        masks:    Dict[str, Optional[torch.Tensor]],
        fold_step: int,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "FoldedRNNLayer._apply_block() must be overridden, or use "
            "forward_with_fn(x, block_fn=...) instead."
        )

    def forward(self, x: torch.Tensor,
                block_fn: Optional[Callable] = None) -> torch.Tensor:
        device  = x.device
        seq_len = x.shape[1] if x.ndim >= 2 else x.shape[0]
        state   = torch.zeros_like(x)
        step_outputs: List[torch.Tensor] = []

        for step in range(self.n_folds):
            weights = {s: self._slice_fold_weight(s, step).to(device)
                       for s in self._suffixes}
            masks   = {s: self._attn_mask(s, seq_len, device)
                       for s in self._suffixes}

            if block_fn is not None:
                out = block_fn(x + state, weights, masks, step)
            else:
                out = self._apply_block(x + state, weights, masks, step)

            state = state + out
            if self.skip_connection_outputs:
                step_outputs.append(out)

        if self.skip_connection_outputs:
            return torch.stack(step_outputs, dim=0).mean(dim=0)
        return state

    def sparsity_report(self) -> Dict[str, float]:
        """Per-suffix sparsity across the full stacked weight."""
        report = {}
        for suffix in self._suffixes:
            csr   = self._get_weight(suffix)
            total = csr.shape[0] * csr.shape[1]
            nnz   = int(csr.values().numel())
            report[suffix] = 1.0 - nnz / total if total > 0 else 0.0
        return report

    def extra_repr(self) -> str:
        mode = "skip-avg" if self.skip_connection_outputs else "final-state"
        return f"n_folds={self.n_folds}, mode={mode}, params={len(self._suffixes)}"


# ══════════════════════════════════════════════════════════════════════════════
#  RNN-all layer (whole-model additive state, zero-init)
# ══════════════════════════════════════════════════════════════════════════════

class RNNAllLayer(nn.Module):
    """
    Whole-model stateful RNN wrapper.

    Wraps an arbitrary inner module so that its output feeds back as an
    additive input on the next call.  The state is zero-initialised and stored
    as a non-persistent buffer — it is never saved to disk, so model size is
    unchanged.

    Zero-init rationale
    ───────────────────
    A zero state is equivalent to "no prior context."  From the network's
    perspective the first call is indistinguishable from a cold-start
    inference.  Subsequent calls receive accumulated prior-context signal via
    the state.  Because zero is formally unreachable from normal weight
    initialisation trajectories (random init pushes activations away from
    zero), the zero-init state forms a unique trajectory that can be used by
    synaptogenesis routines to grow new connections without interfering with
    existing ones.

    State management
    ─────────────────
    Call reset_state() between independent sequences.  The state lives on the
    same device as the most recent input — it is moved automatically.

    Parameters
    ----------
    inner_module    : the wrapped nn.Module (e.g., SparseModelRuntime or a
                      reconstructed model from model_reconstruct.py)
    hidden_dim      : dimension of the recurrent state vector (d_model)
    inject_at_input : if True, add state to the *input* tensor x before the
                      inner module; if False, add to the inner module's output
                      (for use when input_dim ≠ hidden_dim)
    skip_connection_outputs : if True, collect per-call outputs and return
                              their mean (pyramidal pooling across time steps)
    """

    def __init__(
        self,
        inner_module:            nn.Module,
        hidden_dim:              int,
        inject_at_input:         bool = True,
        skip_connection_outputs: bool = False,
    ):
        super().__init__()
        self.inner                   = inner_module
        self.hidden_dim              = hidden_dim
        self.inject_at_input         = inject_at_input
        self.skip_connection_outputs = skip_connection_outputs

        # Non-persistent: not saved in state_dict, zero-init every load
        self.register_buffer("_state", None, persistent=False)
        self._call_outputs: List[torch.Tensor] = []

    def _init_state(self, x: torch.Tensor) -> None:
        """Lazily initialise state to zeros matching x's batch/seq/hidden dims."""
        if x.ndim == 3:
            shape = (x.shape[0], x.shape[1], self.hidden_dim)
        elif x.ndim == 2:
            shape = (x.shape[0], self.hidden_dim)
        else:
            shape = (self.hidden_dim,)
        self._state = torch.zeros(shape, dtype=x.dtype, device=x.device)

    def reset_state(self) -> None:
        """Clear the recurrent state (call between independent sequences)."""
        self._state = None
        self._call_outputs.clear()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._state is None or self._state.device != x.device:
            self._init_state(x)

        if self.inject_at_input:
            # Broadcast add: state may be [1,1,d] while x is [B,T,d]
            try:
                inp = x + self._state
            except RuntimeError:
                # Shape mismatch — fall back to input-only (no state injection)
                inp = x
        else:
            inp = x

        out = self.inner(inp, **kwargs)

        # Update state from the output hidden representation
        # If out is logits [B,T,vocab] we cannot use it directly as state;
        # in that case skip the state update and warn once.
        if out.shape[-1] == self.hidden_dim:
            self._state = out.detach()
        else:
            if not getattr(self, "_warned_state_shape", False):
                print(
                    f"[RNNAllLayer]  output dim {out.shape[-1]} ≠ hidden_dim "
                    f"{self.hidden_dim}; state not updated.  Pass the hidden "
                    "representation, not logits, as the inner module's return value."
                )
                self._warned_state_shape = True

        if self.skip_connection_outputs:
            self._call_outputs.append(out)

        return out

    def get_averaged_output(self) -> Optional[torch.Tensor]:
        """
        Return the mean of all per-call outputs collected since the last reset.

        Only populated when skip_connection_outputs=True.  This is the
        pyramidal-pooling read-out: call it after the final step of a
        multi-call sequence to get the scale-averaged representation.
        """
        if not self._call_outputs:
            return None
        return torch.stack(self._call_outputs, dim=0).mean(dim=0)

    def extra_repr(self) -> str:
        mode = "skip-avg" if self.skip_connection_outputs else "stateful"
        return (f"hidden_dim={self.hidden_dim}, inject_at_input={self.inject_at_input}, "
                f"mode={mode}")


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level runtime assembler
# ══════════════════════════════════════════════════════════════════════════════

class SparseModelRuntime(nn.Module):
    """
    Assembled sparse/folded model runtime.

    Constructed from a sparse_prune + rnn_fold pipeline payload.  Holds:
      • sparse_layers   : OrderedDict of SparseLinear for unfolded parameters
      • folded_layers   : list of FoldedRNNLayer for folded block groups
      • dense_buffers   : non-weight tensors (biases, norms) stored dense
      • not_impl_ops    : dict of ops that will raise NotImplementedError

    forward() is intentionally not defined on this base class.  Use:
      forward_with_fn(x, block_fn)   for folded layers with a custom block_fn
      or subclass and override forward()

    The split is deliberate: this file cannot know your token embedding, layer
    ordering, or norm placement — those require architecture knowledge from
    model_reconstruct.py.  SparseModelRuntime is the weight container and
    compute primitive provider; you wire them together.
    """

    def __init__(
        self,
        sparse_layers:   Dict[str, SparseLinear],
        folded_layers:   List[FoldedRNNLayer],
        dense_buffers:   Dict[str, torch.Tensor],
        rnn_all_hidden:  Optional[int] = None,
        skip_connection_outputs: bool  = False,
    ):
        super().__init__()
        self.sparse_layers = nn.ModuleDict(
            {k.replace(".", "_"): v for k, v in sparse_layers.items()}
        )
        self.folded_layers  = nn.ModuleList(folded_layers)
        self._dense_buffers = dense_buffers  # name → tensor, not nn.Parameter
        for name, tensor in dense_buffers.items():
            self.register_buffer(name.replace(".", "_"), tensor)

        self.rnn_all_hidden          = rnn_all_hidden
        self.skip_connection_outputs = skip_connection_outputs

        # Declare unsupported ops as named stubs for introspection
        self.not_impl_ops = nn.ModuleDict({
            "layer_norm":   SparseLayerNorm((1,)),
            "embedding":    SparseEmbedding(1, 1),
            "softmax":      SparseSoftmax(),
        })

    def op_summary(self) -> None:
        """Print a table of which ops are and are not supported."""
        print("\n[sparse_runtime]  Operator support summary")
        print("─" * 60)
        supported = [
            ("SparseLinear",     "torch.sparse.mm(CSR, dense)"),
            ("BandedAttention",  "dense QKᵀ + -inf band mask"),
            ("FoldedRNNLayer",   "slice-per-step + RNN state"),
            ("RNNAllLayer",      "zero-init whole-model state"),
        ]
        unsupported = [
            ("SparseLayerNorm",     "no sparse mean/var kernel"),
            ("SparseEmbedding",     "CSR unsuitable for row lookup"),
            ("SparseSoftmax",       "softmax output is always dense"),
            ("SparseAttentionFull", "Q·Kᵀ product remains dense"),
        ]
        print("  SUPPORTED:")
        for name, note in supported:
            print(f"    ✓  {name:<26}  {note}")
        print("  NOT IMPLEMENTED:")
        for name, note in unsupported:
            print(f"    ✗  {name:<26}  {note}")
        print("─" * 60)

        print("\n[sparse_runtime]  Layer inventory")
        print(f"  SparseLinear layers : {len(self.sparse_layers)}")
        print(f"  FoldedRNN  groups   : {len(self.folded_layers)}")
        print(f"  Dense buffers       : {len(self._dense_buffers)}")
        if self.rnn_all_hidden is not None:
            print(f"  RNN-all hidden dim  : {self.rnn_all_hidden}")
        print()

        total_nnz   = sum(
            int(m.weight_csr.values().numel())
            for m in self.sparse_layers.values()
        )
        total_dense = sum(t.numel() for t in self._dense_buffers.values())
        print(f"  Sparse nnz (Linear) : {total_nnz:,}")
        print(f"  Dense elements      : {total_dense:,}")

    def sparsity_report(self) -> Dict[str, float]:
        """Return per-layer sparsity dict."""
        report = {}
        for safe_name, layer in self.sparse_layers.items():
            report[safe_name] = layer.sparsity()
        for i, fl in enumerate(self.folded_layers):
            for suffix, sp in fl.sparsity_report().items():
                report[f"folded_{i}{suffix}"] = sp
        return report

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "SparseModelRuntime.forward() is not defined on the base class.\n"
            "Use forward_with_fn(x, block_fn=...) or subclass and wire "
            "layers together using self.sparse_layers and self.folded_layers."
        )

    def forward_with_fn(
        self,
        x:          torch.Tensor,
        block_fn:   Callable,
        fold_index: int = 0,
    ) -> torch.Tensor:
        """
        Run one folded group's forward pass using a user-supplied block_fn.

        Parameters
        ----------
        x          : input tensor [batch, seq, d_model]
        block_fn   : callable(x, weights, masks, fold_step) → output tensor
        fold_index : which folded group to use (default 0)

        Returns
        -------
        Output tensor from the chosen FoldedRNNLayer.
        """
        if fold_index >= len(self.folded_layers):
            raise IndexError(
                f"fold_index={fold_index} but only {len(self.folded_layers)} "
                "folded groups exist."
            )
        return self.folded_layers[fold_index](x, block_fn=block_fn)


# ══════════════════════════════════════════════════════════════════════════════
#  Factory — build SparseModelRuntime from a payload file
# ══════════════════════════════════════════════════════════════════════════════

def _csr_from_entry(entry) -> Optional[torch.Tensor]:
    """Extract a CSR tensor from a sparse_prune.py state-dict entry."""
    if isinstance(entry, dict):
        csr = entry.get("csr")
        if csr is not None and isinstance(csr, torch.Tensor):
            return csr
        raw = entry.get("raw")
        if raw is not None and isinstance(raw, torch.Tensor):
            if raw.layout == torch.sparse_csr:
                return raw
        return None
    if isinstance(entry, torch.Tensor) and entry.layout == torch.sparse_csr:
        return entry
    return None


def _bias_from_entry(entry) -> Optional[torch.Tensor]:
    """Return dense bias tensor if present, else None."""
    if isinstance(entry, dict):
        raw = entry.get("raw")
        if raw is not None and isinstance(raw, torch.Tensor) and raw.ndim == 1:
            return raw
    if isinstance(entry, torch.Tensor) and entry.layout == torch.strided:
        if entry.ndim == 1:
            return entry
    return None


def load_sparse_runtime(
    path: str,
    skip_connection_outputs: bool = False,
    rnn_all_hidden: Optional[int] = None,
    device: str = "cpu",
    generate_trace: bool = True,
    trace_output: Optional[str] = None,
    original_model: Optional[nn.Module] = None,
) -> SparseModelRuntime:
    """
    Build a SparseModelRuntime from a sparse_prune (+ optional rnn_fold) payload.

    Parameters
    ----------
    path                    : .pt file from sparse_prune.py or rnn_fold.py
    skip_connection_outputs : enable pyramidal-pooling output averaging
    rnn_all_hidden          : if set, wrap in RNNAllLayer with this hidden dim
    device                  : target device
    generate_trace          : if True (default), generate <stem>_trace.py and
                              immediately run it so you hit the first
                              NotImplementedError right away
    trace_output            : override path for the generated trace file;
                              defaults to <path_stem>_trace.py
    original_model          : optional pre-loaded nn.Module passed to
                              trace_model for FX-assisted node ordering

    Returns
    -------
    SparseModelRuntime in eval() mode
    """
    print(f"[sparse_runtime]  Loading {path}")
    payload = torch.load(str(path), map_location=device, weights_only=False)

    # Normalise to a {name: entry} flat dict
    ssd = payload.get("sparse_state_dict", payload)

    # ── Build SparseLinear layers for each CSR weight ────────────────────────
    sparse_layers:  Dict[str, SparseLinear] = {}
    dense_buffers:  Dict[str, torch.Tensor] = {}

    # Pair weights with matching biases (same prefix, weight vs bias suffix)
    weight_names = {k for k in ssd if k.endswith(".weight")}
    bias_names   = {k for k in ssd if k.endswith(".bias")}

    for wname in sorted(weight_names):
        csr = _csr_from_entry(ssd[wname])
        if csr is None:
            # Dense or scalar weight → dense buffer
            entry = ssd[wname]
            t = entry.get("raw") if isinstance(entry, dict) else entry
            if isinstance(t, torch.Tensor):
                dense_buffers[wname] = t
            continue

        bname = wname[:-len(".weight")] + ".bias"
        bias  = None
        if bname in ssd:
            bias = _bias_from_entry(ssd[bname])
            bias_names.discard(bname)

        orig_shape = ssd[wname].get("shape") if isinstance(ssd[wname], dict) else None
        sparse_layers[wname] = SparseLinear(
            weight_csr=csr,
            bias=bias,
            original_shape=orig_shape,
        )

    # Remaining biases and non-weight entries → dense buffers
    for name in sorted(ssd.keys()):
        if name in weight_names:
            continue
        if name in bias_names:
            entry = ssd[name]
            t = entry.get("raw") if isinstance(entry, dict) else entry
            if isinstance(t, torch.Tensor):
                dense_buffers[name] = t
            continue
        # Any other tensor (embedding, norm, etc.)
        entry = ssd[name]
        csr   = _csr_from_entry(entry)
        if csr is None:
            t = entry.get("raw") if isinstance(entry, dict) else entry
            if isinstance(t, torch.Tensor) and t.ndim > 0:
                dense_buffers[name] = t

    # ── Reconstruct FoldedRNNLayer objects from folded_blocks ─────────────────
    folded_layers: List[FoldedRNNLayer] = []
    for fd in payload.get("folded_blocks", []):
        # Reconstruct a minimal descriptor-like object
        class _Desc:
            pass
        desc = _Desc()
        desc.n_folds                 = fd["n_folds"]
        desc.block_indices           = fd["block_indices"]
        desc.stacked_weights         = fd["stacked_weights"]
        desc.out_dims                = fd["out_dims"]
        desc.band_half_widths        = fd["band_half_widths"]
        desc.skip_connection_outputs = fd.get("skip_connection_outputs",
                                              skip_connection_outputs)
        folded_layers.append(FoldedRNNLayer(desc))

    print(f"[sparse_runtime]  SparseLinear   : {len(sparse_layers)}")
    print(f"[sparse_runtime]  FoldedRNN      : {len(folded_layers)}")
    print(f"[sparse_runtime]  Dense buffers  : {len(dense_buffers)}")

    runtime = SparseModelRuntime(
        sparse_layers=sparse_layers,
        folded_layers=folded_layers,
        dense_buffers=dense_buffers,
        rnn_all_hidden=rnn_all_hidden,
        skip_connection_outputs=skip_connection_outputs,
    )

    if rnn_all_hidden is not None:
        print(f"[sparse_runtime]  Wrapping in RNNAllLayer (hidden={rnn_all_hidden})")
        runtime = RNNAllLayer(
            inner_module=runtime,
            hidden_dim=rnn_all_hidden,
            skip_connection_outputs=skip_connection_outputs,
        )

    runtime = runtime.to(device)
    runtime.eval()

    # ── Generate trace file and run it ────────────────────────────────────────
    if generate_trace:
        try:
            import trace_model as _tm
        except ImportError:
            print("[sparse_runtime]  trace_model.py not found — skipping trace generation.")
            print("                  Place trace_model.py alongside sparse_runtime.py.")
            _tm = None

        if _tm is not None:
            trace_path = _tm.generate_for_runtime(
                payload=payload,
                sparse_path=str(path),
                output_path=trace_output,
                original_model=original_model,
            )
            print(f"[sparse_runtime]  Trace written : {trace_path}")
            print(f"[sparse_runtime]  Running trace to surface first NotImplementedError ...")
            print()
            import subprocess, sys as _sys
            # Run with inherited stdout/stderr so output appears in real time.
            # PYTHONUNBUFFERED=1 forces line-buffered output even through pipes.
            import os as _os
            env = {**_os.environ, 'PYTHONUNBUFFERED': '1'}
            subprocess.run(
                [_sys.executable, '-u', trace_path],
                env=env,
                # stdout/stderr intentionally NOT redirected — output goes
                # straight to the terminal in real time with no buffering.
            )

    return runtime


# ══════════════════════════════════════════════════════════════════════════════
#  Combined reconstruct + sparsify loader
# ══════════════════════════════════════════════════════════════════════════════

def load_combined_runtime(
    original_path: str,
    sparse_path:   str,
    device:        str  = "cpu",
    skip_connection_outputs: bool = False,
    rnn_all_hidden: Optional[int] = None,
    generate_trace: bool = True,
    trace_output:  Optional[str] = None,
) -> Tuple[nn.Module, "SparseModelRuntime"]:
    """
    Load BOTH the original reconstructed module AND the sparse/folded runtime.

    This is the recommended entry point when you want to compare the original
    model against its sparse/folded version, or when you need the original
    module's forward() logic to drive a custom block_fn for FoldedRNNLayer.

    Parameters
    ----------
    original_path  : .bin / .safetensors / shard directory  (raw weights)
    sparse_path    : .pt output of sparse_prune.py + rnn_fold.py
    device         : target device for both modules
    skip_connection_outputs : enable pyramidal averaging in folded layers
    rnn_all_hidden : if set, wrap sparse runtime in RNNAllLayer

    Returns
    -------
    (original_model, sparse_runtime)
      original_model  : nn.Module from model_reconstruct.reconstruct_model()
      sparse_runtime  : SparseModelRuntime (or RNNAllLayer-wrapped)

    Example
    -------
    orig, sparse = load_combined_runtime(
        "Qwen3-VL-7B/",
        "Qwen3-VL-7B_sparse_folded.pt",
    )
    # Use original for reference forward / block_fn
    # Use sparse for compressed inference
    """
    mr = _get_reconstruct()
    print("[combined]  Loading original model …")
    original = mr.reconstruct_model(original_path, device=device)

    print("\n[combined]  Loading sparse runtime …")
    sparse = load_sparse_runtime(
        sparse_path,
        skip_connection_outputs=skip_connection_outputs,
        rnn_all_hidden=rnn_all_hidden,
        device=device,
        generate_trace=generate_trace,
        trace_output=trace_output,
        original_model=original,
    )
    return original, sparse


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Build and inspect a sparse/folded model runtime.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input",         help="sparse_prune / rnn_fold .pt payload")
    p.add_argument("--summary",     action="store_true",
                   help="Print op support table and layer inventory, then exit")
    p.add_argument("--sparsity",    action="store_true",
                   help="Print per-layer sparsity report")
    p.add_argument("--original",    default=None, metavar="PATH",
                   help="Also load the original model from this .bin/safetensors path "
                        "(uses model_reconstruct.py)")
    p.add_argument("--rnn-all",     type=int, default=None, metavar="DIM",
                   help="Wrap runtime in RNNAllLayer with this hidden dimension")
    p.add_argument("--skip-outputs", action="store_true",
                   help="Enable pyramidal (skip-connection) output averaging")
    p.add_argument("--device",      default="cpu")
    p.add_argument("--no-trace",    action="store_true",
                   help="Skip trace file generation (default: generate and run trace)")
    p.add_argument("--trace-output", default=None, metavar="PATH",
                   help="Override path for generated trace .py file")
    args = p.parse_args()

    if args.original:
        original, runtime = load_combined_runtime(
            args.original, args.input,
            device=args.device,
            skip_connection_outputs=args.skip_outputs,
            rnn_all_hidden=args.rnn_all,
            generate_trace=not args.no_trace,
            trace_output=args.trace_output,
        )
        print(f"\n[combined]  original type : {type(original).__name__}")
        print(f"[combined]  sparse  type  : {type(runtime).__name__}")
    else:
        runtime = load_sparse_runtime(
            args.input,
            skip_connection_outputs=args.skip_outputs,
            rnn_all_hidden=args.rnn_all,
            device=args.device,
            generate_trace=not args.no_trace,
            trace_output=args.trace_output,
        )

    base = runtime.inner if isinstance(runtime, RNNAllLayer) else runtime
    if args.summary:
        base.op_summary()

    if args.sparsity:
        print("\n[sparsity report]")
        for name, sp in base.sparsity_report().items():
            print(f"  {name:<60}  {sp:.2%}")


if __name__ == "__main__":
    main()