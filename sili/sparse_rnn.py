"""
sili.sparse_rnn — sparse RNN layers.

All layers are Module subclasses. C++-backed layers (DISLDOLayer, SISLDOLayer)
carry no Tensor parameters — their weights live in C++ and are updated via their
own optimizer. parameters() returns [] for these.

Forward flow in SparseRNNCell
------------------------------
    obs   (Tensor) ──[DISLDOLayer]────────────────────────────► Tensor
    state (Tensor) ──[CSR.from_dense]──[SISLDOLayer]──────────► Tensor
                                                  sum ──[EnergyDynamics]──► h_out

    h_out.data → CSR cached inside the cell for the next step's recurrent pass.

Backward / training
--------------------
    BPTT=1 (default):
        agent.train_step(obs)          # detaches state, forward, loss.backward(), step()

    Multi-step BPTT:
        for obs in episode:
            action = agent(obs)        # state stays in graph
            ...
        loss.backward()
        agent.step()
        agent.state = agent.state.detach()

    The C++ weight update (step()) is independent of the autograd graph.
    aux_loss.backward() populates gradients on the Tensor path so that
    DISLDOLayer / SISLDOLayer._backward can call _acc() and accumulate
    into the C++ weight grad buffers.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np
import sili._cpu as _cpu

from sili.module import Module
from sili.tensor import Tensor, _acc
from sili.energy import EnergyDynamics, BranchingRatioTracker, EMABranchingRatioTracker


# ══════════════════════════════════════════════════════════════════════════════
#  CSR activation format
# ══════════════════════════════════════════════════════════════════════════════

class CSR(NamedTuple):
    """Sparse row-major activation tensor."""
    ptrs:    np.ndarray   # int32   [rows+1]
    indices: np.ndarray   # int32   [nnz]
    values:  np.ndarray   # float32 [nnz]
    rows:    int
    cols:    int

    @property
    def nnz(self) -> int:
        return len(self.indices)

    @staticmethod
    def from_dense(x: np.ndarray, p: float = 0.03, num_cpus: int = 4) -> "CSR":
        """
        Build CSR keeping the top-k entries by magnitude, k = max(1, round(cols * p)).
        x : float32 [cols] or [batch, cols]

        Independent top-k -- use ONLY when no prior sparsification decision
        already exists for this activation (i.e. true step-0, before any
        EnergyDynamics gate has run). Once a gate decision exists, prefer
        from_kept_indices below: re-deriving sparsity independently here can
        disagree with what the gate already decided.
        """
        x2d = x[np.newaxis, :] if x.ndim == 1 else x
        x2d = np.asarray(x2d, dtype=np.float32)
        k   = max(1, int(x2d.shape[1] * p))
        ptrs, indices, values = _cpu.dense_to_top_k_csr(x2d, k, num_cpus)
        return CSR(ptrs, indices, values, rows=x2d.shape[0], cols=x2d.shape[1])

    @staticmethod
    def from_kept_indices(kept_indices: np.ndarray, values_source: np.ndarray,
                          cols: int) -> "CSR":
        """
        Build a single-row CSR directly from an already-decided set of kept
        column indices (e.g. EnergyDynamics.kept_indices) and a dense array
        to pull values from at those indices.

        This is the "unify the two sparsification passes" path (see
        sili.energy._apply_energy_dynamics's kept_indices docstring): when a
        prior energy-gating decision already exists for this activation,
        build the CSR from THAT decision instead of from_dense's independent
        top-k, which could disagree with it.

        kept_indices  : int array, sorted ascending, indices into values_source
        values_source : float32 [cols] -- values are read from HERE at
                        kept_indices. Pass the PRE-gating activation (e.g.
                        the h fed INTO an EnergyDynamics call), not that
                        call's h_out -- h_out's fired/shutoff positions hold
                        energy-derived constants (2.0, e+2), not the real
                        activation magnitude the gate decided to keep.
        cols          : full (dense) width this row represents
        """
        kept_indices  = np.asarray(kept_indices, dtype=np.int32)
        values_source = np.asarray(values_source, dtype=np.float32).ravel()
        values        = values_source[kept_indices]
        ptrs          = np.array([0, len(kept_indices)], dtype=np.int32)
        return CSR(ptrs, kept_indices, values, rows=1, cols=cols)

    def to_dense(self) -> np.ndarray:
        """Reconstruct dense float32 [rows, cols]."""
        out = np.zeros((self.rows, self.cols), dtype=np.float32)
        p   = np.asarray(self.ptrs)
        idx = np.asarray(self.indices)
        v   = np.asarray(self.values)
        for r in range(self.rows):
            out[r, idx[p[r]:p[r+1]]] = v[p[r]:p[r+1]]
        return out

    def as_tensor(self, backend=None) -> "Tensor":
        """Wrap this CSR as a Tensor. The CSR is the data; grad will be dense."""
        from sili.tensor import get_backend
        b = backend or get_backend("cpu")
        return Tensor(self, backend=b)


# ══════════════════════════════════════════════════════════════════════════════
#  Shared base for C++-backed sparse layers
# ══════════════════════════════════════════════════════════════════════════════

class _SparseLayerBase(Module):
    """
    Module base for layers whose weights live in C++.
    parameters() returns [] — nothing participates in Tensor autograd.
    """

    def parameters(self) -> list:
        return []

    @property
    def in_features(self)  -> int: return self._c.n_inputs
    @property
    def out_features(self) -> int: return self._c.n_outputs
    @property
    def nnz(self)          -> int: return self._c.nnz
    @property
    def num_cpus(self)     -> int: return self._c.num_cpus

    @property
    def out_degree(self) -> int: return self._c.out_degree

    @property
    def weights(self)    -> np.ndarray: return self._c.weights_vals
    @property
    def importance(self) -> np.ndarray: return self._c.importance
    @property
    def indices(self)    -> np.ndarray: return self._c.indices
    @property
    def ptrs(self)       -> np.ndarray: return self._c.ptrs

    @property
    def neuron_input_accum(self) -> np.ndarray: return self._c.neuron_input_accum
    @property
    def neuron_grad_accum(self)  -> np.ndarray: return self._c.neuron_grad_accum

    def step(self, lr: float):
        self._c.optim_weights(lr)

    def decay(self, rate: float):
        self._c.decay_importance(rate)

    def synaptogenesis(self, k: int, lr: float, importance_beta: float, max_weights: int):
        self._c.build_probes(k)
        self._c.optim_synaptogenesis(lr, importance_beta, max_weights)
        self._c.zero_accum()

    def state_dict(self) -> dict:
        return {
            "ptrs":       np.array(self.ptrs),
            "indices":    np.array(self.indices),
            "weights":    np.array(self.weights),
            "importance": np.array(self.importance),
        }

    def load_state_dict(self, d: dict):
        self._c.load_weights(
            d["ptrs"]      .astype(np.int32),
            d["indices"]   .astype(np.int32),
            d["weights"]   .astype(np.float32),
            d["importance"].astype(np.float32),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  DISLDOLayer — Dense Input, Sparse Linear, Dense Output
# ══════════════════════════════════════════════════════════════════════════════

class DISLDOLayer(_SparseLayerBase):
    """Dense observation → state contribution. No CSR on the input side."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                 num_cpus: int = 4, solidify: float = 0.01):
        self._c = _cpu.DISLDOLayer(in_features, out_features, max_weights, num_cpus, solidify)

    def forward(self, x) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float32))
        x_np   = np.asarray(x.data, dtype=np.float32)[np.newaxis, :]
        out_np = self._c.forward(x_np).squeeze(0)
        out    = Tensor(out_np, _children=(x,), _op="disldo", backend=x.backend)

        def _bwd():
            if out.grad is not None:
                dy = np.asarray(out.grad, dtype=np.float32)[np.newaxis, :]
                _acc(x, self._c.backward(dy).squeeze(0))

        out._backward = _bwd
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  SISLDOLayer — Sparse Input, Sparse Linear, Dense Output
# ══════════════════════════════════════════════════════════════════════════════

class SISLDOLayer(_SparseLayerBase):
    """Sparse state → state contribution. Input must be a CSR."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                 num_cpus: int = 4, solidify: float = 0.01, backprop_p: float = 0.03):
        self._c         = _cpu.SISLDOLayer(in_features, out_features, max_weights, num_cpus, solidify)
        self.backprop_p = backprop_p

    def forward(self, x: Tensor) -> Tensor:
        """x.data must be a CSR. grad flows back as a dense ndarray."""
        csr    = x.data
        out_np = self._c.forward_sparse(csr.ptrs, csr.indices, csr.values, csr.rows).squeeze(0)
        out    = Tensor(out_np, _children=(x,), _op="sisldo", backend=x.backend)

        def _bwd():
            if out.grad is not None:
                dy   = np.asarray(out.grad, dtype=np.float32)[np.newaxis, :]
                k    = max(1, int(dy.shape[1] * self.backprop_p))
                dp, di, dv = _cpu.dense_to_top_k_csr(dy, k, self._c.num_cpus)
                dx = self._c.backward(
                    csr.ptrs, csr.indices, csr.values,
                    dy, dp, di, dv,
                    csr.rows, csr.cols,
                ).squeeze(0)
                _acc(x, dx)

        out._backward = _bwd
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  Save/restore helpers for a raw _cpu.SparseLinearLayer
# ══════════════════════════════════════════════════════════════════════════════

def _sparse_linear_layer_state_dict(layer) -> dict:
    """
    Full round-trippable state for a raw _cpu.SparseLinearLayer: weights
    AND per-row value_scale/importance_scale. weights_vals are RAW
    quantized units, not true values -- true value =
    weights_vals[i] * value_scale[row_of_i] (see set_value_scale_raw's own
    docstring) -- so scale must be saved alongside weights, or a reload
    silently uses scale=1.0 (SparseLinearLayer's default for an untouched
    row) instead of whatever was actually calibrated, corrupting every
    true weight value on that row without erroring.

    importance is saved (for inspection) but CANNOT currently be restored
    by _sparse_linear_layer_load_state_dict below: SparseLinearLayer.
    load_weights has no path to set per-connection importance (unlike
    DISLDOLayerV.load_weights, which takes an importance array) -- see
    TODO.md. A reloaded layer's importance starts fresh, not from here.
    """
    n = layer.n_inputs
    return {
        "ptrs":             np.array(layer.ptrs),
        "indices":          np.array(layer.indices),
        "weights":          np.array(layer.weights_vals),
        "importance":       np.array(layer.importance),  # NOT restorable -- see docstring
        "value_scale":      np.array([layer.get_value_scale(r) for r in range(n)], dtype=np.float32),
        "importance_scale": np.array([layer.get_importance_scale(r) for r in range(n)], dtype=np.float32),
    }


def _sparse_linear_layer_load_state_dict(layer, d: dict) -> None:
    """Restore weights and per-row value_scale/importance_scale onto an
    already-constructed layer of the matching shape. Does NOT restore
    importance -- see _sparse_linear_layer_state_dict's docstring."""
    layer.load_weights(
        np.asarray(d["ptrs"],    dtype=np.int32),
        np.asarray(d["indices"], dtype=np.int32),
        np.asarray(d["weights"], dtype=np.float32),
    )
    for r in range(layer.n_inputs):
        layer.set_value_scale_raw(r, float(d["value_scale"][r]))
        layer.set_importance_scale_raw(r, float(d["importance_scale"][r]))


# ══════════════════════════════════════════════════════════════════════════════
#  FoldedLayer — runtime sili Module for a converted folded transformer block
# ══════════════════════════════════════════════════════════════════════════════

class FoldedLayer(Module):
    """
    Runtime sili layer for a folded transformer block.

    All N original transformer layers are stacked into ONE SparseLinearLayer
    per weight suffix (Q, K, V, MLP, etc.).  A single forward() call replaces
    N sequential matmuls.  After synaptogenesis, only connections that survived
    energy-based pruning contribute nonzero terms -- which is why sparsity is
    what makes the design efficient rather than just wider.

    Weights live entirely in C++ (SparseLinearLayer).  parameters() returns []
    -- nothing here participates in the Tensor autograd as a leaf, but the
    layer IS in the graph: forward() returns a Tensor with _children=(x,) and
    a _backward closure that calls backward_dense and uses _acc to accumulate
    dx back into x.grad.

    No torch dependency in forward/backward.  The from_descriptor() factory
    method uses torch once at construction time (acceptable: construction is
    part of the conversion pipeline, not the runtime hot path).

    Shape contract (same as RNNFoldedBlock.forward):
        input  [batch, in_dim]   -> output [batch, out_dim]
    The stacked weights map in_dim -> n_folds*out_dim internally; the fold
    dimension is summed away on the way out (reshape + sum(axis=1)).
    """

    def __init__(
        self,
        layers:        dict,   # {suffix: SparseLinearLayer}
        n_folds:       int,
        out_dims:      dict,   # {suffix: out_dim}
        learning_rate: float = 0.01,
    ):
        self._sili_layers = layers
        self._n_folds     = n_folds
        self._out_dims    = out_dims
        self.lr           = learning_rate

    # ── Factory ------------------------------------------------------------------

    @classmethod
    def from_descriptor(cls, descriptor, learning_rate: float = 0.01,
                        num_cpus: int = 4,
                        max_row_weights: int = 0,
                        bytes_per_row: int = 0) -> "FoldedLayer":
        """
        Build a FoldedLayer from a FoldedBlockDescriptor.

        args:
          max_row_weights -- peak connections per row for synaptogenesis.
                             0 = n_out (the absolute ceiling). For real models,
                             pass your expected synaptogenesis peak (e.g. 100
                             for 2.4% density in a 4096-dim layer) to avoid
                             allocating space for connections you'll never use.
          bytes_per_row   -- index byte budget per row.
                             0 = compute from max_row_weights and the typical
                             ULEB128 cost for this layer's column range:
                               typical_bytes = ceil(log2(n_out) / 7)
                             For n_out <= 128: 1 byte/connection.
                             For n_out <= 16384: 2 bytes/connection.
                             The default adds a small margin for net growth per
                             synaptogenesis step. Pass an explicit value to
                             override (e.g. worst-case: max_row_weights * 5).
        """
        import numpy as np
        import torch as _torch   # local import: conversion step only -- sili does
        # the compute. torch is used once here to densify+transpose the stacked
        # CSR weights, then discarded. Do not use torch in forward/backward paths.
        import warnings; warnings.filterwarnings("ignore")
        _FP4_MAX = 6.0

        layers = {}
        for suffix, csr in descriptor.stacked_weights.items():
            csr_t = csr.to_dense().t().to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
            n_in  = int(csr_t.shape[0])
            n_out = int(csr_t.shape[1])
            nnz   = int(csr_t.values().numel())
            # Budget for the delta-CSR pool: size for the fully-connected
            # maximum (n_in * n_out), not for current nnz. This is the fixed
            # total the staggered equalizer_step() will redistribute within --
            # equalization only moves bytes between rows, never grows the pool.
            # Sizing for n_in*n_out guarantees every row can hold n_out
            # connections after a full equalization pass, which is the absolute
            # ceiling for any max_row_weights value.
            budget = n_in * n_out
            layer = _cpu.SparseLinearLayer(n_in, n_out, budget, num_cpus)
            ptrs = csr_t.crow_indices().numpy().astype(np.int32)
            idx  = csr_t.col_indices().numpy().astype(np.int32)
            vals = csr_t.values().float().numpy().copy()

            # Per-row value scaling: map each row's max-abs to FP4_MAX so the
            # quantizer uses its full resolution.  See conversation for why
            # per-row (not per-layer) is critical for a stacked matrix that
            # spans rows from N different original layers.
            row_scales = np.ones(n_in, dtype=np.float32)
            for r in range(n_in):
                start, end = int(ptrs[r]), int(ptrs[r + 1])
                if end > start:
                    max_abs = float(np.abs(vals[start:end]).max())
                    if max_abs > 0.0:
                        row_scales[r] = max_abs / _FP4_MAX
                        vals[start:end] /= row_scales[r]

            layer.load_weights(ptrs, idx, vals)
            for r in range(n_in):
                if row_scales[r] != 1.0:
                    layer.set_value_scale_raw(r, row_scales[r])

            # Per-row importance_scale: same FP4 representability problem as
            # value_scale but for importance.
            # Importance is updated via activity correlation in forward_dense
            # (magnitude ~ |x| * |h| * lr ~ lr after value scaling).
            # FP4 minimum nonzero is 0.5, so a raw update of lr=0.01 rounds to 0.
            # Setting importance_scale = lr / FP4_MAX maps FP4 range to
            # [-6*lr, +6*lr], making importance updates of order lr representable
            # from the first step. Weight VALUES are NOT changed in forward_dense;
            # they are updated only by backward_dense() via the task gradient.
            imp_scale = learning_rate / _FP4_MAX
            for r in range(n_in):
                layer.set_importance_scale_raw(r, imp_scale)

            # Choose capacity targets for equalize_to_capacity.
            # max_row_weights defaults to n_out (absolute ceiling).
            mrw = max_row_weights if max_row_weights > 0 else n_out

            # bytes_per_row: use the ULEB128 cost for this layer's column range
            # plus a small margin (~4 bytes) for net growth per step.
            # ceil(bits_needed / 7) gives bytes per delta for column indices 0..n_out.
            # This is the TYPICAL cost, not worst-case (uleb128_max=5).
            # Pass bytes_per_row explicitly to override (e.g. worst-case: mrw*5).
            if bytes_per_row > 0:
                bpr = bytes_per_row
            else:
                bits = max(1, n_out - 1).bit_length()
                typ  = (bits + 6) // 7    # ceil(bits / 7)
                bpr  = mrw * typ + 4       # +4 bytes margin per step

            layer.equalize_to_capacity(mrw, bpr)

            layers[suffix] = layer

        return cls(layers, descriptor.n_folds, descriptor.out_dims, learning_rate)

    # ── Module interface ---------------------------------------------------------

    def parameters(self) -> list:
        return []   # weights live in C++, not in the Tensor graph

    # ── Properties --------------------------------------------------------------

    @property
    def in_features(self) -> int:
        return next(iter(self._sili_layers.values())).n_inputs

    @property
    def out_features(self) -> int:
        return next(iter(self._out_dims.values()))

    # ── Forward ------------------------------------------------------------------

    def forward(self, x: "Tensor") -> "Tensor":
        """
        x: sili Tensor [batch, in_dim]  (or [in_dim] -- squeezed automatically)
        Returns: sili Tensor [batch, out_dim]

        Wired into sili autograd: calling loss.backward() propagates through
        this layer automatically. Two separate update paths run in the C++ kernels:

          - forward_dense(x, lr) [this method]:
              Computes output AND updates IMPORTANCE via activity correlation
              (|x| * |h| * lr). This does NOT change weight values.
              Importance tracks which connections are actively used, guiding
              synaptogenesis pruning/growing decisions later.

          - backward_dense(dy, lr) [_backward, called by loss.backward()]:
              Updates WEIGHT VALUES via task gradient (standard backprop).
              Also updates importance via gradient magnitude.
              This is what enables the network to learn tasks.

        Note: calling forward_dense with lr=0 disables importance tracking.
        Weight values never change without a backward() call.
        """
        x_np = np.asarray(x.data, dtype=np.float32)
        squeezed = x_np.ndim == 1
        if squeezed:
            x_np = x_np[np.newaxis, :]
        batch   = x_np.shape[0]
        out_dim = next(iter(self._out_dims.values()))
        lr      = self.lr

        # Single call per suffix -- the full stacked matrix is one layer.
        raw_parts = [layer.forward_dense(x_np, lr)
                     for layer in self._sili_layers.values()]
        raw_np = sum(raw_parts)   # [batch, n_folds * out_dim]

        # Fold sum: [batch, n_folds, out_dim] -> [batch, out_dim]
        summed = raw_np.reshape(batch, self._n_folds, out_dim).sum(axis=1)
        if squeezed:
            summed = summed.squeeze(0)

        out = Tensor(summed, _children=(x,), _op="folded", backend=x.backend)

        # Capture loop variables for the closure (Python late-binding risk).
        _layers  = list(self._sili_layers.values())
        _n_folds = self._n_folds
        _sq      = squeezed

        def _bwd():
            if out.grad is None:
                return
            dy_np = np.asarray(out.grad, dtype=np.float32)
            if dy_np.ndim == 1:
                dy_np = dy_np[np.newaxis, :]
            _batch = dy_np.shape[0]

            # Backward of fold reshape+sum:
            # grad of sum is 1 to each summand -> broadcast dy to all n_folds slots.
            dy_raw = np.tile(
                dy_np.reshape(_batch, 1, out_dim),
                (1, _n_folds, 1)
            ).reshape(_batch, _n_folds * out_dim).astype(np.float32)

            # Each suffix layer gets the same dy_raw; accumulate dx.
            dx_parts = [layer.backward_dense(dy_raw, lr, lr_per_row_nnz=True)
                        for layer in _layers]
            dx_np = sum(dx_parts).reshape(_batch, -1)
            if _sq:
                dx_np = dx_np.squeeze(0)
            _acc(x, dx_np)

        out._backward = _bwd
        return out

    # ── Synaptogenesis -----------------------------------------------------------

    def synaptogenesis(
        self,
        k:                int,
        importance_cutoff: float,
        max_row_weights:   int,
        rows_per_call:     int = 0,
    ) -> None:
        """
        Grow and prune connections across all suffix layers.

        Each call to synap_step() advances ONE row of the layer's internal
        cursor, deciding for that row: remove synapses whose importance fell
        below importance_cutoff, then grow new ones (from the top-k probes)
        until the row reaches max_row_weights.

        To prune and grow uniformly, the net effect of one full sweep
        (all n_inputs rows visited) is:
          - removed: synapses with importance < importance_cutoff
          - added:   up to max_row_weights - surviving_nnz new synapses
          - total:   capped at max_row_weights per row (constant if all rows
                     were already at max_row_weights before pruning)

        args:
          k                 -- probes to build (how many candidate connections
                               per row to consider for growth).  Rule of thumb:
                               k ~ 4 * max_row_weights gives good coverage.
          importance_cutoff -- prune synapses whose stored importance magnitude
                               falls below this threshold (in FP4 stored units;
                               multiply by get_importance_scale(r) for true units)
          max_row_weights   -- target connections per row after this sweep.
                               Vary this over time (e.g. sine wave) to test
                               that the layer can both grow AND shrink.
          rows_per_call     -- 0 (default) = full sweep (all n_inputs rows);
                               N > 0 = advance exactly N rows (staggered mode,
                               useful when called every training step to spread
                               the work across many steps rather than a single
                               large pause).

        Call AFTER backward() and BEFORE the next forward().
        Accumulators are zeroed at the end of each call -- they are valid only
        for the interval between the last zero_accum and this synaptogenesis call.
        """
        for layer in self._sili_layers.values():
            layer.build_probes(k)
            n = rows_per_call if rows_per_call > 0 else layer.n_inputs
            for _ in range(n):
                layer.synap_step(importance_cutoff, max_row_weights)
            layer.equalizer_step()   # staggered 1-row redistribution
            layer.zero_accum()

    def nnz_total(self) -> int:
        """Total live connections across all suffix layers (for monitoring)."""
        return sum(layer.nnz for layer in self._sili_layers.values())

    # ── State persistence --------------------------------------------------------

    def state_dict(self) -> dict:
        out = {}
        for suffix, layer in self._sili_layers.items():
            d = _sparse_linear_layer_state_dict(layer)
            d["n_folds"] = np.array([self._n_folds])
            d["out_dim"] = np.array([self._out_dims[suffix]])
            d["lr"]      = np.array([self.lr], dtype=np.float32)
            out[suffix]  = d
        return out

    def load_state_dict(self, d: dict) -> None:
        """
        Restore weights + per-row value_scale/importance_scale for every
        suffix layer (previously: no load_state_dict existed on this
        class at all -- state_dict()'s output could be saved but never
        loaded back). Does NOT restore importance itself -- see
        _sparse_linear_layer_state_dict's docstring; a reloaded
        connection's importance starts fresh, not from what was saved.
        """
        for suffix, sub in d.items():
            _sparse_linear_layer_load_state_dict(self._sili_layers[suffix], sub)


# ══════════════════════════════════════════════════════════════════════════════
#  FoldedColumnLayer — FoldedLayer variant for the column-averaging mechanism
# ══════════════════════════════════════════════════════════════════════════════

class FoldedColumnLayer(FoldedLayer):
    """
    FoldedLayer variant for the column-averaging mechanism (see
    sili_peridot/todolist.md Phase A3/A4): retains the pre-sum
    [n_folds*out_dim] tensor instead of collapsing the fold axis
    (FoldedLayer.forward sums it away immediately), and pairs it with a
    `recurrent` layer -- structurally the same input_proj+recurrent split
    SparseRNNCell already uses (h = input_proj(obs) + recurrent(state)),
    just built on FoldedLayer/SparseLinearLayer instead of the currently
    -broken DISLDOLayer/SISLDOLayer (see TODO.md).

    in_proj(x) -- this class's own inherited from_descriptor weights (the
    REAL pretrained per-fold-step matrices, stacked) -- only genuinely
    represents "external input -> fold step 1" faithfully; every other
    fold step's band is computed from the same raw x too (FoldedLayer's
    approximation, see build_fold_skip_layer's docstring), not from what
    the prior fold step actually produced.

    recurrent(state) -- build_fold_skip_layer's from-scratch banded
    matrix (see below), mapping THIS layer's own [n_folds*out_dim] output
    space back to itself: fold step i's slot -> fold step i+1's slot (and
    nearby columns within bandwidth), genuinely carrying one step's
    output into the next's input, which in_proj alone cannot do. No
    pretrained content -- purely synaptogenesis/backprop-trained from the
    zero-value banded pre-seed.

    forward(x, state) = in_proj(x) + recurrent(state), returned as the
    new state -- feed it back in for the next call, mirroring
    SparseRNNCell's own calling convention exactly. state defaults to
    zero (true step-0, matching RNNFoldedBlock.forward's state=0 start)
    when not given.

    Feed forward()'s output to sili.energy.column_averaging_loss (after
    whatever EnergyDynamics gating is in the actual model -- see that
    function's docstring for why it must run on the energy-GATED state,
    not this layer's raw output).
    """

    @classmethod
    def from_descriptor(cls, descriptor, learning_rate: float = 0.01,
                        num_cpus: int = 4, max_row_weights: int = 0,
                        bytes_per_row: int = 0,
                        recurrent_bandwidth: int = None,
                        existing_recurrent=None,
                        existing_recurrent_prefer: str = "b") -> "FoldedColumnLayer":
        """
        Like FoldedLayer.from_descriptor, plus builds `recurrent` (see
        class docstring and build_fold_skip_layer) sized to this layer's
        own [n_folds*out_dim] output space.

        recurrent_bandwidth: forwarded to build_fold_skip_layer as
        `bandwidth` -- None (default) uses that function's own default
        (out_dim, i.e. one hop reaches the adjacent fold step).

        existing_recurrent / existing_recurrent_prefer: forwarded to
        build_fold_skip_layer as `existing`/`existing_prefer` -- pass a
        previously-saved recurrent CSR (e.g.
        state_dict_to_true_csr(some_prior_layer.state_dict()["recurrent"]))
        to preserve real trained skip-connection weights when re-running
        from_descriptor on (possibly updated) LLM weights, rather than
        starting recurrent from scratch every time.
        None (default) is the plain "converting a dense LLM, which has no
        skip connections at all" case.
        """
        obj = super().from_descriptor(
            descriptor, learning_rate=learning_rate, num_cpus=num_cpus,
            max_row_weights=max_row_weights, bytes_per_row=bytes_per_row,
        )
        obj.recurrent = build_fold_skip_layer(
            obj._n_folds, obj.column_width, num_cpus=num_cpus,
            bandwidth=recurrent_bandwidth, expected_lr=learning_rate,
            existing=existing_recurrent, existing_prefer=existing_recurrent_prefer,
        )
        return obj

    @property
    def out_features(self) -> int:
        # FoldedLayer's out_features is the PER-FOLD out_dim (post fold
        # -sum); this layer doesn't sum, so its real output width is
        # n_folds times that.
        return self._n_folds * next(iter(self._out_dims.values()))

    @property
    def column_width(self) -> int:
        """Per-fold-step output width -- also recurrent's own row/column
        count divided by n_folds."""
        return next(iter(self._out_dims.values()))

    def in_proj(self, x: "Tensor") -> "Tensor":
        """
        The pretrained-weight half of forward(): does NOT sum over the
        fold axis like FoldedLayer.forward does -- returns
        [batch, n_folds*out_dim] (or [n_folds*out_dim] if x was 1-D),
        every fold step's own out_dim-sized projection, concatenated
        rather than collapsed.
        """
        x_np = np.asarray(x.data, dtype=np.float32)
        squeezed = x_np.ndim == 1
        if squeezed:
            x_np = x_np[np.newaxis, :]
        lr = self.lr

        raw_parts = [layer.forward_dense(x_np, lr)
                     for layer in self._sili_layers.values()]
        raw_np = sum(raw_parts)   # [batch, n_folds*out_dim] -- kept as-is
        if squeezed:
            raw_np = raw_np.squeeze(0)

        out = Tensor(raw_np, _children=(x,), _op="folded_column_in_proj", backend=x.backend)

        _layers = list(self._sili_layers.values())
        _sq     = squeezed

        def _bwd():
            if out.grad is None:
                return
            dy_np = np.asarray(out.grad, dtype=np.float32)
            if dy_np.ndim == 1:
                dy_np = dy_np[np.newaxis, :]
            dx_parts = [layer.backward_dense(dy_np, lr, lr_per_row_nnz=True)
                        for layer in _layers]
            dx_np = sum(dx_parts).reshape(dy_np.shape[0], -1)
            if _sq:
                dx_np = dx_np.squeeze(0)
            _acc(x, dx_np)

        out._backward = _bwd
        return out

    def forward(self, x: "Tensor", state: "Tensor" = None) -> "Tensor":
        """
        h = in_proj(x) + recurrent(state) -- same pattern as
        SparseRNNCell.forward. Returns the new state; feed it back in as
        `state` on the next call. state=None (default) uses zeros -- true
        step-0, matching RNNFoldedBlock.forward's state=0 start.
        """
        raw = self.in_proj(x)
        if state is None:
            state = Tensor(np.zeros(self.out_features, dtype=np.float32),
                           backend=x.backend)
        rec = apply_fold_skip(self.recurrent, state, lr=self.lr)
        return raw + rec

    def state_dict(self) -> dict:
        """FoldedLayer.state_dict() (in_proj) plus recurrent -- the
        original version of this method (inherited, unoverridden) saved
        only in_proj, silently dropping any trained recurrent weights on
        save/reload."""
        out = super().state_dict()
        out["recurrent"] = _sparse_linear_layer_state_dict(self.recurrent)
        return out

    def load_state_dict(self, d: dict) -> None:
        """Restore in_proj (via FoldedLayer.load_state_dict) and
        recurrent. Does NOT restore importance for either -- see
        _sparse_linear_layer_state_dict's docstring."""
        super().load_state_dict({k: v for k, v in d.items() if k != "recurrent"})
        _sparse_linear_layer_load_state_dict(self.recurrent, d["recurrent"])


def _build_banded_csr(total: int, bandwidth: int):
    """Pure CSR array generation for a [total, total] banded-diagonal
    pattern: row r connects to columns c with abs(r-c) < bandwidth,
    clipped to [0, total). All values 0.0. Split out of
    build_fold_skip_layer so its structural pattern can be unioned with
    an existing CSR (see csr_union) before any SparseLinearLayer gets
    constructed."""
    positions = np.arange(total)
    lo = np.clip(positions - bandwidth + 1, 0, None)
    hi = np.clip(positions + bandwidth - 1, None, total - 1)
    row_lengths = (hi - lo + 1).astype(np.int64)

    ptrs = np.zeros(total + 1, dtype=np.int64)
    ptrs[1:] = np.cumsum(row_lengths)
    nnz = int(ptrs[-1])

    idx = np.empty(nnz, dtype=np.int32)
    pos = 0
    for r in range(total):
        n = int(row_lengths[r])
        idx[pos:pos + n] = np.arange(lo[r], hi[r] + 1, dtype=np.int32)
        pos += n

    return ptrs.astype(np.int32), idx, np.zeros(nnz, dtype=np.float32)


def state_dict_to_true_csr(d: dict):
    """
    Convert one _sparse_linear_layer_state_dict()-shaped dict (e.g.
    layer.state_dict()["recurrent"]) into a (ptrs, idx, vals) CSR with
    vals in TRUE units -- the format csr_union/build_fold_skip_layer's
    `existing` expects. Necessary because state_dict() saves RAW
    quantized weights + a separate per-row value_scale (see
    _sparse_linear_layer_state_dict's docstring for why they're kept
    separate there); csr_union needs one merged "true value" per
    position instead, since raw units from two differently-scaled
    sources aren't comparable/summable directly.
    """
    ptrs   = np.asarray(d["ptrs"],  dtype=np.int32)
    idx    = np.asarray(d["indices"], dtype=np.int32)
    raw    = np.asarray(d["weights"], dtype=np.float32)
    scale  = np.asarray(d["value_scale"], dtype=np.float32)
    vals   = raw.copy()
    for r in range(len(ptrs) - 1):
        start, end = int(ptrs[r]), int(ptrs[r + 1])
        if end > start:
            vals[start:end] = raw[start:end] * scale[r]
    return ptrs, idx, vals


def csr_union(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b,
              n_rows: int, prefer: str = "a"):
    """
    Merge two CSRs of the SAME shape (n_rows rows, same implicit column
    space) into one CSR holding the union of their nonzero positions.
    `vals_a`/`vals_b` must already be in the SAME units (true values --
    i.e. already multiplied by whatever per-row value_scale their source
    used, not raw FP4 levels; mixing raw units from two differently
    -scaled sources here would silently be wrong) -- rescaling to raw
    FP4 units happens after the union, once a single per-row scale for
    the MERGED row is chosen (see build_fold_skip_layer's `existing`).

    Where both inputs have an entry at the same (row, col), `prefer`
    decides the result: 'a' (default) keeps A's value, 'b' keeps B's,
    'sum' adds them. Positions present in only one input use that value
    directly.

    Not used in the normal forward/backward/training path -- only at
    construction/loading time, when a layer's structural pattern needs
    to come from more than one source. See build_fold_skip_layer's
    `existing` parameter for the concrete case this exists for: a dense
    LLM has no skip connections at all, so converting one starts from an
    empty set that gets unioned with the fresh banded pre-seed (trivial
    union, degenerates to just the band); a previously-trained
    recurrent, or manually-specified additional skip connections, are
    the non-trivial cases.
    """
    assert prefer in ("a", "b", "sum")
    row_idx = [None] * n_rows
    row_val = [None] * n_rows
    for r in range(n_rows):
        a_start, a_end = int(ptrs_a[r]), int(ptrs_a[r + 1])
        b_start, b_end = int(ptrs_b[r]), int(ptrs_b[r + 1])
        merged = dict(zip(idx_a[a_start:a_end].tolist(), vals_a[a_start:a_end].tolist()))
        for c, v in zip(idx_b[b_start:b_end].tolist(), vals_b[b_start:b_end].tolist()):
            if c in merged:
                if prefer == "sum":
                    merged[c] += v
                elif prefer == "b":
                    merged[c] = v
                # prefer == "a": keep A's existing value, ignore B's
            else:
                merged[c] = v
        cols_sorted = sorted(merged.keys())
        row_idx[r] = np.asarray(cols_sorted, dtype=np.int32)
        row_val[r] = np.asarray([merged[c] for c in cols_sorted], dtype=np.float32)

    new_idx  = np.concatenate(row_idx) if n_rows > 0 else np.zeros(0, dtype=np.int32)
    new_vals = np.concatenate(row_val) if n_rows > 0 else np.zeros(0, dtype=np.float32)
    new_ptrs = np.zeros(n_rows + 1, dtype=np.int32)
    for r in range(n_rows):
        new_ptrs[r + 1] = new_ptrs[r] + len(row_idx[r])

    return new_ptrs, new_idx, new_vals


def build_fold_skip_layer(n_folds: int, out_dim: int, num_cpus: int = 4,
                          bandwidth: int = None,
                          headroom_fraction: float = 0.5,
                          expected_lr: float = 0.01,
                          existing=None,
                          existing_prefer: str = "b") -> "_cpu.SparseLinearLayer":
    """
    Sparse layer mapping a FoldedColumnLayer's own [n_folds*out_dim]
    output space back to itself: literal skip connections between
    virtual (fold-depth) layers, pre-seeded as a banded pattern. With no
    `existing` (the default), this is entirely from-scratch and
    zero-valued -- the case for converting a dense LLM, which has no
    skip connections at all to bring forward.

    Why: FoldedLayer's single-matmul-then-sum trick computes every fold
    step's contribution from the SAME external input, independent of
    every other step -- a first-order approximation of the true fold
    recurrence (see RNNFoldedBlock.forward in conversion/rnn_fold.py:
    state=0; for i: state += block_i(x+state), each step seeing
    everything accumulated by every prior step). Apply this layer to a
    FoldedColumnLayer's raw output (refined = raw + skip.forward_dense(raw))
    to let nearby fold steps influence each other -- a cheap correction
    toward the true recurrence's behavior without paying for n_folds real
    sequential matmuls.

    bandwidth: connect flat positions r, c whenever abs(r - c) < bandwidth.
    Default (None) uses out_dim -- i.e. one hop reaches the adjacent fold
    step at any nearby column, not just the exact same column index
    (a wider band lets training find useful cross-column routes, not only
    identity-like passthrough). At real model scale (out_dim in the
    thousands) this is genuinely large (~2*out_dim connections per row);
    tune bandwidth down if construction time/memory becomes a problem --
    not solved here, just noted.

    expected_lr: FP4 gotcha (see project notes) -- a freshly zero-valued
    connection is structurally stuck at zero unless its per-row
    value_scale is set relative to the learning rate that will actually
    be used to train it (FP4's minimum nonzero magnitude is
    0.5*value_scale; an update of order expected_lr rounds back to zero
    under the default scale otherwise). Used as the fallback scale for
    any row whose FINAL (post-union) values are still all zero; rows with
    real nonzero values (from `existing`) instead get max_abs/FP4_MAX,
    same as from_descriptor's own pretrained-weight scaling, so real
    trained magnitudes aren't clipped by a scale sized for zero-escape.

    existing: optional (ptrs, idx, vals) CSR, same [n_folds*out_dim,
    n_folds*out_dim] shape, vals in TRUE units (see csr_union) -- unioned
    with the fresh banded pre-seed before construction. NOT used in the
    normal forward/backward/training path -- only when recurrent's
    structural pattern needs to come from more than one source: loading
    from a dense LLM (the default, existing=None -- union with an empty
    set is a no-op, so this degenerates to plain banded pre-seeding) is
    the immediate case; a previously-trained recurrent (pass its saved
    CSR here to preserve real trained values while still guaranteeing the
    band is present) or manually-specified additional skip connections
    are the non-trivial ones.
    existing_prefer: csr_union's prefer arg for positions in both the
    fresh band and `existing` -- default 'b' (existing's value wins over
    the fresh zero-valued pre-seed, since bringing forward real trained
    weights is the point of passing `existing` at all).
    """
    assert n_folds >= 1 and out_dim >= 1
    bw = out_dim if bandwidth is None else bandwidth
    assert bw >= 1
    total = n_folds * out_dim

    ptrs, idx, vals = _build_banded_csr(total, bw)

    if existing is not None:
        ex_ptrs, ex_idx, ex_vals = existing
        ptrs, idx, vals = csr_union(ptrs, idx, vals, ex_ptrs, ex_idx, ex_vals,
                                    total, prefer=existing_prefer)

    nnz = len(idx)
    row_lengths = ptrs[1:] - ptrs[:-1]
    max_row_weights = int(row_lengths.max()) if nnz > 0 else 1
    budget = nnz + int(headroom_fraction * nnz) + total

    # Per-row scale: real (post-union) magnitude gets full FP4 resolution
    # (max_abs/FP4_MAX, same as from_descriptor's pretrained-weight
    # scaling); a row that's still all-zero falls back to expected_lr's
    # zero-escape scale, same reasoning as the no-`existing` case.
    _FP4_MAX = 6.0
    row_scales = np.full(total, expected_lr / _FP4_MAX, dtype=np.float32)
    for r in range(total):
        start, end = int(ptrs[r]), int(ptrs[r + 1])
        if end > start:
            max_abs = float(np.abs(vals[start:end]).max())
            if max_abs > 0.0:
                row_scales[r] = max_abs / _FP4_MAX
                vals[start:end] = vals[start:end] / row_scales[r]

    layer = _cpu.SparseLinearLayer(total, total, budget, num_cpus)
    layer.load_weights(ptrs, idx, vals)
    layer.equalize_to_capacity(max_row_weights)

    for r in range(total):
        layer.set_value_scale_raw(r, float(row_scales[r]))
        layer.set_importance_scale_raw(r, expected_lr / _FP4_MAX)

    return layer


def apply_fold_skip(skip_layer, x: "Tensor", lr: float = 0.01) -> "Tensor":
    """
    Apply a build_fold_skip_layer()-constructed layer to a
    FoldedColumnLayer's raw output, wired into the Tensor autograd graph
    (forward_dense + backward_dense, the same pattern FoldedLayer/
    FoldedColumnLayer use for their own suffix layers). Typical use:
    refined = raw + apply_fold_skip(skip, raw, lr) -- the skip
    connections contribute a residual correction on top of the
    independently-computed per-fold-step output.
    """
    x_np = np.asarray(x.data, dtype=np.float32)
    squeezed = x_np.ndim == 1
    if squeezed:
        x_np = x_np[np.newaxis, :]

    out_np = skip_layer.forward_dense(x_np, lr)
    if squeezed:
        out_np = out_np.squeeze(0)
    out = Tensor(out_np, _children=(x,), _op="fold_skip", backend=x.backend)

    def _bwd():
        if out.grad is None:
            return
        dy = np.asarray(out.grad, dtype=np.float32)
        if dy.ndim == 1:
            dy = dy[np.newaxis, :]
        dx = skip_layer.backward_dense(dy, lr, lr_per_row_nnz=True)
        if squeezed:
            dx = dx.squeeze(0)
        _acc(x, dx)

    out._backward = _bwd
    return out


class LayerMemoryState:
    """
    Python-side tracker for a SparseLinearLayer's memory equalization cursor.

    The C++ equalizer_step() advances an internal row cursor each call; this
    class mirrors that cursor in Python and provides memory statistics. Use
    it to integrate equalization into training loops with visibility.

    Normal training loop:
        mem = LayerMemoryState(sparse_layer)
        for step in range(n_steps):
            out = layer(x); loss.backward()
            synap_schedule.step()
            mem.step()            # one equalization step per training step

    Synaptogenesis on a row that has no blank space will throw. The throw
    signals that equalization hasn't caught up yet. Calling mem.step() once
    per training step ensures blank space is continuously redistributed as
    synaptogenesis adds and removes connections.
    """

    def __init__(self, layer):
        self._layer  = layer   # SparseLinearLayer (_cpu object)
        self._cursor = 0       # mirrors C++ _equalize_row
        self._calls  = 0

    def step(self) -> None:
        """One equalization step (advance cursor by one row)."""
        self._layer.equalizer_step()
        self._cursor = (self._cursor + 1) % max(1, self._layer.n_inputs)
        self._calls += 1

    @property
    def cursor_row(self) -> int:
        """Which row will be equalized next."""
        return self._cursor

    @property
    def calls(self) -> int:
        """Total equalization steps taken."""
        return self._calls

    @property
    def cycles(self) -> float:
        """Full equalization cycles completed (n_inputs steps = 1 cycle)."""
        n = max(1, self._layer.n_inputs)
        return self._calls / n

    @property
    def nnz(self) -> int:
        return self._layer.nnz


class SynaptogenesisSchedule:
    """
    Schedule for calling FoldedLayer.synaptogenesis() at regular intervals
    with a (optionally varying) max_row_weights target.

    Constant connections (default):
        sched = SynaptogenesisSchedule(layer, base_connections=64,
                                       every_n_steps=20)

    Sine-wave connections (useful for testing grow/shrink both work):
        sched = SynaptogenesisSchedule(layer, base_connections=64,
                                       amplitude=0.3, period=200,
                                       every_n_steps=20)

    During training:
        for step, (x, y) in enumerate(data):
            out  = layer(x)
            loss = criterion(out, y)
            loss.backward()
            sched.step()           # handles synaptogenesis cadence internally

    The sine wave is:
        max_row_weights(t) = round(base * (1 + amplitude * sin(2*pi*t/period)))

    With amplitude=0, this is constant at base.  The sine wave exercises both
    growth (max > base) and pruning (max < base) and is a clean regression:
    after many full cycles, nnz_total should oscillate around base * n_rows.
    """

    def __init__(
        self,
        layer:             "FoldedLayer",
        base_connections:  int,
        k_factor:          int   = 4,        # probes = k_factor * max_row_weights
        importance_cutoff: float = 0.0,      # stored-unit importance threshold
        amplitude:         float = 0.0,      # 0 = constant, 0.3 = +-30%
        period:            int   = 200,      # steps per full sine cycle
        every_n_steps:     int   = 20,       # run synaptogenesis every N steps
        rows_per_call:     int   = 0,        # 0 = full sweep
    ):
        self._layer             = layer
        self._base              = base_connections
        self._k_factor          = k_factor
        self._importance_cutoff = importance_cutoff
        self._amplitude         = amplitude
        self._period            = period
        self._every             = every_n_steps
        self._rows_per_call     = rows_per_call
        self._t                 = 0      # training steps counted
        self._synap_t           = 0      # synaptogenesis calls counted

    def current_max_row_weights(self) -> int:
        """Current target based on the sine wave at this step."""
        if self._amplitude == 0.0:
            return self._base
        import math
        factor = 1.0 + self._amplitude * math.sin(
            2.0 * math.pi * self._synap_t / self._period)
        return max(1, round(self._base * factor))

    def step(self) -> bool:
        """
        Advance one training step. Runs synaptogenesis if the cadence fires.
        Returns True if synaptogenesis ran this step.
        """
        self._t += 1
        if self._t % self._every != 0:
            return False
        mrw = self.current_max_row_weights()
        k   = max(1, self._k_factor * mrw)
        self._layer.synaptogenesis(
            k, self._importance_cutoff, mrw, self._rows_per_call)
        self._synap_t += 1
        return True

    @property
    def t(self) -> int:
        """Training steps elapsed."""
        return self._t

    @property
    def synap_calls(self) -> int:
        """Synaptogenesis calls made so far."""
        return self._synap_t


# ══════════════════════════════════════════════════════════════════════════════
#  SparseRNNCell
# ══════════════════════════════════════════════════════════════════════════════

class SparseRNNCell(Module):
    """
    One sparse RNN step:

        recurrent_out = recurrent(csr, state)     [measured alone, see below]
        h             = input_proj(obs) + recurrent_out
        h_out         = energy(h)

    Returns (h_out: Tensor, aux_loss: Tensor, actual_p: float) — h_out (dense)
    is returned unchanged as the new state, same contract as before, so
    argmax/save/inspection on it keep working without modification.

    Unifying the sparsification passes (see energy.kept_indices docstring):
    the CSR fed into `recurrent()` at the top of the NEXT call is built from
    THIS call's own energy-gating decision (kept_indices + the PRE-gating h
    values at those indices) rather than an independent top-k re-derivation
    that could disagree with it. That decision is cached on the cell
    (`_prev_kept_indices` / `_prev_h_dense`) rather than smuggled through the
    state Tensor itself, since state.data must stay dense for argmax/save to
    keep working, and CSR must not be built from state.data's own values
    anyway -- state.data (h_out) has fire/shutoff positions flattened to
    energy-derived constants (2.0, e+2), not the real activation magnitude
    the gate decided to keep. The cache is invalidated on reset()/whenever
    the caller hands in a state the cell didn't itself just produce (e.g.
    after SparseRNNAgent.load()), falling back to CSR.from_dense (the true
    step-0 path) exactly once until the cell has run again.

    Branching-ratio measurement (see energy.BranchingRatioTracker /
    energy.EMABranchingRatioTracker): recurrent-only activity is measured
    on recurrent_out BEFORE it's summed with input_proj(obs) -- measuring
    on the combined h cannot distinguish a genuinely self-propagating
    recurrent pathway from fresh input alone carrying activity while the
    recurrent branching factor is silently 0. `branching_tracker` selects
    which estimator backs `self.branching_recurrent`: "window" (
    a hard sliding window, also the only one that supports
    avalanche_sizes() for a SOC power-law-tail check) or "ema" (default, O(1)
    memory, exponentially-discounted -- prefer this when you want a
    continuously-updated read with a tunable fast/long-term tradeoff via
    `branching_ema_alpha`, e.g. for `dynamic_density_from_branching_ratio`
    reacting promptly to a regime change). Want both a fast EMA read and
    the avalanche-size check at once? Construct a second tracker yourself
    (`EMABranchingRatioTracker`/`BranchingRatioTracker` from `sili.energy`)
    and feed it the same recurrent_out activity this cell already
    computes each step -- not built into this class, since which
    additional trackers (if any) matter is a caller decision, not
    something this cell should hardcode a combination of.
    """

    def __init__(self, n_inputs: int, state_size: int, max_weights: int,
                 num_cpus: int = 4, solidify: float = 0.01, percent_active: float = 0.03,
                 dynamic_density_from_branching_ratio: bool = False,
                 branching_tracker: str = "ema",
                 branching_window: int = 200,
                 branching_ema_alpha: float = 0.05):
        assert branching_tracker in ("window", "ema"), \
            f"branching_tracker must be 'window' or 'ema', got {branching_tracker!r}"
        r = percent_active / 0.02
        self.input_proj = DISLDOLayer(n_inputs,   state_size, max_weights, num_cpus, solidify)
        self.recurrent  = SISLDOLayer(state_size, state_size, max_weights, num_cpus, solidify,
                                      backprop_p=percent_active)
        # density IS the target active fraction; p is a hard compute-limit
        # ceiling that must sit clearly above it (~5x here), not the thing
        # that shapes learned sparsity -- see EnergyDynamics's own
        # `density <= p * 0.8` assertion and its docstring for why. This
        # inverts what this constructor did before (density used to be
        # derived FROM percent_active*0.9 while p was set TO percent_active
        # directly, i.e. density could exceed p*0.8 -- the actual bug).
        density = min(0.9, percent_active)
        p       = min(1.0, percent_active * 5.0)
        self.energy     = EnergyDynamics(
            drive          = 0.08*percent_active * r,
            activation_cost= 0.08 * r,
            density        = density,
            exploration    = 0.001 * r,
            reactivity     = 0.01  * r,
            precision      = 0.04  * r,
            setpoint       = 1.0,
            activation_threshold = 1e-4,
            p              = p,
        )
        self.state_size      = state_size
        self._percent_active = percent_active

        # Recurrent-only branching-ratio measurement (A6 item 5) and its
        # optional (default-off) use to nudge the KL density target (A6's
        # "biggest structural change" -- a first-cut proportional adjustment,
        # not a first-principles derivation; see energy-params.md). "window"
        # is the default so existing behavior/callers are unaffected;
        # "ema" trades the avalanche_sizes() check away for O(1) memory and
        # a tunable fast/long-term response via branching_ema_alpha -- see
        # class docstring.
        if branching_tracker == "window":
            self.branching_recurrent = BranchingRatioTracker(window=branching_window)
        else:
            self.branching_recurrent = EMABranchingRatioTracker(alpha=branching_ema_alpha)
        self.branching_tracker_mode = branching_tracker
        self.dynamic_density_from_branching_ratio = bool(dynamic_density_from_branching_ratio)

        # Cache for unifying the sparsification passes -- see class docstring.
        self._prev_kept_indices: Optional[np.ndarray] = None
        self._prev_h_dense:      Optional[np.ndarray] = None

    def parameters(self) -> list:
        return []

    def forward(self, obs: Tensor, state: Tensor) -> Tuple[Tensor, Tensor, float]:
        # state.data is a CSR only if the caller explicitly handed us one
        # (e.g. warm-starting from a saved CSR); the cell's own output is
        # always dense (see class docstring). Normal path: build the CSR
        # from this cell's OWN cached gating decision when we have one and
        # the caller hasn't reset/replaced the state since; otherwise fall
        # back to the true step-0 independent top-k.
        if not isinstance(state.data, CSR):
            if self._prev_kept_indices is not None:
                state_csr = CSR.from_kept_indices(
                    self._prev_kept_indices, self._prev_h_dense, cols=self.state_size)
            else:
                state_csr = CSR.from_dense(
                    np.asarray(state.data, dtype=np.float32),
                    p=self._percent_active,
                    num_cpus=self.input_proj.num_cpus,
                )
            state = state_csr.as_tensor(state.backend)

        # Measure the recurrent pathway's OWN activity before it's mixed
        # with input_proj(obs) -- see class docstring / BranchingRatioTracker.
        recurrent_out = self.recurrent(state)
        recurrent_activity = float(np.sum(
            np.abs(np.asarray(recurrent_out.data, dtype=np.float32))
            > self.energy.activation_threshold
        ))
        self.branching_recurrent.update(recurrent_activity)

        h = self.input_proj(obs) + recurrent_out

        density_override = None
        if self.dynamic_density_from_branching_ratio:
            m = self.branching_recurrent.branching_ratio()
            if m is not None:
                # First-cut proportional nudge around the configured base
                # density, centered on the intended near-critical band
                # [0.97, 0.99] -- NOT a first-principles derivation from m.
                # Bounded to +/-2x the base density so a noisy early
                # estimate can't send the target somewhere degenerate.
                m_target = 0.98
                density_override = float(np.clip(
                    self.energy.density * (1.0 + 2.0 * (m - m_target)),
                    self.energy.density * 0.5, self.energy.density * 2.0,
                ))

        new_state, aux_loss, actual_p = self.energy(h, density_override=density_override)

        # Cache this call's gating decision for the NEXT call's CSR
        # construction (pre-gating h, not h_out -- see class docstring).
        self._prev_kept_indices = self.energy.kept_indices
        self._prev_h_dense      = np.asarray(h.data, dtype=np.float32).ravel().copy()

        return new_state, aux_loss, actual_p

    def reset(self):
        # Invalidate the sparsification-pass cache -- the next state the
        # caller hands in did NOT come from this cell's own last forward
        # call (e.g. after an external reset or a loaded checkpoint), so
        # the cached kept_indices/h_dense no longer describe it.
        self._prev_kept_indices = None
        self._prev_h_dense      = None
        self.branching_recurrent.reset()

    def step(self, lr: float):
        self.input_proj.step(lr)
        self.recurrent .step(lr)

    def decay(self, rate: float):
        self.input_proj.decay(rate)
        self.recurrent .decay(rate)

    def synaptogenesis(self, k: int, lr: float, importance_beta: float, max_weights: int):
        self.input_proj.synaptogenesis(k, lr, importance_beta, max_weights)
        self.recurrent .synaptogenesis(k, lr, importance_beta, max_weights)

    def state_dict(self) -> dict:
        return {
            "input_proj": self.input_proj.state_dict(),
            "recurrent":  self.recurrent .state_dict(),
            "energy":     self.energy    .state_dict(),
        }

    def load_state_dict(self, d: dict):
        self.input_proj.load_state_dict(d["input_proj"])
        self.recurrent .load_state_dict(d["recurrent"])
        self.energy    .load_state_dict(d["energy"])
        self.reset()  # loaded weights, not a state this cell itself produced



# ══════════════════════════════════════════════════════════════════════════════
#  SparseRNNAgent
# ══════════════════════════════════════════════════════════════════════════════

class SparseRNNAgent(Module):
    """
    Sparse RNN agent.

        action = argmax(state.data[:n_actions])

    State is a Tensor. For BPTT=1 call train_step(); for multi-step BPTT
    call forward() in a loop then loss.backward() then step() manually.

    aux_loss is public so callers can sum it with task losses before backward.
    """

    def __init__(self, n_inputs: int, n_actions: int, state_size: int, max_weights: int,
                 num_cpus: int = 4, solidify: float = 0.01, percent_active: float = 0.03,
                 lr: float = 1e-3, importance_beta: float = 0.01,
                 importance_decay: float = 1e-3*0.03, synaptogenesis_k: int = 64,
                 synaptogenesis_every: int = 20):
        assert n_actions <= state_size

        self.cell = SparseRNNCell(n_inputs, state_size, max_weights, num_cpus,
                                  solidify, percent_active)

        self.state = Tensor(np.zeros(state_size, dtype=np.float32))

        self.n_inputs    = n_inputs
        self.n_actions   = n_actions
        self.state_size  = state_size
        self.max_weights = max_weights

        self.lr                   = lr
        self.importance_beta      = importance_beta
        self.importance_decay     = importance_decay
        self.synaptogenesis_k     = synaptogenesis_k
        self.synaptogenesis_every = synaptogenesis_every

        self._step_count = 0
        self.aux_loss:   Optional[Tensor] = None
        self._actual_p:  float = 0.0

    def parameters(self) -> list:
        return []

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, obs: Tensor) -> int:
        """Run one step. State stays in the autograd graph (use for multi-step BPTT)."""
        h_out, aux_loss, actual_p = self.cell(obs, self.state)
        self.state      = h_out
        self.aux_loss   = aux_loss
        self._actual_p  = actual_p
        return int(np.argmax(np.asarray(self.state.data, dtype=np.float32)[:self.n_actions]))

    def train_step(self, obs: Tensor) -> int:
        """
        BPTT=1 convenience wrapper. Detaches state before forward so gradients
        don't flow across steps, then runs aux_loss.backward() and step().

        Use aux_loss directly before calling this if you want to add a task loss:
            action   = agent.forward(obs)
            combined = agent.aux_loss + task_loss(action, target)
            combined.backward()
            agent.step()

        The comment below is kept from the original as a design note:
        Using aux_loss + force-firing rather than a scalar reward can be far
        more information-rich — touching a hot stove produces a burning
        sensation where you put your hand, not just a global 'bad' signal.
        """
        self.state = self.state.detach()
        action = self.forward(obs)
        self.aux_loss.backward()
        self.step()
        return action

    # ── Optimization ─────────────────────────────────────────────────────────

    def step(self):
        self.cell.step(self.lr)
        self.cell.decay(self.importance_decay)
        self._step_count += 1
        if self._step_count % self.synaptogenesis_every == 0:
            self.cell.synaptogenesis(
                self.synaptogenesis_k, self.lr,
                self.importance_beta, self.max_weights)

    def reset_state(self):
        self.state    = Tensor(np.zeros(self.state_size, dtype=np.float32))
        self.aux_loss = None
        self.cell.reset()

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        d    = self.cell.state_dict()
        flat = {
            "_step_count": np.array([self._step_count]),
            "_state":      np.asarray(self.state.data, dtype=np.float32),
        }
        for section, sub in d.items():
            for k, v in sub.items():
                flat[f"{section}__{k}"] = v
        np.savez_compressed(path, **flat)

    def load(self, path: str):
        raw = np.load(path, allow_pickle=False)
        self._step_count = int(raw["_step_count"][0])
        self.state       = Tensor(raw["_state"].copy())

        def _section(prefix):
            return {k[len(prefix)+2:]: raw[k] for k in raw if k.startswith(prefix + "__")}

        self.cell.load_state_dict({
            "input_proj": _section("input_proj"),
            "recurrent":  _section("recurrent"),
            "energy":     {"energy": raw.get("energy__energy", np.zeros(0, dtype=np.float32))},
        })


# ══════════════════════════════════════════════════════════════════════════════
#  UnifiedOptimizer
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedOptimizer:
    """
    Steps both standard Tensor parameters (Linear, RMSNorm, etc.) and C++-backed
    sparse layers in one call. Useful when mixing Module types in one model.
    """

    def __init__(self, model: Module, lr: float = 0.001):
        self.lr               = lr
        self._tensor_params   = model.parameters()       # Tensors via _iter_leaves
        self._sparse_layers   = self._find_sparse(model)

    def _find_sparse(self, module: Module) -> list:
        out = []
        for val in module.__dict__.values():
            if isinstance(val, _SparseLayerBase):
                out.append(val)
            elif isinstance(val, Module):
                out.extend(self._find_sparse(val))
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, _SparseLayerBase):
                        out.append(item)
                    elif isinstance(item, Module):
                        out.extend(self._find_sparse(item))
        return out

    def step(self):
        for p in self._tensor_params:
            if p.grad is not None:
                p.data -= self.lr * p.grad
                p.grad  = None
        for layer in self._sparse_layers:
            layer.step(self.lr)
