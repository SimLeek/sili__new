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
from sili.energy import EnergyDynamics


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

    @staticmethod
    def from_dense(x: np.ndarray, p: float = 0.03, num_cpus: int = 4) -> "CSR":
        """
        Build CSR keeping the top-k entries by magnitude, k = max(1, round(cols * p)).
        x : float32 [cols] or [batch, cols]
        """
        x2d = x[np.newaxis, :] if x.ndim == 1 else x
        x2d = np.asarray(x2d, dtype=np.float32)
        k   = max(1, int(x2d.shape[1] * p))
        ptrs, indices, values = _cpu.dense_to_top_k_csr(x2d, k, num_cpus)
        return CSR(ptrs, indices, values, rows=x2d.shape[0], cols=x2d.shape[1])

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
                        num_cpus: int = 4) -> "FoldedLayer":
        """
        Build a FoldedLayer from a FoldedBlockDescriptor (produced by rnn_fold.py).

        Uses torch once to densify and transpose the stacked CSR weights, then
        discards it.  The returned FoldedLayer has no further torch dependency.
        """
        import numpy as np
        import torch as _torch   # local import: conversion step only
        import warnings; warnings.filterwarnings("ignore")
        _FP4_MAX = 6.0

        layers = {}
        for suffix, csr in descriptor.stacked_weights.items():
            csr_t = csr.to_dense().t().to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
            n_in  = int(csr_t.shape[0])
            n_out = int(csr_t.shape[1])
            nnz   = int(csr_t.values().numel())
            layer = _cpu.SparseLinearLayer(n_in, n_out, int(nnz * 1.2) + 64, num_cpus)

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

        Wired into sili autograd: calling loss.backward() will propagate through
        this layer automatically.  Weight updates (Hebbian + gradient) happen
        inside the C++ kernels:
          - Hebbian importance update: forward_dense(x, lr)  [this method]
          - Gradient weight+importance update: backward_dense(dy, lr) [_backward]
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

    def synaptogenesis(self, k: int, importance_cutoff: float,
                       max_row_weights: int) -> None:
        """
        Grow and prune connections across all suffix layers.
        Call after backward() and before the next forward().
        """
        for layer in self._sili_layers.values():
            layer.build_probes(k)
            layer.synap_step(importance_cutoff, max_row_weights)
            layer.equalizer_step()

    # ── State persistence --------------------------------------------------------

    def state_dict(self) -> dict:
        out = {}
        for suffix, layer in self._sili_layers.items():
            out[suffix] = {
                "ptrs":       np.array(layer.ptrs),
                "indices":    np.array(layer.indices),
                "weights":    np.array(layer.weights_vals),
                "importance": np.array(layer.importance),
                "n_folds":    np.array([self._n_folds]),
                "out_dim":    np.array([self._out_dims[suffix]]),
                "lr":         np.array([self.lr], dtype=np.float32),
            }
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  SparseRNNCell
# ══════════════════════════════════════════════════════════════════════════════

class SparseRNNCell(Module):
    """
    One sparse RNN step:

        h     = input_proj(obs) + recurrent(csr, state)
        h_out = energy(h)

    The CSR of h_out is cached internally and reused as the recurrent input
    on the next call — no redundant sparsification outside the cell.

    Returns (h_out: Tensor, aux_loss: Tensor, actual_p: float).
    """

    def __init__(self, n_inputs: int, state_size: int, max_weights: int,
                 num_cpus: int = 4, solidify: float = 0.01, percent_active: float = 0.03):
        r = percent_active / 0.02
        self.input_proj = DISLDOLayer(n_inputs,   state_size, max_weights, num_cpus, solidify)
        self.recurrent  = SISLDOLayer(state_size, state_size, max_weights, num_cpus, solidify,
                                      backprop_p=percent_active)
        self.energy     = EnergyDynamics(
            drive          = 0.08*percent_active * r,
            activation_cost= 0.08 * r,
            density        = min(0.25, percent_active * 0.9),
            exploration    = 0.001 * r,
            reactivity     = 0.01  * r,
            precision      = 0.04  * r,
            setpoint       = 1.0,
            kl_eps         = 1e-4,
            p              = percent_active,
        )
        self.state_size      = state_size
        self._percent_active = percent_active

    def parameters(self) -> list:
        return []

    def forward(self, obs: Tensor, state: Tensor) -> Tuple[Tensor, Tensor, float]:
        # state.data is a CSR after the first step; dense Tensor on step 0.
        # Normalise: if state.data is not already a CSR, sparsify it once.
        if not isinstance(state.data, CSR):
            state = CSR.from_dense(
                np.asarray(state.data, dtype=np.float32),
                p=self._percent_active,
                num_cpus=self.input_proj.num_cpus,
            ).as_tensor(state.backend)

        h                   = self.input_proj(obs) + self.recurrent(state)
        new_state, aux_loss, actual_p = self.energy(h)

        # new_state is a dense Tensor — return it directly as the new state.
        # The next forward call converts it to CSR at the top when the
        # recurrent layer needs it. Storing CSR here would force the recurrent
        # input to be the energy-gated pattern (2.0 / 0) rather than the
        # actual activation values.
        print(self.input_proj.nnz, self.recurrent.nnz, end='\r')
        return new_state, aux_loss, actual_p

    def reset(self):
        pass  # no cached state — CSR lives in the state Tensor itself

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