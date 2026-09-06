"""
sili.sparse_rnn — sparse RNN layers.

All layers are Module subclasses. C++-backed layers (DISLDOLayer, SISLDOLayer)
carry no Tensor parameters — their weights live in a real _cpu.SparseLinearLayer
and are updated INLINE during backward, not via a separate optimizer step.
parameters() returns [] for these.

Forward flow in SparseRNNCell::

    obs   (Tensor) ──[DISLDOLayer]────────────────────────────► Tensor
    state (Tensor) ──[CSR.from_dense]──[SISLDOLayer]──────────► Tensor
                                                  sum ──[EnergyDynamics]──► h_out

Weight VALUE updates happen inline inside backward, not via a separate
optimizer.step() -- required to keep memory bounded. Structural growth
(synaptogenesis) is the one genuinely separate call. See
docs/research/sparse_rnn.rst:sparse_rnn.module_overview for the full
design rationale (BPTT modes, why no periodic importance decay).
"""

from __future__ import annotations

import time as _diag_time
from typing import NamedTuple

import numpy as np

from sili import _cpu

_DIAG_TIMING: dict = {}

from sili.energy import BranchingRatioTracker, EMABranchingRatioTracker, EnergyDynamics
from sili.module import Module
from sili.tensor import Tensor, _acc

# ══════════════════════════════════════════════════════════════════════════════
#  CSR activation format
# ══════════════════════════════════════════════════════════════════════════════


class CSR(NamedTuple):
    """Sparse row-major activation tensor."""

    ptrs: np.ndarray  # int32   [rows+1]
    indices: np.ndarray  # int32   [nnz]
    values: np.ndarray  # float32 [nnz]
    rows: int
    cols: int

    @property
    def nnz(self) -> int:
        return len(self.indices)

    @staticmethod
    def from_dense(x: np.ndarray, p: float = 0.03, num_cpus: int = 4) -> CSR:
        """
        Build CSR keeping the top-k entries by magnitude PER ROW,
        k = max(1, round(cols * p)).
        x : float32 [cols] or [batch, cols]

        Independent top-k -- use ONLY when no prior sparsification decision
        already exists for this activation (i.e. true step-0, before any
        EnergyDynamics gate has run). Once a gate decision exists, prefer
        from_kept_indices below: re-deriving sparsity independently here can
        disagree with what the gate already decided.
        """
        x2d = x[np.newaxis, :] if x.ndim == 1 else x
        x2d = np.asarray(x2d, dtype=np.float32)
        k = max(1, int(x2d.shape[1] * p))
        ptrs, indices, values = _graded_top_k_csr(x2d, np.full(x2d.shape[0], k, dtype=np.int32), num_cpus)
        return CSR(ptrs, indices, values, rows=x2d.shape[0], cols=x2d.shape[1])

    @staticmethod
    def from_kept_indices(kept_indices: np.ndarray, values_source: np.ndarray, cols: int) -> CSR:
        """
        Build a single-row CSR directly from an already-decided set of kept
        column indices (e.g. EnergyDynamics.kept_indices) and a dense array
        to pull values from at those indices -- the "unify the two
        sparsification passes" path: prefer this over from_dense's
        independent top-k whenever a prior gating decision already exists.

        kept_indices  : int array, sorted ascending, indices into values_source
        values_source : float32 [cols]. Pass the PRE-gating activation, not
                        h_out -- h_out's fired/shutoff positions hold
                        energy-derived constants, not the real magnitude.
        cols          : full (dense) width this row represents
        """
        kept_indices = np.asarray(kept_indices, dtype=np.int32)
        values_source = np.asarray(values_source, dtype=np.float32).ravel()
        values = values_source[kept_indices]
        ptrs = np.array([0, len(kept_indices)], dtype=np.int32)
        return CSR(ptrs, kept_indices, values, rows=1, cols=cols)

    def to_dense(self) -> np.ndarray:
        """Reconstruct dense float32 [rows, cols]."""
        out = np.zeros((self.rows, self.cols), dtype=np.float32)
        p = np.asarray(self.ptrs)
        idx = np.asarray(self.indices)
        v = np.asarray(self.values)
        for r in range(self.rows):
            out[r, idx[p[r] : p[r + 1]]] = v[p[r] : p[r + 1]]
        return out

    def as_tensor(self, backend=None) -> Tensor:
        """Wrap this CSR as a Tensor. The CSR is the data; grad will be dense."""
        from sili.tensor import get_backend

        b = backend or get_backend("cpu")
        return Tensor(self, backend=b)


# ══════════════════════════════════════════════════════════════════════════════
#  Shared base for C++-backed sparse layers
# ══════════════════════════════════════════════════════════════════════════════


class _SparseLayerBase(Module):
    """
    Module base for layers whose weights live in a real _cpu.SparseLinearLayer.
    parameters() returns [] — nothing participates in Tensor autograd; weight
    VALUE updates happen inline inside backward_dense/backward_sparse instead
    (see module docstring).
    """

    def parameters(self) -> list:
        return []

    @property
    def in_features(self) -> int:
        return self._c.n_inputs

    @property
    def out_features(self) -> int:
        return self._c.n_outputs

    @property
    def nnz(self) -> int:
        return self._c.nnz

    @property
    def num_cpus(self) -> int:
        return self._c.num_cpus

    @property
    def out_degree(self) -> int:
        return self._c.out_degree

    @property
    def weights(self) -> np.ndarray:
        return self._c.weights_vals

    @property
    def importance(self) -> np.ndarray:
        return self._c.importance

    @property
    def indices(self) -> np.ndarray:
        return self._c.indices

    @property
    def ptrs(self) -> np.ndarray:
        return self._c.ptrs

    @property
    def neuron_input_accum(self) -> np.ndarray:
        return self._c.neuron_input_accum

    @property
    def neuron_grad_accum(self) -> np.ndarray:
        return self._c.neuron_grad_accum

    def synaptogenesis(
        self,
        k: int,
        importance_cutoff: float,
        max_row_weights: int,
        importance_eps: float = 1e-3,
        max_prune_per_step: int = 8,
    ):
        """Structural growth + memory rebalancing -- the only call here
        that isn't inline with forward/backward, since it changes which
        synapses exist rather than updating a value. Meant to be called
        every online step.

        importance_eps: read-time-only floor on importance_cutoff
        comparisons in synap_step, never written to storage. See
        docs/research/sparse_rnn.rst:sparse_layer_base.synaptogenesis_importance_eps_ghost_floor.

        max_prune_per_step: safety ceiling on removals per row per call
        (default rarely binds)."""
        self._c.build_probes(k)
        self._c.synap_step(
            importance_cutoff, max_row_weights, max_prune_per_step=max_prune_per_step, importance_eps=importance_eps
        )
        self._c.equalizer_step()

    def magnitude_rescale_output(self, target: float, correction_rate: float, scale_invariant: bool = False) -> None:
        """Passthrough to the real C++ magnitude_rescale_output. Not every
        backend has this bound (e.g. fp32 DISLDOLayerV has no scale to
        rescale) -- guard with hasattr(layer._c, "magnitude_rescale_output")."""
        self._c.magnitude_rescale_output(target, correction_rate, scale_invariant)

    def apply_amortized_l2_decay(self, chunk_size: int, decay_factor: float) -> dict:
        """Passthrough to the real C++ apply_amortized_l2_decay (amortized
        decoupled weight decay + rolling health stats via a persistent
        per-layer cursor). Bound on every backend. decay_factor is derived
        by the caller from a target half-life and this layer's nnz."""
        return self._c.apply_amortized_l2_decay(chunk_size, decay_factor)

    def state_dict(self) -> dict:
        return {
            "ptrs": np.array(self.ptrs),
            "indices": np.array(self.indices),
            "weights": np.array(self.weights),
            "importance": np.array(self.importance),
        }

    def load_state_dict(self, d: dict):
        # load_weights takes (ptrs, indices, weights) only -- no importance
        # array in this API generation; a loaded layer's importance starts
        # fresh and rebuilds through subsequent training.
        self._c.load_weights(
            d["ptrs"].astype(np.int32),
            d["indices"].astype(np.int32),
            d["weights"].astype(np.float32),
        )
        # load_weights is a tight/exact-fit load with no spare per-row
        # headroom -- restore growth room or a subsequent synap_step raises.
        max_row_weights = getattr(self, "_max_row_weights", None)
        if max_row_weights is not None:
            self._c.equalize_to_capacity(max_row_weights)


def _preseed_random_sparse(
    c, n_inputs: int, n_outputs: int, max_weights: int, rng: np.random.Generator | None = None
) -> int:
    """Bootstrap a freshly-constructed (zero-connection) SparseLinearLayer
    with a random-sparse init, an optimization for faster initial activity
    (not strictly required -- EnergyDynamics's forced-firing can bootstrap
    an all-zero layer on its own, just slower). Also reserves per-row
    growth headroom via equalize_to_capacity, called AFTER load_weights
    (order matters -- see anchor below).

    CSR rows are INPUTS, columns are OUTPUTS.

    `rng`: defaults to fresh OS entropy (np.random.default_rng()), NOT
    affected by np.random.seed(). Pass an explicit
    np.random.default_rng(seed) for reproducible wiring in tests.

    See docs/research/sparse_rnn.rst:preseed_random_sparse.bootstrap_rationale_and_capacity_order.
    """
    per_row = max(2, max_weights // max(1, n_inputs))
    if rng is None:
        rng = np.random.default_rng()
    k = max(1, min(n_outputs, per_row // 2))  # leave half the row's headroom free to grow into
    scale = 1.0 / np.sqrt(k)
    ptrs = np.zeros(n_inputs + 1, dtype=np.int32)
    indices = np.empty(n_inputs * k, dtype=np.int32)
    values = np.empty(n_inputs * k, dtype=np.float32)
    for row in range(n_inputs):
        cols = np.sort(rng.choice(n_outputs, size=k, replace=False))
        indices[row * k : (row + 1) * k] = cols
        values[row * k : (row + 1) * k] = rng.standard_normal(k).astype(np.float32) * scale
        ptrs[row + 1] = ptrs[row] + k
    if hasattr(c, "equalize_to_capacity"):
        # SparseLinearLayer (FP4) convention -- 3-arg load_weights.
        c.load_weights(ptrs, indices, values)
        c.equalize_to_capacity(per_row)
    else:
        # DISLDOLayerV (32-bit fallback) -- no equalize_to_capacity
        # binding, and load_weights takes an explicit importance array.
        importance = np.zeros_like(values)
        c.load_weights(ptrs, indices, values, importance)
    return per_row


def _overflow_guard_array(arr: np.ndarray, clip: float, near: float, coef: float) -> np.ndarray:
    """Elementwise, context-free correction for one AQRS scale/additive
    channel array (value_scale, output_scale, additive_u, additive_v).
    Two parts: (1) an auto-correcting shrink once |value| exceeds `near`
    (a hinge-squared penalty applied directly in value-space, since these
    are C++-internal RMSprop state, not autograd leaves) and (2) a hard
    clip to `clip` as the final numerical-safety net (nan_to_num first).
    `near` should sit comfortably below `clip`. Per-channel, not
    aggregate-S. See
    docs/research/sparse_rnn.rst:overflow_guard_array.two_part_correction_design."""
    x = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=clip, neginf=-clip)
    excess = np.maximum(np.abs(x) - near, 0.0)
    corrected = x - coef * excess * np.sign(x)
    return np.clip(corrected, -clip, clip).astype(np.float32)


def _orthogonality_penalty_array(flat: np.ndarray, rank: int, coef: float) -> np.ndarray:
    """Ongoing (every-step) diversity penalty for one AQRS channel array
    (value_scale, output_scale, additive_u, additive_v). Standard
    soft-orthogonality regularizer, gradient step on
    sum_{k1!=k2}(cos(M_k1,M_k2))^2 -- correlation of DIRECTION only.
    Computed in NORMALIZED (unit-direction) space so the correction stays
    LINEAR in each channel's own magnitude, not cubic (a raw-magnitude
    version was a real bug -- see anchor). Gram's diagonal is zeroed
    (only cross-channel correlation is penalized; AQRS's own gamma_k
    already controls magnitude separately). rank<=1 is a no-op.

    See
    docs/research/sparse_rnn.rst:orthogonality_penalty_array.normalized_space_bug."""
    if rank <= 1:
        return flat
    n = flat.size // rank
    m = np.asarray(flat, dtype=np.float32).reshape(n, rank)
    norms = np.linalg.norm(m, axis=0, keepdims=True)  # [1, rank]
    u = m / np.maximum(norms, 1e-6)  # unit-direction columns, provably finite
    gram = u.T @ u  # cosine similarities, entries in [-1, 1]
    np.fill_diagonal(gram, 0.0)
    correction_dir = u @ gram  # bounded by ~rank regardless of M's raw scale
    m_new = m - coef * norms * correction_dir  # step scales linearly with each channel's own magnitude
    return m_new.reshape(-1).astype(np.float32)


def _apply_channel_orthogonality_penalty(c, coef: float) -> None:
    """Shared impl for DISLDOLayer/DISLDOLayerDeterministic/DISLDOLayer8's
    apply_channel_orthogonality_penalty. Skips additive_u/v when empty
    (additive_rank==0)."""
    scale_rank = c.get_scale_rank()
    vs = np.asarray(c.get_value_scale_raw_vector(), dtype=np.float32)
    if vs.size:
        c.set_value_scale_raw_vector(_orthogonality_penalty_array(vs, scale_rank, coef).tolist())
    os_ = np.asarray(c.get_output_scale_raw_vector(), dtype=np.float32)
    if os_.size:
        c.set_output_scale_raw_vector(_orthogonality_penalty_array(os_, scale_rank, coef).tolist())
    additive_rank = c.get_additive_rank()
    au = np.asarray(c.get_additive_u_raw_vector(), dtype=np.float32)
    if au.size:
        c.set_additive_u_raw_vector(_orthogonality_penalty_array(au, additive_rank, coef).tolist())
    av = np.asarray(c.get_additive_v_raw_vector(), dtype=np.float32)
    if av.size:
        c.set_additive_v_raw_vector(_orthogonality_penalty_array(av, additive_rank, coef).tolist())


def _apply_scale_overflow_guard(c, clip: float, near: float, coef: float) -> None:
    """Shared impl for DISLDOLayer/DISLDOLayerDeterministic/DISLDOLayer8's
    apply_scale_overflow_guard. Skips additive_u/v when additive_rank==0."""
    vs = np.asarray(c.get_value_scale_raw_vector(), dtype=np.float32)
    if vs.size:
        c.set_value_scale_raw_vector(_overflow_guard_array(vs, clip, near, coef).tolist())
    os_ = np.asarray(c.get_output_scale_raw_vector(), dtype=np.float32)
    if os_.size:
        c.set_output_scale_raw_vector(_overflow_guard_array(os_, clip, near, coef).tolist())
    au = np.asarray(c.get_additive_u_raw_vector(), dtype=np.float32)
    if au.size:
        c.set_additive_u_raw_vector(_overflow_guard_array(au, clip, near, coef).tolist())
    av = np.asarray(c.get_additive_v_raw_vector(), dtype=np.float32)
    if av.size:
        c.set_additive_v_raw_vector(_overflow_guard_array(av, clip, near, coef).tolist())


def _default_rank_cap(n_inputs: int, n_outputs: int) -> int:
    """Default scale_rank_max/additive_rank_max, derived from actual
    storage cost: the largest per-branch rank K such that both branches
    growing to K simultaneously still cost no more, combined with the
    base fp4 weights, than a plain dense fp32 matrix of the same shape.
    See docs/research/sparse_rnn.rst:default_rank_cap.byte_budget_derivation."""
    base_bytes = n_inputs * n_outputs * 1
    fp32_dense_bytes = n_inputs * n_outputs * 4
    remaining_budget = max(0, fp32_dense_bytes - base_bytes)
    per_channel_bytes = 4 * (n_inputs + n_outputs + 1)
    return max(1, int((remaining_budget / 2) // per_channel_bytes))


def _seed_scale_rank(
    c, rank: int, n_inputs: int, n_outputs: int, rng: np.random.Generator | None = None, scale: float = 0.05
) -> None:
    """Sets scale_rank on the C++ layer and seeds components k>=1 (both
    value_scale_k and output_scale_k) with small random values -- both
    default to 0.0, a chicken-and-egg deadlock otherwise. rank=1 is a
    no-op. Loop order matters (row-major, k ascending) -- do not reorder
    without re-verifying. See
    docs/research/sparse_rnn.rst:seed_scale_rank.chicken_egg_deadlock.
    """
    if rank <= 1:
        return
    if rng is None:
        rng = np.random.default_rng()
    c.set_scale_rank(rank)
    for r in range(n_inputs):
        for k in range(1, rank):
            c.set_value_scale_raw_k(r, k, float(rng.normal(0.0, scale)))
    for col in range(n_outputs):
        for k in range(1, rank):
            c.set_output_scale_raw_k(col, k, float(rng.normal(0.0, scale)))


def _seed_additive_rank(
    c, rank: int, n_inputs: int, n_outputs: int, rng: np.random.Generator | None = None, scale: float = 0.05
) -> None:
    """Sets additive_rank on the C++ layer and seeds EVERY component
    (k=0..rank-1, unlike _seed_scale_rank's k>=1) with small independent
    random values on both additive_u and additive_v -- same
    chicken-and-egg deadlock as _seed_scale_rank, additive instead of
    multiplicative. No k==0 baseline channel to preserve here
    (additive_rank==0 is a true no-op), so every channel is seeded. See
    docs/research/sparse_rnn.rst:seed_scale_rank.chicken_egg_deadlock.
    """
    if rank <= 0:
        return
    if rng is None:
        rng = np.random.default_rng()
    c.set_additive_rank(rank)
    for r in range(n_inputs):
        for k in range(rank):
            c.set_additive_u_raw_k(r, k, float(rng.normal(0.0, scale)))
    for col in range(n_outputs):
        for k in range(rank):
            c.set_additive_v_raw_k(col, k, float(rng.normal(0.0, scale)))


def _activate_gamma_tracking(c, additive_rank: int) -> None:
    """Turns on EMA tracking for both branches' gamma -- required for
    apply_dynamic_rank_control's Theorem 10 triggers to evaluate against
    real signal instead of permanently-zero EMA state. Neither
    _seed_scale_rank nor _seed_additive_rank touches gamma, so this is
    the explicit opt-in step. Writes gamma_k(0)=1.0 for both branches
    (same as the lazy default -- transparent, no behavior change).
    Additive branch only activated if additive_rank>0. See
    docs/research/sparse_rnn.rst:activate_gamma_tracking.opt_in_design.
    """
    if hasattr(c, "set_scale_gamma_raw_k"):
        c.set_scale_gamma_raw_k(0, 1.0)
    if additive_rank > 0 and hasattr(c, "set_additive_gamma_raw_k"):
        c.set_additive_gamma_raw_k(0, 1.0)


def _preseed_dense(c, n_inputs: int, n_outputs: int, rng: np.random.Generator | None = None, quantize_fn=None) -> int:
    """Fully dense counterpart to `_preseed_random_sparse` -- every
    (input, output) pair connected, loaded straight into block4 via
    `load_dense_codes`. `quantize_fn` defaults to `_cpu.fp4_quantize_array`
    but is a parameter so a caller can swap in a different scheme.

    Uses a fixed raw scale (not fan-in-scaled to n_inputs) plus a
    per-column `output_scale` correction fit to the REAL post-quantization
    live count -- two real bugs (FP4's zero-rounding floor silently
    killing most of a naively-scaled dense init, then an uncorrected
    fixed scale collapsing training to chance) were found and fixed here.
    See docs/research/sparse_rnn.rst:preseed_dense.two_real_bugs for the
    derivation and the wrong-axis correction that was also caught and
    fixed.
    """
    if rng is None:
        rng = np.random.default_rng()
    if quantize_fn is None:
        quantize_fn = _cpu.fp4_quantize_array
    scale = 1.5  # fixed, not fan-in-scaled -- keeps codes representable
    dense = rng.standard_normal((n_inputs, n_outputs)).astype(np.float32) * scale
    weight_codes = quantize_fn(dense.flatten())
    importance_codes = np.zeros(n_inputs * n_outputs, dtype=np.uint8)
    c.load_dense_codes(weight_codes, importance_codes)
    # Fan-in correction via output_scale metadata, from REAL per-column
    # live counts (code 0 = not live).
    live = weight_codes.reshape(n_inputs, n_outputs) != 0
    col_nnz = live.sum(axis=0)
    output_scale = 1.0 / (scale * np.sqrt(np.maximum(col_nnz, 1)))
    for col in range(n_outputs):
        c.set_output_scale_raw(col, float(output_scale[col]))
    return n_outputs  # every row is already at max capacity -- nothing left to grow into


def _preseed_dense_scattered(c, n_inputs: int, n_outputs: int, rng: np.random.Generator | None = None) -> int:
    """Fully dense counterpart to `_preseed_random_sparse`, for storage
    types with no block4 support at all. Every (input, output) pair
    connected via plain scattered CSR -- no quantization floor to correct
    for, so plain fan-in-normalized (1/sqrt(n_inputs)) Gaussian init
    (ordinary Xavier/Kaiming). Caller MUST construct `c` with
    max_weights >= n_inputs*n_outputs. See
    docs/research/sparse_rnn.rst:preseed_dense.two_real_bugs (last
    paragraph). NOT used by DISLDOLayer32/DISLDOLayerV any more -- see
    `_preseed_dense_fp32`, which loads straight into block4 instead, now
    that DeltaCSRBiValues<float> has its own block4 support (task #350)."""
    if rng is None:
        rng = np.random.default_rng()
    scale = 1.0 / np.sqrt(max(1, n_inputs))
    ptrs = np.arange(0, n_inputs * n_outputs + 1, n_outputs, dtype=np.int32)
    indices = np.tile(np.arange(n_outputs, dtype=np.int32), n_inputs)
    values = rng.standard_normal(n_inputs * n_outputs).astype(np.float32) * scale
    importance = np.zeros(n_inputs * n_outputs, dtype=np.float32)
    c.load_weights(ptrs, indices, values, importance)
    return n_outputs  # every row is already at max capacity -- nothing left to grow into


def _preseed_dense_fp32(c, n_inputs: int, n_outputs: int, rng: np.random.Generator | None = None) -> int:
    """Fully dense counterpart to `_preseed_random_sparse`, for
    DeltaCSRBiValues<float>/DISLDOLayerV -- loads straight into block4
    (`load_dense_values`, backing `block4_load_dense_fp32` in
    delta_csr_memory.hpp) instead of `_preseed_dense_scattered`'s plain CSR,
    so a dense fp32 arm gets the SAME SIMD dense-tile forward/backward path
    FP4/FP8 already had (task #350) -- the real point of a dense arm is to
    be fast, and a fully-connected matrix trivially fills every 4x4 tile
    completely, maximizing that benefit.

    No output_scale fan-in correction (unlike `_preseed_dense`'s FP4/FP8
    version) -- there's no quantization floor here to correct FOR, so
    ordinary fan-in-normalized (1/sqrt(n_inputs)) Gaussian init is already
    correct on its own, same as `_preseed_dense_scattered`'s reasoning."""
    if rng is None:
        rng = np.random.default_rng()
    scale = 1.0 / np.sqrt(max(1, n_inputs))
    weight_values = (rng.standard_normal((n_inputs, n_outputs)).astype(np.float32) * scale).flatten()
    importance_values = np.zeros(n_inputs * n_outputs, dtype=np.float32)
    c.load_dense_values(weight_values, importance_values)
    return n_outputs  # every row is already at max capacity -- nothing left to grow into


def _preseed_empty(c, n_inputs: int, n_outputs: int, max_weights: int) -> int:
    """The genuinely-designed zero-weight-init: NO connections at all
    (nnz=0), not a dense grid pre-loaded with weight=0 -- real synapses
    are created by synaptogenesis() instead. Only reserves per-row growth
    headroom via equalize_to_capacity, safe to call directly on a fresh,
    never-load_weights'd layer. See
    docs/research/sparse_rnn.rst:preseed_empty.design_vs_zero_grid."""
    per_row = max(2, max_weights // max(1, n_inputs))
    c.equalize_to_capacity(per_row)
    return per_row


def _graded_top_k_csr(dy2d: np.ndarray, k_per_row, num_cpus: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genuinely per-row top-k selection: row r independently keeps its
    own top-k_per_row[r] largest-magnitude entries. Backed by a real C++
    kernel (`_cpu.dense_to_graded_top_k_csr`) -- NOT equivalent to
    `_cpu.dense_to_top_k_csr(dy2d, k, cpus)` with a uniform k, even at
    matching average density, since that spends k GLOBALLY across the
    flattened array, not per row. See
    docs/research/sparse_rnn.rst:graded_top_k_csr.python_to_cpp_migration."""
    _rows, cols = dy2d.shape
    k_arr = np.array([min(int(k), cols) for k in k_per_row], dtype=np.int32)
    dy2d_f32 = np.ascontiguousarray(dy2d, dtype=np.float32)
    ptrs, indices, values = _cpu.dense_to_graded_top_k_csr(dy2d_f32, k_arr, num_cpus)
    return ptrs, indices, values


def _nucleus_top_k_csr(
    x2d: np.ndarray, r_target, num_cpus: int = 4, k_min: int = 0, k_max: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nucleus/energy-threshold top-k: row r independently keeps the
    SMALLEST set of its own top-|v| entries whose captured squared-
    magnitude ratio R(v,k) = sum(v_topk^2)/sum(v^2) is >= r_target[r]. k
    is a CONSEQUENCE of r_target and the row's own data, not a fixed
    constant (same math as truncated-SVD captured-variance / nucleus
    sampling on squared magnitude).

    r_target: scalar or per-row array/list.
    k_min/k_max: hardware-driven density floor/ceiling applied after the
    fact; an all-zero row stays at k=0 regardless. See
    docs/research/sparse_rnn.rst:nucleus_top_k_csr.r_target_math."""
    rows, _cols = x2d.shape
    if np.isscalar(r_target):
        r_arr = np.full(rows, float(r_target), dtype=np.float32)
    else:
        r_arr = np.asarray(r_target, dtype=np.float32)
    x2d_f32 = np.ascontiguousarray(x2d, dtype=np.float32)
    ptrs, indices, values = _cpu.dense_to_nucleus_top_k_csr(
        x2d_f32, r_arr, num_cpus, k_min=k_min, k_max=(-1 if k_max is None else int(k_max))
    )
    return ptrs, indices, values


# ══════════════════════════════════════════════════════════════════════════════
#  DISLDOLayer — Dense Input, Sparse Linear, Dense Output
# ══════════════════════════════════════════════════════════════════════════════


class DISLDOLayer(_SparseLayerBase):
    """Dense observation → state contribution. No CSR on either side --
    dense in, dense out, via SparseLinearLayer.forward_dense/backward_dense
    (both sides dense, weight update inline in backward_dense -- see module
    docstring). No batch dimension required for online (single-sample) use."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        rng: np.random.Generator | None = None,
        dense: bool = False,
        scale_rank: int = 1,
        empty_init: bool = False,
        additive_rank: int = 0,
        dynamic_rank_control: bool = False,
        scale_rank_max: int | None = None,
        additive_rank_max: int | None = None,
    ):
        self._c = _cpu.SparseLinearLayer(in_features, out_features, max_weights, num_cpus)
        if empty_init:
            self._max_row_weights = _preseed_empty(self._c, in_features, out_features, max_weights)
        elif dense:
            self._max_row_weights = _preseed_dense(self._c, in_features, out_features, rng)
        else:
            self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)
        # Policy cap raised BEFORE seeding -- set_scale_rank/set_additive_rank
        # (called inside _seed_scale_rank/_seed_additive_rank below) validate
        # rank<=scale_rank_max/additive_rank_max (task #295), so a caller
        # requesting scale_rank/additive_rank above the default cap=4 needs
        # the cap raised first. None means "use the default formula".
        default_cap = _default_rank_cap(in_features, out_features)
        self._c.set_scale_rank_max(scale_rank_max if scale_rank_max is not None else default_cap)
        self._c.set_additive_rank_max(additive_rank_max if additive_rank_max is not None else default_cap)
        _seed_scale_rank(self._c, scale_rank, in_features, out_features, rng)
        _seed_additive_rank(self._c, additive_rank, in_features, out_features, rng)
        if dynamic_rank_control:
            _activate_gamma_tracking(self._c, additive_rank)

    def forward(
        self,
        x,
        learning_rate: float = 0.0,
        lr_per_row_nnz: bool = True,
        damp_by_importance: bool = True,
        min_decay_frac: float | None = None,
        max_abs_delta: float | None = None,
        max_ci: float | None = None,
        scale_invariant: bool = False,
        requires_grad: bool = True,
        dy_sparsity_p: float | None = None,
        dy_sparsity_schedule: list[float] | None = None,
        dy_r_target=None,
        dy_k_min: int = 0,
        dy_k_max: int | None = None,
    ) -> Tensor:
        # See docs/research/sparse_rnn.rst:disldo_layer_forward.design_notes
        # for the rationale behind min_decay_frac/max_abs_delta/max_ci
        # defaulting to None, the 1-D squeeze convention, lr_per_row_nnz,
        # and CSR-typed-input dispatch below.
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float32))
        if x.is_csr:
            csr = x.data
            was_1d = csr.rows == 1
            # backward_sparse needs the real dense x regardless of
            # forward's own path -- reconstruct it once, thread it
            # through the closure explicitly (never read a cached
            # last_input off the C++ side).
            x_dense = csr.to_dense()
            out_np = self._c.forward_sparse(csr.ptrs, csr.indices, csr.values, csr.rows)
        else:
            x_np = np.asarray(x.data, dtype=np.float32)
            was_1d = x_np.ndim == 1
            x_dense = x_np if x_np.ndim == 2 else x_np[np.newaxis, :]
            out_np = self._c.forward_dense(x_np)
        if was_1d:
            out_np = out_np.squeeze(0)
        # requires_grad=False: skip building the graph node entirely
        # (torch.no_grad()-style opt-out).
        if not requires_grad:
            return Tensor(out_np, backend=x.backend)
        out = Tensor(out_np, _children=(x,), _op="disldo", backend=x.backend)

        def _bwd():
            if out.grad is not None:
                dy = np.asarray(out.grad, dtype=np.float32)
                extra = {}
                if min_decay_frac is not None:
                    extra["min_decay_frac"] = min_decay_frac
                if max_abs_delta is not None:
                    extra["max_abs_delta"] = max_abs_delta
                if max_ci is not None:
                    extra["max_ci"] = max_ci
                if scale_invariant:
                    extra["scale_invariant"] = True
                # dy_sparsity_p/dy_sparsity_schedule/dy_r_target: see
                # docs/research/sparse_rnn.rst:disldo_layer_forward.dy_sparsity_schedule_and_nucleus_grad
                if dy_sparsity_schedule is not None:
                    # Per-row graded density, overrides scalar dy_sparsity_p.
                    dy2d = dy if dy.ndim == 2 else dy[np.newaxis, :]
                    if len(dy_sparsity_schedule) != dy2d.shape[0]:
                        raise ValueError(
                            f"dy_sparsity_schedule has {len(dy_sparsity_schedule)} entries, "
                            f"but dy has {dy2d.shape[0]} rows"
                        )
                    k_per_row = [max(1, int(dy2d.shape[1] * p)) for p in dy_sparsity_schedule]
                    dp, di, dv = _graded_top_k_csr(dy2d, k_per_row, self._c.num_cpus)
                    dx = self._c.backward_sparse(
                        x_dense,
                        dp,
                        di,
                        dv,
                        dy2d.shape[0],
                        learning_rate,
                        lr_per_row_nnz=lr_per_row_nnz,
                        damp_by_importance=damp_by_importance,
                        **extra,
                    )
                elif dy_r_target is not None:
                    # Nucleus/energy-threshold grad sparsification -- k is
                    # a CONSEQUENCE of dy_r_target and this step's actual
                    # gradient energy, not a fixed fraction.
                    dy2d = dy if dy.ndim == 2 else dy[np.newaxis, :]
                    dp, di, dv = _nucleus_top_k_csr(dy2d, dy_r_target, self._c.num_cpus, k_min=dy_k_min, k_max=dy_k_max)
                    dx = self._c.backward_sparse(
                        x_dense,
                        dp,
                        di,
                        dv,
                        dy2d.shape[0],
                        learning_rate,
                        lr_per_row_nnz=lr_per_row_nnz,
                        damp_by_importance=damp_by_importance,
                        **extra,
                    )
                elif dy_sparsity_p is None:
                    _t0 = _diag_time.perf_counter()
                    dx = self._c.backward_dense(
                        x_dense,
                        dy if dy.ndim == 2 else dy[np.newaxis, :],
                        learning_rate,
                        lr_per_row_nnz=lr_per_row_nnz,
                        damp_by_importance=damp_by_importance,
                        **extra,
                    )
                    _DIAG_TIMING["backward_dense"] = _DIAG_TIMING.get("backward_dense", 0.0) + (
                        _diag_time.perf_counter() - _t0
                    )
                    _DIAG_TIMING["backward_dense_n"] = _DIAG_TIMING.get("backward_dense_n", 0) + 1
                else:
                    _t0 = _diag_time.perf_counter()
                    dy2d = dy if dy.ndim == 2 else dy[np.newaxis, :]
                    k_dy = max(1, int(dy2d.shape[1] * dy_sparsity_p))
                    dp, di, dv = _graded_top_k_csr(dy2d, np.full(dy2d.shape[0], k_dy, dtype=np.int32), self._c.num_cpus)
                    _t1 = _diag_time.perf_counter()
                    dx = self._c.backward_sparse(
                        x_dense,
                        dp,
                        di,
                        dv,
                        dy2d.shape[0],
                        learning_rate,
                        lr_per_row_nnz=lr_per_row_nnz,
                        damp_by_importance=damp_by_importance,
                        **extra,
                    )
                    _t2 = _diag_time.perf_counter()
                    _DIAG_TIMING["topk_csr"] = _DIAG_TIMING.get("topk_csr", 0.0) + (_t1 - _t0)
                    _DIAG_TIMING["backward_sparse"] = _DIAG_TIMING.get("backward_sparse", 0.0) + (_t2 - _t1)
                    _DIAG_TIMING["backward_sparse_n"] = _DIAG_TIMING.get("backward_sparse_n", 0) + 1
                    _DIAG_TIMING["x_dense_shape"] = x_dense.shape
                    _DIAG_TIMING["dp_shape"] = (len(dp), len(di), len(dv))
                if was_1d:
                    dx = dx.squeeze(0)
                _acc(x, dx)

        out._backward = _bwd
        return out

    def apply_dynamic_rank_control(
        self,
        tau_death: float = 0.05,
        tau_active: float = 0.3,
        theta: float = 1e-4,
        seed_scale: float = 0.05,
        scale_grace_period_steps: int = 50,
        additive_grace_period_steps: int = 5000,
    ) -> bool:
        """AQRS Theorem 10 dynamic rank control -- evaluates both branches'
        apoptosis/neurogenesis triggers against their own EMA state
        (updated automatically inside backward) and performs at most one
        mutation PER BRANCH per call. Call once per training step, after
        backward. Returns True if either branch mutated.

        theta's default (1e-4) is tuned against gradient normalized by
        layer size (n_in*n_out). scale_grace_period_steps/
        additive_grace_period_steps are separate per-branch cooldowns --
        the additive branch defaults 100x longer (5000 vs 50), a
        biological homeostatic-vs-Hebbian-timescale analog. See
        docs/research/sparse_rnn.rst:disldo_layer.apply_dynamic_rank_control_theorem10.
        """
        mutated_scale = self._c.apply_dynamic_rank_control(
            tau_death, tau_active, theta, seed_scale, scale_grace_period_steps, scale_grace_period_steps
        )
        mutated_additive = self._c.apply_additive_dynamic_rank_control(
            tau_death, tau_active, theta, seed_scale, additive_grace_period_steps, additive_grace_period_steps
        )
        return mutated_scale or mutated_additive

    def apply_scale_overflow_guard(self, clip: float = 200.0, near: float = 20.0, coef: float = 0.1) -> None:
        """AQRS scale/additive channel numerical-safety pass -- see
        _overflow_guard_array. Call once per training step, any time
        after backward (independent of apply_dynamic_rank_control)."""
        _apply_scale_overflow_guard(self._c, clip, near, coef)

    def apply_channel_orthogonality_penalty(self, coef: float = 0.01) -> None:
        """AQRS channel-diversity pass -- see _orthogonality_penalty_array.
        Call once per training step, any time after backward (independent
        of apply_scale_overflow_guard/apply_dynamic_rank_control)."""
        _apply_channel_orthogonality_penalty(self._c, coef)


class DISLDOLayerResync(DISLDOLayer):
    """DISLDOLayer (true C++ FP4 storage) with the DeferredScaleWrite fix
    applied -- the FP4 counterpart of DISLDOLayer8Resync. See
    docs/research/sparse_rnn.rst:disldo_layer_variants.diagnostic_history."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        rng: np.random.Generator | None = None,
    ):
        self._c = _cpu.SparseLinearLayerResync(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)


class DISLDOLayerNoScale(DISLDOLayer):
    """DISLDOLayer (true C++ FP4 storage) with value_scale/output_scale
    permanently forced to 1.0 -- never trained, nothing to go stale. See
    docs/research/sparse_rnn.rst:disldo_layer_variants.diagnostic_history."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        rng: np.random.Generator | None = None,
    ):
        self._c = _cpu.SparseLinearLayerNoScale(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)


class DISLDOLayerDeterministic(DISLDOLayer):
    """DISLDOLayer (true C++ FP4 storage), same RMSprop scale handling as
    plain DISLDOLayer, but weight/importance storage uses deterministic
    nearest-neighbour rounding (fp4_quantize) instead of stochastic dithered
    rounding (fp4_quantize_stochastic) -- isolates rounding noise from the
    value_scale mechanism. See
    docs/research/sparse_rnn.rst:disldo_layer_variants.diagnostic_history."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        rng: np.random.Generator | None = None,
        dense: bool = False,
        scale_rank: int = 1,
        empty_init: bool = False,
        additive_rank: int = 0,
        dynamic_rank_control: bool = False,
        scale_rank_max: int | None = None,
        additive_rank_max: int | None = None,
    ):
        self._c = _cpu.SparseLinearLayerDeterministic(in_features, out_features, max_weights, num_cpus)
        if empty_init:
            self._max_row_weights = _preseed_empty(self._c, in_features, out_features, max_weights)
        elif dense:
            self._max_row_weights = _preseed_dense(self._c, in_features, out_features, rng)
        else:
            self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)
        default_cap = _default_rank_cap(in_features, out_features)
        self._c.set_scale_rank_max(scale_rank_max if scale_rank_max is not None else default_cap)
        self._c.set_additive_rank_max(additive_rank_max if additive_rank_max is not None else default_cap)
        _seed_scale_rank(self._c, scale_rank, in_features, out_features, rng)
        _seed_additive_rank(self._c, additive_rank, in_features, out_features, rng)
        if dynamic_rank_control:
            _activate_gamma_tracking(self._c, additive_rank)


class DISLDOLayerResyncDeterministic(DISLDOLayer):
    """DeferredScaleWrite fix + deterministic rounding together."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        rng: np.random.Generator | None = None,
    ):
        self._c = _cpu.SparseLinearLayerResyncDeterministic(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)


class DISLDOLayerNoScaleDeterministic(DISLDOLayer):
    """value_scale/output_scale forced off + deterministic rounding
    together -- the closest real-hardware match to the zero-trained-scale,
    deterministic-quantize design of fixed_digit_residual_quantize."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        rng: np.random.Generator | None = None,
    ):
        self._c = _cpu.SparseLinearLayerNoScaleDeterministic(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)


# ══════════════════════════════════════════════════════════════════════════════
#  DISLDOLayer32 — same DISLDO math, DeltaCSRBiValues<float> (32-bit) instead
#  of FP4BiPacked -- isolates FP4 quantization itself as a variable. See
#  docs/research/sparse_rnn.rst:disldo_layer_variants.diagnostic_history.
# ══════════════════════════════════════════════════════════════════════════════


class DISLDOLayer32(_SparseLayerBase):
    """Same disldo_forward/disldo_backward kernels as DISLDOLayer, generic
    over VALUES_TYPE -- this instantiation uses the 32-bit float fallback
    (_cpu.DISLDOLayerV) instead of 4-bit FP4BiPacked. Same call convention
    as DISLDOLayer. Growth-driven synaptogenesis (`synap_row_step`) now
    promotes/demotes into the SAME block4 dense-tile SIMD path FP4/FP8 use
    (task #350) -- float32 needs no encode/decode step at all, so its block4
    branch is simpler (and, per its own C++ comment, potentially faster)
    than either of theirs.

    dense=True: fully-connected init via `_preseed_dense_fp32`, straight
    into block4 (not the scattered CSR path `_preseed_dense_scattered`
    used before task #350) -- max_weights is still expanded automatically
    to cover every (input, output) pair. See
    docs/research/sparse_rnn.rst:disldo_layer_variants.diagnostic_history."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        rng: np.random.Generator | None = None,
        dense: bool = False,
    ):
        if dense:
            max_weights = max(max_weights, in_features * out_features)
        self._c = _cpu.DISLDOLayerV(in_features, out_features, max_weights, num_cpus)
        if dense:
            self._max_row_weights = _preseed_dense_fp32(self._c, in_features, out_features, rng)
        else:
            self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)

    def forward(
        self,
        x,
        learning_rate: float = 0.0,
        lr_per_row_nnz: bool = True,
        damp_by_importance: bool = True,
        min_decay_frac: float | None = None,
        max_abs_delta: float | None = None,
        max_ci: float | None = None,
        scale_invariant: bool = False,
        requires_grad: bool = True,
        dy_sparsity_p: float | None = None,
        dy_r_target=None,
        dy_k_min: int = 0,
        dy_k_max: int | None = None,
    ) -> Tensor:
        # CSR-typed input / dy_sparsity_p: mirrors DISLDOLayer.forward's
        # x.is_csr dispatch -- DISLDOLayerV uses the same DeltaCSRBiValues
        # storage family, so forward_sparse/backward_sparse unify dense+
        # sparse on the same weights exactly like DISLDOLayer does.
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float32))
        if x.is_csr:
            csr = x.data
            was_1d = csr.rows == 1
            x_dense = csr.to_dense()
            out_np = self._c.forward_sparse(csr.ptrs, csr.indices, csr.values, csr.rows)
        else:
            x_np = np.asarray(x.data, dtype=np.float32)
            was_1d = x_np.ndim == 1
            x_dense = x_np if x_np.ndim == 2 else x_np[np.newaxis, :]
            out_np = self._c.forward(x_np)
        if was_1d:
            out_np = out_np.squeeze(0)
        # See DISLDOLayer.forward's own requires_grad comment.
        if not requires_grad:
            return Tensor(out_np, backend=x.backend)
        out = Tensor(out_np, _children=(x,), _op="disldo32", backend=x.backend)

        def _bwd():
            if out.grad is not None:
                dy = np.asarray(out.grad, dtype=np.float32)
                extra = {}
                if min_decay_frac is not None:
                    extra["min_decay_frac"] = min_decay_frac
                if max_abs_delta is not None:
                    extra["max_abs_delta"] = max_abs_delta
                if max_ci is not None:
                    extra["max_ci"] = max_ci
                if scale_invariant:
                    extra["scale_invariant"] = True
                if dy_r_target is not None:
                    # See DISLDOLayer.forward's own dy_r_target comment.
                    dy2d = dy if dy.ndim == 2 else dy[np.newaxis, :]
                    dp, di, dv = _nucleus_top_k_csr(dy2d, dy_r_target, self._c.num_cpus, k_min=dy_k_min, k_max=dy_k_max)
                    dx = self._c.backward_sparse(
                        x_dense,
                        dp,
                        di,
                        dv,
                        dy2d.shape[0],
                        learning_rate,
                        lr_per_row_nnz=lr_per_row_nnz,
                        damp_by_importance=damp_by_importance,
                        **extra,
                    )
                elif dy_sparsity_p is None:
                    dx = self._c.backward(
                        x_dense,
                        dy if dy.ndim == 2 else dy[np.newaxis, :],
                        learning_rate,
                        lr_per_row_nnz=lr_per_row_nnz,
                        damp_by_importance=damp_by_importance,
                        **extra,
                    )
                else:
                    dy2d = dy if dy.ndim == 2 else dy[np.newaxis, :]
                    k_dy = max(1, int(dy2d.shape[1] * dy_sparsity_p))
                    dp, di, dv = _graded_top_k_csr(dy2d, np.full(dy2d.shape[0], k_dy, dtype=np.int32), self._c.num_cpus)
                    dx = self._c.backward_sparse(
                        x_dense,
                        dp,
                        di,
                        dv,
                        dy2d.shape[0],
                        learning_rate,
                        lr_per_row_nnz=lr_per_row_nnz,
                        damp_by_importance=damp_by_importance,
                        **extra,
                    )
                if was_1d:
                    dx = dx.squeeze(0)
                _acc(x, dx)

        out._backward = _bwd
        return out


class DISLDOLayer8(_SparseLayerBase):
    """Same disldo_forward/disldo_backward kernels as DISLDOLayer/
    DISLDOLayer32, generic over VALUES_TYPE -- this instantiation uses
    real 8-bit storage (_cpu.SparseLinearLayer8, OCP MX E4M3 per-value
    codec) instead of FP4BiPacked or the 32-bit fallback. Same call
    convention as DISLDOLayer/DISLDOLayer32.

    Combined with the existing rank-1 value_scale/output_scale mechanism,
    this is the "8-bit + rank-1 scale, weight AND importance both
    quantized" scheme. dense=True uses the block4 dense-tile SIMD path
    (Block4Tile8/Store8). See
    docs/research/sparse_rnn.rst:disldo_layer_variants.diagnostic_history."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        rng: np.random.Generator | None = None,
        dense: bool = False,
        scale_rank: int = 1,
        additive_rank: int = 0,
        dynamic_rank_control: bool = False,
        scale_rank_max: int | None = None,
        additive_rank_max: int | None = None,
    ):
        self._c = _cpu.SparseLinearLayer8(in_features, out_features, max_weights, num_cpus)
        if dense:
            self._max_row_weights = _preseed_dense(
                self._c, in_features, out_features, rng, quantize_fn=_cpu.fp8_quantize_array
            )
        else:
            self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)
        default_cap = _default_rank_cap(in_features, out_features)
        self._c.set_scale_rank_max(scale_rank_max if scale_rank_max is not None else default_cap)
        self._c.set_additive_rank_max(additive_rank_max if additive_rank_max is not None else default_cap)
        _seed_scale_rank(self._c, scale_rank, in_features, out_features, rng)
        # additive_rank: structurally fixes fp8 MQAR input-independent
        # collapse ("mumbling") -- see AQRS_DESIGN.md Theorem 3/4.
        _seed_additive_rank(self._c, additive_rank, in_features, out_features, rng)
        if dynamic_rank_control:
            _activate_gamma_tracking(self._c, additive_rank)

    def forward(
        self,
        x,
        learning_rate: float = 0.0,
        lr_per_row_nnz: bool = True,
        damp_by_importance: bool = True,
        min_decay_frac: float | None = None,
        max_abs_delta: float | None = None,
        max_ci: float | None = None,
        scale_invariant: bool = False,
        requires_grad: bool = True,
    ) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float32))
        x_np = np.asarray(x.data, dtype=np.float32)
        was_1d = x_np.ndim == 1
        out_np = self._c.forward(x_np)
        if was_1d:
            out_np = out_np.squeeze(0)
        # See DISLDOLayer.forward's own requires_grad comment.
        if not requires_grad:
            return Tensor(out_np, backend=x.backend)
        out = Tensor(out_np, _children=(x,), _op="disldo8", backend=x.backend)

        def _bwd():
            if out.grad is not None:
                dy = np.asarray(out.grad, dtype=np.float32)
                extra = {}
                if min_decay_frac is not None:
                    extra["min_decay_frac"] = min_decay_frac
                if max_abs_delta is not None:
                    extra["max_abs_delta"] = max_abs_delta
                if max_ci is not None:
                    extra["max_ci"] = max_ci
                if scale_invariant:
                    extra["scale_invariant"] = True
                dx = self._c.backward(
                    x_np,
                    dy,
                    learning_rate,
                    lr_per_row_nnz=lr_per_row_nnz,
                    damp_by_importance=damp_by_importance,
                    **extra,
                )
                if was_1d:
                    dx = dx.squeeze(0)
                _acc(x, dx)

        out._backward = _bwd
        return out

    def apply_dynamic_rank_control(
        self,
        tau_death: float = 0.05,
        tau_active: float = 0.3,
        theta: float = 1e-4,
        seed_scale: float = 0.05,
        scale_grace_period_steps: int = 50,
        additive_grace_period_steps: int = 5000,
    ) -> bool:
        """Same as DISLDOLayer.apply_dynamic_rank_control -- duplicated
        (not inherited) since DISLDOLayer8 doesn't subclass DISLDOLayer
        (separate VALUES_TYPE)."""
        mutated_scale = self._c.apply_dynamic_rank_control(
            tau_death, tau_active, theta, seed_scale, scale_grace_period_steps, scale_grace_period_steps
        )
        mutated_additive = self._c.apply_additive_dynamic_rank_control(
            tau_death, tau_active, theta, seed_scale, additive_grace_period_steps, additive_grace_period_steps
        )
        return mutated_scale or mutated_additive

    def apply_scale_overflow_guard(self, clip: float = 200.0, near: float = 20.0, coef: float = 0.1) -> None:
        """Same as DISLDOLayer.apply_scale_overflow_guard -- duplicated,
        not inherited (DISLDOLayer8 doesn't subclass DISLDOLayer)."""
        _apply_scale_overflow_guard(self._c, clip, near, coef)

    def apply_channel_orthogonality_penalty(self, coef: float = 0.01) -> None:
        """Same as DISLDOLayer.apply_channel_orthogonality_penalty --
        duplicated, not inherited."""
        _apply_channel_orthogonality_penalty(self._c, coef)


class DISLDOLayer8Resync(DISLDOLayer8):
    """DISLDOLayer8 (true C++ E4M3 storage) with the DeferredScaleWrite
    fix: touched entries' stored codes are written out only after
    value_scale/output_scale are both finalized for the backward() call,
    instead of immediately under the stale pre-update scale. See
    docs/research/sparse_rnn.rst:disldo_layer_variants.diagnostic_history."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        rng: np.random.Generator | None = None,
    ):
        self._c = _cpu.SparseLinearLayer8Resync(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)


class DISLDOLayer8AdaMax(DISLDOLayer8):
    """Same as DISLDOLayer8Resync (DeferredScaleWrite also on), but
    value_scale/output_scale use an AdaMax-style decayed running-max
    update instead of RMSprop -- see AdaMaxScalePolicy's own docstring,
    delta_csr_types.hpp."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        rng: np.random.Generator | None = None,
    ):
        self._c = _cpu.SparseLinearLayer8AdaMax(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)


# ══════════════════════════════════════════════════════════════════════════════
#  SISLDOLayer — Sparse Input, Sparse Linear, Dense Output
# ══════════════════════════════════════════════════════════════════════════════


class SISLDOLayer(_SparseLayerBase):
    """Sparse state → state contribution. Input must be a CSR.

    Forward exploits sparse ACTIVATIONS (forward_sparse); backward exploits
    a top-k'd sparse GRADIENT (backward_sparse) -- independent axes, not a
    matched pair. backward_sparse's required dense `x` is reconstructed
    from the same CSR forward_sparse was called with, NOT read from
    `self._c.last_input` (which forward_sparse never populates). See
    docs/research/sparse_rnn.rst:sisldo_layer.dense_x_reconstruction_not_last_input."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        backprop_p: float = 0.03,
        rng: np.random.Generator | None = None,
    ):
        self._c = _cpu.SparseLinearLayer(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)
        self.backprop_p = backprop_p

    def forward(self, x: Tensor, learning_rate: float = 0.0) -> Tensor:
        """x.data must be a CSR. grad flows back as a dense ndarray.
        `learning_rate` is not forwarded to forward_sparse -- weight/
        importance updates happen only in backward_sparse."""
        csr = x.data
        x_dense = csr.to_dense()
        out_np = self._c.forward_sparse(csr.ptrs, csr.indices, csr.values, csr.rows).squeeze(0)
        out = Tensor(out_np, _children=(x,), _op="sisldo", backend=x.backend)

        def _bwd():
            if out.grad is not None:
                dy = np.asarray(out.grad, dtype=np.float32)[np.newaxis, :]
                k = max(1, int(dy.shape[1] * self.backprop_p))
                dp, di, dv = _cpu.dense_to_top_k_csr(dy, k, self._c.num_cpus)
                dx = self._c.backward_sparse(
                    x_dense,
                    dp,
                    di,
                    dv,
                    csr.rows,
                    learning_rate,
                    lr_per_row_nnz=True,
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
    plus per-row value_scale/importance_scale. weights_vals are RAW
    quantized units (true value = weights_vals[i] * value_scale[row_of_i],
    see set_value_scale_raw) -- scale must travel with the weights or a
    reload silently corrupts every true value via the default scale=1.0.

    importance is saved for inspection only -- load_weights has no path to
    restore per-connection importance (see TODO.md), so a reload starts
    it fresh.
    """
    n = layer.n_inputs
    return {
        "ptrs": np.array(layer.ptrs),
        "indices": np.array(layer.indices),
        "weights": np.array(layer.weights_vals),
        "importance": np.array(layer.importance),  # NOT restorable -- see docstring
        "value_scale": np.array([layer.get_value_scale(r) for r in range(n)], dtype=np.float32),
        "importance_scale": np.array([layer.get_importance_scale(r) for r in range(n)], dtype=np.float32),
    }


def _sparse_linear_layer_load_state_dict(layer, d: dict) -> None:
    """Restore weights and per-row value_scale/importance_scale onto an
    already-constructed layer of the matching shape. Does not restore
    importance (see _sparse_linear_layer_state_dict)."""
    layer.load_weights(
        np.asarray(d["ptrs"], dtype=np.int32),
        np.asarray(d["indices"], dtype=np.int32),
        np.asarray(d["weights"], dtype=np.float32),
    )
    for r in range(layer.n_inputs):
        layer.set_value_scale_raw(r, float(d["value_scale"][r]))
        layer.set_importance_scale_raw(r, float(d["importance_scale"][r]))


def fit_rank1_scale_envelope(row_idx, col_idx, abs_vals, n_rows: int, n_cols: int, n_iters: int = 6):
    """
    Alternating max-fit producing a rank-1 (outer-product) envelope:
    row_scale[r] * col_scale[c] >= |M[r, c]| for every entry, using
    O(rows+cols) parameters instead of O(rows*cols) (Sinkhorn-style, but
    max- rather than sum-based).

    Takes M's nonzero entries as COO-style triplets (row_idx, col_idx,
    abs_vals), never a dense matrix -- a folded/stacked layer's matrix
    can be too large to densify safely. Returns (row_scale, col_scale),
    float32. See
    docs/research/sparse_rnn.rst:fit_rank1_scale_envelope.coo_no_densify.
    """
    import numpy as np

    col_scale = np.ones(n_cols, dtype=np.float32)
    row_scale = np.ones(n_rows, dtype=np.float32)
    for _ in range(n_iters):
        row_scale = np.full(n_rows, 1e-12, dtype=np.float32)
        np.maximum.at(row_scale, row_idx, abs_vals / col_scale[col_idx])
        col_scale = np.full(n_cols, 1e-12, dtype=np.float32)
        np.maximum.at(col_scale, col_idx, abs_vals / row_scale[row_idx])
    return row_scale, col_scale


# ══════════════════════════════════════════════════════════════════════════════
#  FoldedLayer — runtime sili Module for a converted folded transformer block
# ══════════════════════════════════════════════════════════════════════════════


class FoldedLayer(Module):
    """
    Runtime sili layer for a folded transformer block.

    All N original transformer layers are stacked into ONE SparseLinearLayer
    per weight suffix (Q, K, V, MLP, etc.). A single forward() call replaces
    N sequential matmuls. Weights live entirely in C++; parameters() returns
    [] but the layer IS in the autograd graph via a _backward closure.

    No torch dependency in forward/backward -- from_descriptor() uses torch
    once at construction time only.

    Shape contract (same as RNNFoldedBlock.forward):
        input  [batch, in_dim]   -> output [batch, out_dim]
    The stacked weights map in_dim -> n_folds*out_dim internally; the fold
    dimension is summed away on the way out (reshape + sum(axis=1)).
    """

    def __init__(
        self,
        layers: dict,  # {suffix: SparseLinearLayer}
        n_folds: int,
        out_dims: dict,  # {suffix: out_dim}
        learning_rate: float = 0.01,
    ):
        self._sili_layers = layers
        self._n_folds = n_folds
        self._out_dims = out_dims
        self.lr = learning_rate

    # ── Factory ------------------------------------------------------------------

    @classmethod
    def from_descriptor(
        cls,
        descriptor,
        learning_rate: float = 0.01,
        num_cpus: int = 4,
        max_row_weights: int = 0,
        bytes_per_row: int = 0,
        value_scale_mode: str = "per_row",
        rank1_iters: int = 6,
        compact_after_build: bool = True,
    ) -> FoldedLayer:
        """
        Build a FoldedLayer from a FoldedBlockDescriptor.

        args:
          max_row_weights -- peak connections per row for synaptogenesis.
                             0 = n_out (the absolute ceiling).
          bytes_per_row   -- index byte budget per row. 0 = compute from
                             max_row_weights and the typical ULEB128 cost
                             for this layer's column range, plus a small
                             growth margin. Pass explicitly to override.
          value_scale_mode -- "per_row" (default): one value_scale per
                             input row. "rank1": also fits a per-output-
                             column scale (fit_rank1_scale_envelope).
                             Prefer "rank1" for real pretrained-weight
                             conversion.
          rank1_iters      -- alternating-fit iterations for "rank1" mode.
          compact_after_build -- strip equalize_to_capacity's per-row growth
                             headroom right after loading (default True).
                             See docs/research/sparse_rnn.rst:folded_layer.csr_layout_conversion_no_densify.
        """
        import warnings

        import numpy as np

        # the compute. Do not use torch in forward/backward paths.
        warnings.filterwarnings("ignore")
        _FP4_MAX = 6.0
        if value_scale_mode not in ("per_row", "rank1"):
            raise ValueError(f"value_scale_mode must be 'per_row' or 'rank1', got {value_scale_mode!r}")

        layers = {}
        for suffix, csr in descriptor.stacked_weights.items():
            # csr.t() + .to_sparse_csr(): metadata-only CSC relabel, then
            # nnz-proportional reorg into row-major order -- NEVER
            # .to_dense() here (can be too large to safely materialize).
            # See docs/research/sparse_rnn.rst:folded_layer.csr_layout_conversion_no_densify.
            csr_t = csr.t().to_sparse_csr()
            n_in = int(csr_t.shape[0])
            n_out = int(csr_t.shape[1])
            # Budget sized for the fully-connected maximum (n_in*n_out),
            # not current nnz -- the fixed total equalizer_step()
            # redistributes within, never grows.
            budget = n_in * n_out
            layer = _cpu.SparseLinearLayer(n_in, n_out, budget, num_cpus)
            ptrs = csr_t.crow_indices().numpy().astype(np.int32)
            idx = csr_t.col_indices().numpy().astype(np.int32)
            vals = csr_t.values().float().numpy().copy()

            if value_scale_mode == "rank1":
                # One scale per input row AND one per output column,
                # fit directly from the nonzero triplets -- no dense
                # intermediate (see fit_rank1_scale_envelope).
                row_of_nnz = np.repeat(np.arange(n_in, dtype=np.int64), np.diff(ptrs))
                row_env, col_env = fit_rank1_scale_envelope(
                    row_of_nnz, idx.astype(np.int64), np.abs(vals), n_in, n_out, n_iters=rank1_iters
                )
                row_scales = (row_env / _FP4_MAX).astype(np.float32)
                col_scales = col_env.astype(np.float32)
                combined = row_scales[row_of_nnz] * col_scales[idx]
                nonzero_combined = combined > 0
                vals[nonzero_combined] /= combined[nonzero_combined]
            else:
                # Per-row: map each row's max-abs to FP4_MAX for full
                # quantizer resolution.
                row_scales = np.ones(n_in, dtype=np.float32)
                col_scales = np.ones(n_out, dtype=np.float32)  # left at default (1.0) in this mode
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
            for c in range(n_out):
                if col_scales[c] != 1.0:
                    layer.set_output_scale_raw(c, col_scales[c])

            # Per-row importance_scale: same FP4 representability problem as
            # value_scale but for importance (raw lr=0.01 updates round to
            # 0 under FP4's 0.5 minimum nonzero). See
            # docs/research/sparse_rnn.rst:folded_layer.csr_layout_conversion_no_densify.
            imp_scale = learning_rate / _FP4_MAX
            for r in range(n_in):
                layer.set_importance_scale_raw(r, imp_scale)

            # Choose capacity targets for equalize_to_capacity.
            # max_row_weights defaults to n_out (absolute ceiling).
            mrw = max_row_weights if max_row_weights > 0 else n_out

            # bytes_per_row: typical (not worst-case) ULEB128 cost for this
            # column range, plus a small growth margin. Pass explicitly to
            # override (e.g. worst-case: mrw*5).
            if bytes_per_row > 0:
                bpr = bytes_per_row
            else:
                bits = max(1, n_out - 1).bit_length()
                typ = (bits + 6) // 7  # ceil(bits / 7)
                bpr = mrw * typ + 4  # +4 bytes margin per step

            layer.equalize_to_capacity(mrw, bpr)
            if compact_after_build:
                layer.compact()

            layers[suffix] = layer

        return cls(layers, descriptor.n_folds, descriptor.out_dims, learning_rate)

    # ── Module interface ---------------------------------------------------------

    def parameters(self) -> list:
        return []  # weights live in C++, not in the Tensor graph

    # ── Properties --------------------------------------------------------------

    @property
    def in_features(self) -> int:
        return next(iter(self._sili_layers.values())).n_inputs

    @property
    def out_features(self) -> int:
        return next(iter(self._out_dims.values()))

    # ── Forward ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        x: sili Tensor [batch, in_dim]  (or [in_dim] -- squeezed automatically)
        Returns: sili Tensor [batch, out_dim]

        Wired into sili autograd: loss.backward() propagates through this
        layer automatically. forward_dense(x) is a pure computation (no
        side effects); backward_dense(dy, lr), called by loss.backward(),
        is the only place weight values AND importance change.
        """
        x_np = np.asarray(x.data, dtype=np.float32)
        squeezed = x_np.ndim == 1
        if squeezed:
            x_np = x_np[np.newaxis, :]
        batch = x_np.shape[0]
        out_dim = next(iter(self._out_dims.values()))
        lr = self.lr  # used by the backward closure below, not forward

        # Single call per suffix -- the full stacked matrix is one layer.
        raw_parts = [layer.forward_dense(x_np) for layer in self._sili_layers.values()]
        raw_np = sum(raw_parts)  # [batch, n_folds * out_dim]

        # Fold sum: [batch, n_folds, out_dim] -> [batch, out_dim]
        summed = raw_np.reshape(batch, self._n_folds, out_dim).sum(axis=1)
        if squeezed:
            summed = summed.squeeze(0)

        out = Tensor(summed, _children=(x,), _op="folded", backend=x.backend)

        # Capture loop variables for the closure (Python late-binding risk).
        _layers = list(self._sili_layers.values())
        _n_folds = self._n_folds
        _sq = squeezed

        def _bwd():
            if out.grad is None:
                return
            dy_np = np.asarray(out.grad, dtype=np.float32)
            if dy_np.ndim == 1:
                dy_np = dy_np[np.newaxis, :]
            _batch = dy_np.shape[0]

            # Backward of fold reshape+sum:
            # grad of sum is 1 to each summand -> broadcast dy to all n_folds slots.
            dy_raw = (
                np.tile(dy_np.reshape(_batch, 1, out_dim), (1, _n_folds, 1))
                .reshape(_batch, _n_folds * out_dim)
                .astype(np.float32)
            )

            # Each suffix layer gets the same dy_raw; accumulate dx.
            dx_parts = [layer.backward_dense(dy_raw, lr, lr_per_row_nnz=True) for layer in _layers]
            dx_np = sum(dx_parts).reshape(_batch, -1)
            if _sq:
                dx_np = dx_np.squeeze(0)
            _acc(x, dx_np)

        out._backward = _bwd
        return out

    # ── Synaptogenesis -----------------------------------------------------------

    def synaptogenesis(
        self,
        k: int,
        importance_cutoff: float,
        max_row_weights: int,
        rows_per_call: int = 0,
    ) -> None:
        """
        Grow and prune connections across all suffix layers. Each call to
        synap_step() advances ONE row of the layer's internal cursor:
        remove synapses below importance_cutoff, then grow new ones (from
        the top-k probes) until the row reaches max_row_weights.

        k: probes to build, rule of thumb k ~ 4*max_row_weights.
        importance_cutoff: FP4 stored-unit threshold (multiply by
        get_importance_scale(r) for true units).
        max_row_weights: target connections per row after this sweep.
        rows_per_call: 0 (default) = full sweep; N > 0 = staggered mode,
        advancing exactly N rows per call.

        Call AFTER backward() and BEFORE the next forward(). Accumulators
        are zeroed at the end of each call.
        """
        for layer in self._sili_layers.values():
            layer.build_probes(k)
            n = rows_per_call if rows_per_call > 0 else layer.n_inputs
            for _ in range(n):
                layer.synap_step(importance_cutoff, max_row_weights)
            layer.equalizer_step()  # staggered 1-row redistribution
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
            d["lr"] = np.array([self.lr], dtype=np.float32)
            out[suffix] = d
        return out

    def load_state_dict(self, d: dict) -> None:
        """Restore weights + per-row value_scale/importance_scale for
        every suffix layer. Does not restore importance (see
        _sparse_linear_layer_state_dict)."""
        for suffix, sub in d.items():
            _sparse_linear_layer_load_state_dict(self._sili_layers[suffix], sub)


# ══════════════════════════════════════════════════════════════════════════════
#  FoldedColumnLayer — FoldedLayer variant for the column-averaging mechanism
# ══════════════════════════════════════════════════════════════════════════════


class FoldedColumnLayer(FoldedLayer):
    """
    FoldedLayer variant for the column-averaging mechanism: retains the
    pre-sum [n_folds*out_dim] tensor instead of collapsing the fold axis,
    paired with a `recurrent` layer (same input_proj+recurrent split
    SparseRNNCell uses, built on SparseLinearLayer).

    in_proj(x): real pretrained per-fold-step matrices (stacked), plus a
    zero-valued trainable skip band from input to every fold-depth column.
    recurrent(state): from-scratch banded matrix mapping this layer's
    output space back to itself (fold step i -> i+1 and nearby columns).
    forward(x, state) = in_proj(x) + recurrent(state), the new state.
    state defaults to zero when not given.

    Feed forward()'s output to sili.energy.column_averaging_loss, after
    EnergyDynamics gating (not this layer's raw output).
    """

    @classmethod
    def from_descriptor(
        cls,
        descriptor,
        learning_rate: float = 0.01,
        num_cpus: int = 4,
        max_row_weights: int = 0,
        bytes_per_row: int = 0,
        recurrent_bandwidth: int | None = None,
        existing_recurrent=None,
        existing_recurrent_prefer: str = "b",
        input_skip_bandwidth: int | None = None,
    ) -> FoldedColumnLayer:
        """
        Like FoldedLayer.from_descriptor, plus builds `recurrent` sized to
        this layer's [n_folds*out_dim] output space, and unconditionally
        unions a zero-valued input->column skip pre-seed onto every
        suffix's in_proj weights (see build_input_skip_preseed).

        recurrent_bandwidth: forwarded to build_fold_skip_layer's `bandwidth`.
        existing_recurrent/existing_recurrent_prefer: forwarded to
        build_fold_skip_layer's `existing`/`existing_prefer` -- pass a
        previously-saved recurrent CSR (state_dict_to_true_csr(...)) to
        preserve trained skip weights across re-runs.
        input_skip_bandwidth: forwarded to build_input_skip_preseed's
        `bandwidth` per suffix -- real weights always win at any overlap.
        """
        obj = super().from_descriptor(
            descriptor,
            learning_rate=learning_rate,
            num_cpus=num_cpus,
            max_row_weights=max_row_weights,
            bytes_per_row=bytes_per_row,
        )
        obj.recurrent = build_fold_skip_layer(
            obj._n_folds,
            obj.column_width,
            num_cpus=num_cpus,
            bandwidth=recurrent_bandwidth,
            expected_lr=learning_rate,
            existing=existing_recurrent,
            existing_prefer=existing_recurrent_prefer,
        )

        out_dim = obj.column_width
        for suffix, layer in list(obj._sili_layers.items()):
            preseed_ptrs, preseed_idx, _ = build_input_skip_preseed(
                layer.n_inputs, obj._n_folds, out_dim, bandwidth=input_skip_bandwidth
            )
            obj._sili_layers[suffix] = _rebuild_layer_with_preseed(
                layer,
                preseed_ptrs,
                preseed_idx,
                num_cpus=num_cpus,
                expected_lr=learning_rate,
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

    def in_proj(self, x: Tensor) -> Tensor:
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
        lr = self.lr  # used by the backward closure below, not forward

        raw_parts = [layer.forward_dense(x_np) for layer in self._sili_layers.values()]
        raw_np = sum(raw_parts)  # [batch, n_folds*out_dim] -- kept as-is
        if squeezed:
            raw_np = raw_np.squeeze(0)

        out = Tensor(raw_np, _children=(x,), _op="folded_column_in_proj", backend=x.backend)

        _layers = list(self._sili_layers.values())
        _sq = squeezed

        def _bwd():
            if out.grad is None:
                return
            dy_np = np.asarray(out.grad, dtype=np.float32)
            if dy_np.ndim == 1:
                dy_np = dy_np[np.newaxis, :]
            dx_parts = [layer.backward_dense(dy_np, lr, lr_per_row_nnz=True) for layer in _layers]
            dx_np = sum(dx_parts).reshape(dy_np.shape[0], -1)
            if _sq:
                dx_np = dx_np.squeeze(0)
            _acc(x, dx_np)

        out._backward = _bwd
        return out

    def forward(self, x: Tensor, state: Tensor = None) -> Tensor:
        """
        h = in_proj(x) + recurrent(state) -- same pattern as
        SparseRNNCell.forward. Returns the new state; feed it back in as
        `state` on the next call. state=None (default) uses zeros -- true
        step-0, matching RNNFoldedBlock.forward's state=0 start.
        """
        raw = self.in_proj(x)
        if state is None:
            state = Tensor(np.zeros(self.out_features, dtype=np.float32), backend=x.backend)
        rec = apply_fold_skip(self.recurrent, state, lr=self.lr)
        return raw + rec

    def state_dict(self) -> dict:
        """FoldedLayer.state_dict() (in_proj) plus recurrent."""
        out = super().state_dict()
        out["recurrent"] = _sparse_linear_layer_state_dict(self.recurrent)
        return out

    def load_state_dict(self, d: dict) -> None:
        """Restore in_proj (via FoldedLayer.load_state_dict) and
        recurrent."""
        super().load_state_dict({k: v for k, v in d.items() if k != "recurrent"})
        _sparse_linear_layer_load_state_dict(self.recurrent, d["recurrent"])


def _build_banded_csr(total: int, bandwidth: int):
    """Zero-valued [total, total] banded-diagonal CSR: row r connects to
    columns c with abs(r-c) < bandwidth, clipped to [0, total). Split out
    of build_fold_skip_layer so the pattern can be unioned with an
    existing CSR (see csr_union) before any SparseLinearLayer is built."""
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
        idx[pos : pos + n] = np.arange(lo[r], hi[r] + 1, dtype=np.int32)
        pos += n

    return ptrs.astype(np.int32), idx, np.zeros(nnz, dtype=np.float32)


def _build_rectangular_banded_csr(rows: int, cols: int, bandwidth: int):
    """Zero-valued banded CSR from `rows` positions to `cols` positions --
    generalizes _build_banded_csr to a rectangular shape, where rows and
    cols differ in size so there's no exact diagonal. Row r connects to
    columns near round(r * cols / rows) (a geometric, proportional-position
    diagonal), within `bandwidth` either side, clipped to [0, cols)."""
    assert rows >= 1 and cols >= 1 and bandwidth >= 1
    positions = np.arange(rows)
    centers = (positions.astype(np.float64) * cols / rows).astype(np.int64)
    lo = np.clip(centers - bandwidth + 1, 0, None)
    hi = np.clip(centers + bandwidth - 1, None, cols - 1)
    row_lengths = (hi - lo + 1).astype(np.int64)

    ptrs = np.zeros(rows + 1, dtype=np.int64)
    ptrs[1:] = np.cumsum(row_lengths)
    nnz = int(ptrs[-1])

    idx = np.empty(nnz, dtype=np.int32)
    pos = 0
    for r in range(rows):
        n = int(row_lengths[r])
        idx[pos : pos + n] = np.arange(lo[r], hi[r] + 1, dtype=np.int32)
        pos += n

    return ptrs.astype(np.int32), idx, np.zeros(nnz, dtype=np.float32)


def state_dict_to_true_csr(d: dict):
    """
    Convert one _sparse_linear_layer_state_dict()-shaped dict (e.g.
    layer.state_dict()["recurrent"]) into a (ptrs, idx, vals) CSR with
    vals in TRUE units -- the format csr_union/build_fold_skip_layer's
    `existing` expects, since raw FP4 units from differently-scaled
    sources aren't directly comparable.
    """
    ptrs = np.asarray(d["ptrs"], dtype=np.int32)
    idx = np.asarray(d["indices"], dtype=np.int32)
    raw = np.asarray(d["weights"], dtype=np.float32)
    scale = np.asarray(d["value_scale"], dtype=np.float32)
    vals = raw.copy()
    for r in range(len(ptrs) - 1):
        start, end = int(ptrs[r]), int(ptrs[r + 1])
        if end > start:
            vals[start:end] = raw[start:end] * scale[r]
    return ptrs, idx, vals


def csr_union(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b, n_rows: int, prefer: str = "a", num_cpus: int = 4):
    """
    Merge two CSRs of the SAME shape into one holding the union of their
    nonzero positions. `vals_a`/`vals_b` must already be in TRUE units
    (not raw FP4 levels). Where both inputs have an entry at (row, col),
    `prefer` decides the result: 'a' (default) keeps A, 'b' keeps B, 'sum'
    adds them. Construction/loading time only. OpenMP-parallel
    (_cpu.csr_union) -- each row is an independent two-pointer merge.
    """
    assert prefer in ("a", "b", "sum")
    return _cpu.csr_union(
        np.asarray(ptrs_a, dtype=np.int32),
        np.asarray(idx_a, dtype=np.int32),
        np.asarray(vals_a, dtype=np.float32),
        np.asarray(ptrs_b, dtype=np.int32),
        np.asarray(idx_b, dtype=np.int32),
        np.asarray(vals_b, dtype=np.float32),
        int(n_rows),
        prefer,
        num_cpus,
    )


def build_fold_skip_layer(
    n_folds: int,
    out_dim: int,
    num_cpus: int = 4,
    bandwidth: int | None = None,
    headroom_fraction: float = 0.5,
    expected_lr: float = 0.01,
    existing=None,
    existing_prefer: str = "b",
) -> _cpu.SparseLinearLayer:
    """
    Sparse layer mapping a FoldedColumnLayer's own [n_folds*out_dim]
    output space back to itself: skip connections between virtual
    (fold-depth) layers, pre-seeded as a zero-valued banded pattern
    (approximates RNNFoldedBlock.forward's fold recurrence).

    bandwidth: connect flat positions r, c whenever abs(r-c) < bandwidth.
    Default (None) uses out_dim.
    expected_lr: fallback per-row value_scale (expected_lr/FP4_MAX) for
    still-all-zero rows, so gradient updates don't round back to zero
    under FP4; rows with real nonzero values get max_abs/FP4_MAX instead.
    existing: optional (ptrs, idx, vals) CSR in TRUE units, unioned with
    the fresh band before construction, to preserve a previously-trained
    recurrent CSR across re-runs. existing_prefer: default 'b' (existing's
    trained value wins over the fresh zero).
    """
    assert n_folds >= 1 and out_dim >= 1
    bw = out_dim if bandwidth is None else bandwidth
    assert bw >= 1
    total = n_folds * out_dim

    ptrs, idx, vals = _build_banded_csr(total, bw)

    if existing is not None:
        ex_ptrs, ex_idx, ex_vals = existing
        ptrs, idx, vals = csr_union(
            ptrs, idx, vals, ex_ptrs, ex_idx, ex_vals, total, prefer=existing_prefer, num_cpus=num_cpus
        )

    nnz = len(idx)
    row_lengths = ptrs[1:] - ptrs[:-1]
    max_row_weights = int(row_lengths.max()) if nnz > 0 else 1
    budget = nnz + int(headroom_fraction * nnz) + total

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


def build_input_skip_preseed(n_in: int, n_folds: int, out_dim: int, bandwidth: int | None = None):
    """
    Zero-valued, trainable skip connections from external input directly to
    every fold-depth column -- the in_proj analogue of build_fold_skip_layer
    (which does this for `recurrent`). Without this, in_proj's only
    connections are the original dense LLM's fixed per-layer weights, with
    no room for training to grow a new input->column path.

    Returns a (ptrs, idx, vals) CSR shaped [n_in, n_folds*out_dim], all
    vals 0.0 -- meant to be unioned onto a suffix's real stacked weights
    (see FoldedColumnLayer.from_descriptor's `input_skip_bandwidth`).

    bandwidth: geometric banding (see _build_rectangular_banded_csr) --
    input dim i connects to columns near round(i * n_folds*out_dim / n_in),
    within `bandwidth` either side. Default (None) uses out_dim.
    """
    total_out = n_folds * out_dim
    bw = out_dim if bandwidth is None else bandwidth
    return _build_rectangular_banded_csr(n_in, total_out, bw)


def _rebuild_layer_with_preseed(
    layer, preseed_ptrs, preseed_idx, num_cpus: int, expected_lr: float, headroom_fraction: float = 0.5
):
    """
    Rebuild a SparseLinearLayer, unioning a zero-valued pre-seed CSR's
    structural positions onto the layer's real (already-trained/pretrained)
    weights. Real values always win at any overlap (prefer="a") -- the
    pre-seed only adds new zero-valued positions, never clobbers real data.
    Budget/scale handling mirrors build_fold_skip_layer (nnz-based
    headroom, per-row FP4 rescaling recomputed on the merged row).
    """
    n_in, n_out = layer.n_inputs, layer.n_outputs
    real_ptrs, real_idx, real_vals = state_dict_to_true_csr(_sparse_linear_layer_state_dict(layer))
    preseed_vals = np.zeros(len(preseed_idx), dtype=np.float32)
    ptrs, idx, vals = csr_union(
        real_ptrs, real_idx, real_vals, preseed_ptrs, preseed_idx, preseed_vals, n_in, prefer="a", num_cpus=num_cpus
    )

    nnz = len(idx)
    row_lengths = ptrs[1:] - ptrs[:-1]
    max_row_weights = int(row_lengths.max()) if nnz > 0 else 1
    budget = nnz + int(headroom_fraction * nnz) + n_in

    _FP4_MAX = 6.0
    row_scales = np.full(n_in, expected_lr / _FP4_MAX, dtype=np.float32)
    for r in range(n_in):
        start, end = int(ptrs[r]), int(ptrs[r + 1])
        if end > start:
            max_abs = float(np.abs(vals[start:end]).max())
            if max_abs > 0.0:
                row_scales[r] = max_abs / _FP4_MAX
                vals[start:end] = vals[start:end] / row_scales[r]

    new_layer = _cpu.SparseLinearLayer(n_in, n_out, budget, num_cpus)
    new_layer.load_weights(ptrs, idx, vals)
    new_layer.equalize_to_capacity(max_row_weights)
    for r in range(n_in):
        new_layer.set_value_scale_raw(r, float(row_scales[r]))
        new_layer.set_importance_scale_raw(r, expected_lr / _FP4_MAX)
    return new_layer


def apply_fold_skip(skip_layer, x: Tensor, lr: float = 0.01) -> Tensor:
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

    out_np = skip_layer.forward_dense(x_np)
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
    class mirrors that cursor in Python and provides memory statistics.

    Call mem.step() once per training step (after backward, alongside
    synap_schedule.step()) so blank space is continuously redistributed --
    synaptogenesis on a row with no blank space will throw otherwise.
    """

    def __init__(self, layer):
        self._layer = layer  # SparseLinearLayer (_cpu object)
        self._cursor = 0  # mirrors C++ _equalize_row
        self._calls = 0

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
    with a (optionally varying) max_row_weights target:
        max_row_weights(t) = round(base * (1 + amplitude * sin(2*pi*t/period)))
    amplitude=0 (default) is constant at base. A nonzero amplitude exercises
    both growth and pruning -- a clean regression check (nnz_total should
    oscillate around base * n_rows over many cycles). Call sched.step()
    once per training step, after backward.
    """

    def __init__(
        self,
        layer: FoldedLayer,
        base_connections: int,
        k_factor: int = 4,  # probes = k_factor * max_row_weights
        importance_cutoff: float = 0.0,  # stored-unit importance threshold
        amplitude: float = 0.0,  # 0 = constant, 0.3 = +-30%
        period: int = 200,  # steps per full sine cycle
        every_n_steps: int = 20,  # run synaptogenesis every N steps
        rows_per_call: int = 0,  # 0 = full sweep
    ):
        self._layer = layer
        self._base = base_connections
        self._k_factor = k_factor
        self._importance_cutoff = importance_cutoff
        self._amplitude = amplitude
        self._period = period
        self._every = every_n_steps
        self._rows_per_call = rows_per_call
        self._t = 0  # training steps counted
        self._synap_t = 0  # synaptogenesis calls counted

    def current_max_row_weights(self) -> int:
        """Current target based on the sine wave at this step."""
        if self._amplitude == 0.0:
            return self._base
        import math

        factor = 1.0 + self._amplitude * math.sin(2.0 * math.pi * self._synap_t / self._period)
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
        k = max(1, self._k_factor * mrw)
        self._layer.synaptogenesis(k, self._importance_cutoff, mrw, self._rows_per_call)
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

    Returns (h_out: Tensor, aux_loss: Tensor, actual_p: float) -- h_out
    (dense) is returned unchanged as the new state.

    Unifies the two sparsification passes: the CSR fed into `recurrent()`
    at the top of the NEXT call is built from THIS call's own energy-gating
    decision (cached as `_prev_kept_indices`/`_prev_h_dense`), not an
    independent top-k re-derivation. Cache invalidated on reset()/whenever
    the caller hands in a state this cell didn't itself just produce.

    Branching-ratio measurement: recurrent-only activity is measured on
    recurrent_out BEFORE it's summed with input_proj(obs), so a genuinely
    self-propagating recurrent pathway can be distinguished from input
    alone carrying activity. `branching_tracker` selects "window" (hard
    sliding window, supports avalanche_sizes()) or "ema" (default, O(1)
    memory, tunable via `branching_ema_alpha`).

    See docs/research/sparse_rnn.rst:sparse_rnn_cell.overview.
    """

    def __init__(
        self,
        n_inputs: int,
        state_size: int,
        max_weights: int,
        num_cpus: int = 4,
        percent_active: float = 0.03,
        dynamic_density_from_branching_ratio: bool = False,
        branching_tracker: str = "ema",
        branching_window: int = 200,
        branching_ema_alpha: float = 0.05,
    ):
        assert branching_tracker in ("window", "ema"), (
            f"branching_tracker must be 'window' or 'ema', got {branching_tracker!r}"
        )
        r = percent_active / 0.02
        self.input_proj = DISLDOLayer(n_inputs, state_size, max_weights, num_cpus)
        self.recurrent = SISLDOLayer(state_size, state_size, max_weights, num_cpus, backprop_p=percent_active)
        # density IS the target active fraction; p is a hard compute-limit
        # ceiling clearly above it (~5x). See
        # docs/research/sparse_rnn.rst:sparse_rnn_cell.density_p_inversion_bug.
        density = min(0.9, percent_active)
        p = min(1.0, percent_active * 5.0)
        # activation_cost=0.08*r grows unbounded with percent_active --
        # clamped to EnergyDynamics's asserted [0.01, 0.5] range so small
        # states (needing higher percent_active) don't fail construction.
        activation_cost = min(0.5, max(0.01, 0.08 * r))
        self.energy = EnergyDynamics(
            drive=0.08 * percent_active * r,
            activation_cost=activation_cost,
            density=density,
            exploration=0.001 * r,
            reactivity=0.01 * r,
            precision=0.04 * r,
            setpoint=1.0,
            activation_threshold=1e-4,
            p=p,
        )
        self.state_size = state_size
        self._percent_active = percent_active

        # Recurrent-only branching-ratio measurement and its optional
        # (default-off) use to nudge the KL density target -- see
        # docs/research/sparse_rnn.rst:sparse_rnn_cell.dynamic_density_nudge.
        if branching_tracker == "window":
            self.branching_recurrent = BranchingRatioTracker(window=branching_window)
        else:
            self.branching_recurrent = EMABranchingRatioTracker(alpha=branching_ema_alpha)
        self.branching_tracker_mode = branching_tracker
        self.dynamic_density_from_branching_ratio = bool(dynamic_density_from_branching_ratio)

        # Cache for unifying the sparsification passes -- see class docstring.
        self._prev_kept_indices: np.ndarray | None = None
        self._prev_h_dense: np.ndarray | None = None

    def parameters(self) -> list:
        return []

    def forward(
        self, obs: Tensor, state: Tensor, learning_rate: float = 0.0, requires_grad: bool = True
    ) -> tuple[Tensor, Tensor, float]:
        # state.data is a CSR only if the caller explicitly handed us one;
        # normal path builds the CSR from this cell's own cached gating
        # decision, falling back to independent top-k at true step-0.
        if not isinstance(state.data, CSR):
            if self._prev_kept_indices is not None:
                state_csr = CSR.from_kept_indices(self._prev_kept_indices, self._prev_h_dense, cols=self.state_size)
            else:
                state_csr = CSR.from_dense(
                    np.asarray(state.data, dtype=np.float32),
                    p=self._percent_active,
                    num_cpus=self.input_proj.num_cpus,
                )
            state = state_csr.as_tensor(state.backend)

        # Measure the recurrent pathway's OWN activity before it's mixed
        # with input_proj(obs) -- see class docstring.
        recurrent_out = self.recurrent(state, learning_rate)
        recurrent_activity = float(
            np.sum(np.abs(np.asarray(recurrent_out.data, dtype=np.float32)) > self.energy.activation_threshold)
        )
        self.branching_recurrent.update(recurrent_activity)

        h = self.input_proj(obs, learning_rate, requires_grad=requires_grad) + recurrent_out

        density_override = None
        if self.dynamic_density_from_branching_ratio:
            m = self.branching_recurrent.branching_ratio()
            if m is not None:
                # First-cut proportional nudge, not a derivation -- see
                # docs/research/sparse_rnn.rst:sparse_rnn_cell.dynamic_density_nudge.
                m_target = 0.98
                density_override = float(
                    np.clip(
                        self.energy.density * (1.0 + 2.0 * (m - m_target)),
                        self.energy.density * 0.5,
                        self.energy.density * 2.0,
                    )
                )

        new_state, aux_loss, actual_p = self.energy(h, density_override=density_override)

        # Cache this call's gating decision for the NEXT call's CSR
        # construction (pre-gating h, not h_out).
        self._prev_kept_indices = self.energy.kept_indices
        self._prev_h_dense = np.asarray(h.data, dtype=np.float32).ravel().copy()

        return new_state, aux_loss, actual_p

    def reset(self):
        # Invalidate the sparsification-pass cache -- the next state the
        # caller hands in didn't come from this cell's own last forward.
        self._prev_kept_indices = None
        self._prev_h_dense = None
        self.branching_recurrent.reset()

    def synaptogenesis(self, k: int, importance_cutoff: float, max_weights: int):
        """Structural growth + memory rebalancing for both sub-layers --
        see _SparseLayerBase.synaptogenesis. max_weights is the TOTAL
        per-layer budget, converted to a PER-ROW cap for each sub-layer
        here using its own in_features (input_proj/recurrent generally
        have different row counts)."""
        input_proj_cap = max(1, max_weights // self.input_proj.in_features)
        recurrent_cap = max(1, max_weights // self.recurrent.in_features)
        self.input_proj.synaptogenesis(k, importance_cutoff, input_proj_cap)
        self.recurrent.synaptogenesis(k, importance_cutoff, recurrent_cap)

    def state_dict(self) -> dict:
        return {
            "input_proj": self.input_proj.state_dict(),
            "recurrent": self.recurrent.state_dict(),
            "energy": self.energy.state_dict(),
        }

    def load_state_dict(self, d: dict):
        self.input_proj.load_state_dict(d["input_proj"])
        self.recurrent.load_state_dict(d["recurrent"])
        self.energy.load_state_dict(d["energy"])
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

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        state_size: int,
        max_weights: int,
        num_cpus: int = 4,
        percent_active: float = 0.03,
        lr: float = 1e-3,
        importance_cutoff: float = 0.01,
        synaptogenesis_k: int = 4,
    ):
        # synaptogenesis_k: build_probes(k) is O(k^2), not O(k) -- see
        # docs/research/sparse_rnn.rst:sparse_rnn_agent.build_probes_k_scaling.
        assert n_actions <= state_size

        self.cell = SparseRNNCell(n_inputs, state_size, max_weights, num_cpus, percent_active)

        self.state = Tensor(np.zeros(state_size, dtype=np.float32))

        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.state_size = state_size
        self.max_weights = max_weights

        self.lr = lr
        self.importance_cutoff = importance_cutoff
        self.synaptogenesis_k = synaptogenesis_k

        self._step_count = 0
        self.aux_loss: Tensor | None = None
        self._actual_p: float = 0.0

    def parameters(self) -> list:
        return []

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, obs: Tensor) -> int:
        """Run one step. State stays in the autograd graph (use for multi-step
        BPTT). Threads self.lr into the cell so weight VALUE updates fire
        inline whenever backward() eventually runs -- forward() itself never
        updates anything."""
        h_out, aux_loss, actual_p = self.cell(obs, self.state, self.lr)
        self.state = h_out
        self.aux_loss = aux_loss
        self._actual_p = actual_p
        return int(np.argmax(np.asarray(self.state.data, dtype=np.float32)[: self.n_actions]))

    def train_step(self, obs: Tensor) -> int:
        """
        BPTT=1 convenience wrapper. Detaches state before forward so gradients
        don't flow across steps, then runs aux_loss.backward() (fires the
        inline weight-value updates) and step() (structural growth only).

        Use aux_loss directly before calling this if you want to add a task loss:
            action   = agent.forward(obs)
            combined = agent.aux_loss + task_loss(action, target)
            combined.backward()
            agent.step()
        """
        self.state = self.state.detach()
        action = self.forward(obs)
        self.aux_loss.backward()
        self.step()
        return action

    # ── Optimization ─────────────────────────────────────────────────────────

    def step(self):
        """Structural growth + memory rebalancing only -- weight VALUE
        updates already happened inline during backward(). Called every
        step, not throttled by an "every N steps" cadence (would
        reintroduce the lag spikes this design avoids)."""
        self.cell.synaptogenesis(self.synaptogenesis_k, self.importance_cutoff, self.max_weights)
        self._step_count += 1

    def reset_state(self):
        self.state = Tensor(np.zeros(self.state_size, dtype=np.float32))
        self.aux_loss = None
        self.cell.reset()

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        d = self.cell.state_dict()
        flat = {
            "_step_count": np.array([self._step_count]),
            "_state": np.asarray(self.state.data, dtype=np.float32),
        }
        for section, sub in d.items():
            for k, v in sub.items():
                flat[f"{section}__{k}"] = v
        np.savez_compressed(path, **flat)

    def load(self, path: str):
        raw = np.load(path, allow_pickle=False)
        self._step_count = int(raw["_step_count"][0])
        self.state = Tensor(raw["_state"].copy())

        def _section(prefix):
            return {k[len(prefix) + 2 :]: raw[k] for k in raw if k.startswith(prefix + "__")}

        self.cell.load_state_dict(
            {
                "input_proj": _section("input_proj"),
                "recurrent": _section("recurrent"),
                "energy": {"energy": raw.get("energy__energy", np.zeros(0, dtype=np.float32))},
            }
        )


# ══════════════════════════════════════════════════════════════════════════════
#  UnifiedOptimizer
# ══════════════════════════════════════════════════════════════════════════════


class UnifiedOptimizer:
    """
    Steps both standard Tensor parameters (Linear, RMSNorm, etc.) and C++-backed
    sparse layers in one call. Useful when mixing Module types in one model.
    """

    def __init__(self, model: Module, lr: float = 0.001):
        self.lr = lr
        self._tensor_params = model.parameters()  # Tensors via _iter_leaves
        self._sparse_layers = self._find_sparse(model)

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
                p.grad = None
        for layer in self._sparse_layers:
            layer.step(self.lr)
