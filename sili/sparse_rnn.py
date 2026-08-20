"""
sili.sparse_rnn — sparse RNN layers.

All layers are Module subclasses. C++-backed layers (DISLDOLayer, SISLDOLayer)
carry no Tensor parameters — their weights live in a real _cpu.SparseLinearLayer
and are updated INLINE during backward, not via a separate optimizer step.
parameters() returns [] for these.

Forward flow in SparseRNNCell
------------------------------
    obs   (Tensor) ──[DISLDOLayer]────────────────────────────► Tensor
    state (Tensor) ──[CSR.from_dense]──[SISLDOLayer]──────────► Tensor
                                                  sum ──[EnergyDynamics]──► h_out

    h_out.data → CSR cached inside the cell for the next step's recurrent pass.

Backward / training
--------------------
Weight VALUE updates are inline, not a separate optimizer.step(): DISLDOLayer/
SISLDOLayer.forward(x, learning_rate) store learning_rate and their _backward
closure calls SparseLinearLayer.backward_dense/backward_sparse with it
directly — gradient computation AND the weight update happen together, during
aux_loss.backward()/loss.backward() itself. This is required to keep memory
bounded (no separate accumulate-then-apply buffer held between calls) — the
old two-phase "compute gradient, then separately .step(lr)" convention this
module used to assume was actually wrong for this system, not merely
unimplemented.

    BPTT=1 (default):
        agent.train_step(obs)   # detaches state, forward(learning_rate=self.lr), aux_loss.backward()

    Multi-step BPTT:
        for obs in episode:
            action = agent(obs, learning_rate=0.0)   # state stays in graph, no update during rollout
            ...
        loss.backward()                              # inline weight updates fire here
        agent.state = agent.state.detach()

Only structural growth (synaptogenesis: build_probes + synap_step +
equalizer_step) is a genuinely separate call, since it changes which synapses
exist rather than updating a value. It's cheap and O(1)-ish by design —
SparseRNNAgent.step() calls it every online step, not on an "every N steps"
cadence (a periodic throttle would reintroduce the lag spikes this stepwise
design exists to avoid). There is no importance-decay call in this API
generation — importance already settles via the ADSP-style activity-correlation
tracking inside forward/backward, and a separate periodic decay interacting
correctly with FP4-quantized stored values would need more care than a simple
multiply (values only resolve to FP4 granularity, and large-error entries
already get squeezed out through ongoing training) — a deliberate difference
from the old design, not an oversight.
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
    Module base for layers whose weights live in a real _cpu.SparseLinearLayer.
    parameters() returns [] — nothing participates in Tensor autograd; weight
    VALUE updates happen inline inside backward_dense/backward_sparse instead
    (see module docstring).
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

    def synaptogenesis(self, k: int, importance_cutoff: float, max_row_weights: int,
                       importance_eps: float = 1e-3, max_prune_per_step: int = 8):
        """Structural growth + memory rebalancing -- the only call here
        that ISN'T inline with forward/backward, since it changes which
        synapses exist rather than updating a value. No learning_rate:
        growth isn't a value update. Meant to be called every online
        step (cheap, O(1)-ish by design) -- see module docstring for why
        an "every N steps" cadence would be wrong here.

        importance_eps is a READ-time-only "ghost" floor -- NEVER
        written to storage anywhere. It floors an EXISTING synapse's
        stored importance ONLY for synap_step's importance_cutoff
        comparison -- a synapse whose real, ongoing-training importance
        has decayed to exactly the FP4 zero code (a real, discrete
        quantization bucket many independently-decaying synapses can
        land on simultaneously -- FP4's smallest nonzero magnitude is
        0.5, so 0 is a wide, common landing bucket) isn't automatically
        "below cutoff" the instant it gets there. Does NOT affect
        removal-priority sort order or capacity-driven
        (keep > max_row_weights) pruning, and does NOT affect
        build_probes at all -- a freshly-grown synapse is stored with
        whatever its REAL probe score is (input_accum*grad_accum,
        often exactly 0 for a row with no activity yet); this floor
        just stops it being evicted purely for reading as 0 on its
        first subsequent visit, giving real backprop time to move it.

        max_prune_per_step: caps how many connections THIS row's call
        may remove at once, regardless of why they tied for lowest
        importance -- a safety ceiling (default rarely binds), not a
        throttle on ordinary capacity trimming."""
        self._c.build_probes(k)
        self._c.synap_step(importance_cutoff, max_row_weights, max_prune_per_step=max_prune_per_step,
                           importance_eps=importance_eps)
        self._c.equalizer_step()

    def magnitude_rescale_output(self, target: float, correction_rate: float,
                                 scale_invariant: bool = False) -> None:
        """Passthrough to the real C++ magnitude_rescale_output (see
        delta_csr_types.hpp's own docstring). Not every backend has
        this bound -- e.g. the fp32 DISLDOLayerV backend has no scale
        concept to rescale -- so callers sweeping this on/off across
        mixed precisions should guard with
        `hasattr(layer._c, "magnitude_rescale_output")` (the wrapped
        C++ object, NOT this Python wrapper -- this method itself
        exists on every _SparseLayerBase subclass via inheritance and
        would raise AttributeError from self._c if the backend lacks
        it, rather than silently no-op)."""
        self._c.magnitude_rescale_output(target, correction_rate, scale_invariant)

    def state_dict(self) -> dict:
        return {
            "ptrs":       np.array(self.ptrs),
            "indices":    np.array(self.indices),
            "weights":    np.array(self.weights),
            "importance": np.array(self.importance),
        }

    def load_state_dict(self, d: dict):
        # SparseLinearLayer.load_weights takes (ptrs, indices, weights) --
        # no importance array in this API generation. A loaded layer's
        # importance starts fresh and rebuilds through subsequent
        # training, rather than being restored from the saved value.
        self._c.load_weights(
            d["ptrs"]   .astype(np.int32),
            d["indices"].astype(np.int32),
            d["weights"].astype(np.float32),
        )
        # load_weights is a tight/exact-fit load, no spare per-row byte
        # headroom -- a subsequent synap_step would immediately raise
        # ("ran out of blank space") without this. Restore growth room
        # to this layer's configured per-row cap, same as a fresh
        # pre-seeded layer already has (see _preseed_random_sparse).
        max_row_weights = getattr(self, "_max_row_weights", None)
        if max_row_weights is not None:
            self._c.equalize_to_capacity(max_row_weights)


def _preseed_random_sparse(c, n_inputs: int, n_outputs: int, max_weights: int,
                           rng: Optional[np.random.Generator] = None) -> int:
    """A freshly-constructed SparseLinearLayer has zero connections and
    produces literal all-zero output until synaptogenesis grows some.
    NOT strictly required when EnergyDynamics is in the loop (as it is
    for SparseRNNCell): its forced-firing (energy accumulates via
    `drive` alone, independent of input/weights, until a neuron crosses
    the fire threshold and outputs a forced constant) already gives an
    all-zero-weight layer a real, if slower, way to bootstrap activity
    and a gradient signal to grow from -- closer to RL-style forced
    exploration than a dead end. This is purely an optimization for a
    faster/more immediate initial bootstrap (useful in tests, or
    whenever waiting several steps for forced-firing to kick in isn't
    wanted) via a standard random-sparse-init pattern -- not a
    requirement.

    Also calls equalize_to_capacity so later synap_step calls have
    per-row headroom to grow into -- synap_step raises if a row's
    pre-allocated space is already full, and load_weights' own
    allocation is a tight/exact fit with no spare room. (Not
    expand_headroom_to -- despite its docstring promising per-row
    headroom after a full equalizer_step pass, that didn't hold up
    empirically here; equalize_to_capacity, called AFTER load_weights,
    is what actually works.)

    CSR rows are INPUTS, columns are OUTPUTS (disldo_forward iterates
    `for r in range(n_inputs)`, each row's nonzero entries its output
    connections) -- see linear_disldo.hpp.

    `rng`: defaults to `np.random.default_rng()` (fresh OS entropy,
    genuinely non-reproducible by design -- NOT affected by
    `np.random.seed()`, which only controls the legacy global
    `RandomState`, not this Generator API). Pass an explicit
    `np.random.default_rng(seed)` to get reproducible wiring for tests
    -- direct request: keep true randomness as the default, only make
    it OVERRIDABLE, not seeded-by-default.
    """
    # load_weights is a tight/exact-fit encode regardless of any prior
    # allocation -- equalize_to_capacity must be called AFTER it, not
    # before, or the headroom gets compacted away immediately (verified
    # empirically).
    per_row = max(2, max_weights // max(1, n_inputs))
    if rng is None:
        rng = np.random.default_rng()
    k     = max(1, min(n_outputs, per_row // 2))  # leave half the row's headroom free to grow into
    scale = 1.0 / np.sqrt(k)
    ptrs    = np.zeros(n_inputs + 1, dtype=np.int32)
    indices = np.empty(n_inputs * k, dtype=np.int32)
    values  = np.empty(n_inputs * k, dtype=np.float32)
    for row in range(n_inputs):
        cols = np.sort(rng.choice(n_outputs, size=k, replace=False))
        indices[row * k:(row + 1) * k] = cols
        values[row * k:(row + 1) * k]  = rng.standard_normal(k).astype(np.float32) * scale
        ptrs[row + 1] = ptrs[row] + k
    if hasattr(c, "equalize_to_capacity"):
        # SparseLinearLayer (FP4) convention -- 3-arg load_weights,
        # equalize_to_capacity grows per-row headroom for later
        # synaptogenesis.
        c.load_weights(ptrs, indices, values)
        c.equalize_to_capacity(per_row)
    else:
        # DISLDOLayerV (32-bit DeltaCSRBiValues<float> fallback) --
        # identical disldo_forward/backward math, no FP4 quantization.
        # No equalize_to_capacity binding (no growth-headroom use case
        # here -- this class exists for isolating FP4 quantization
        # itself as a variable, not for synaptogenesis experiments) and
        # load_weights takes an explicit importance array (zeros: fresh
        # connections start with no accumulated importance, matching
        # the FP4 class's own default).
        importance = np.zeros_like(values)
        c.load_weights(ptrs, indices, values, importance)
    return per_row


def _seed_scale_rank(c, rank: int, n_inputs: int, n_outputs: int,
                     rng: Optional[np.random.Generator] = None,
                     scale: float = 0.05) -> None:
    """Sets scale_rank on the C++ layer and seeds components k>=1 (both
    value_scale_k and output_scale_k) with small random values. Required:
    per scale_rank's own docstring (delta_csr_types.hpp), k>=1 defaults to
    0.0 on BOTH sides -- a genuine chicken-and-egg deadlock where neither
    factor can ever get a nonzero gradient on its own unless at least one
    side starts nonzero. rank=1 is a no-op (nothing to seed, matches the
    original single-component behavior exactly).

    Loop order matters here, not just cosmetically: set_value_scale_raw_k/
    set_output_scale_raw_k lazily resize+fill new slots with 1.0 (the
    correct default for component 0), so as long as every row's k=0 slot
    gets touched by that fill BEFORE this function's own k>=1 writes land
    (true here: row-major, k ascending from 1, one index at a time --
    each row's k=0 slot is always filled-by-resize immediately before this
    row's own k=1 write), every component-0 slot ends up correctly at 1.0
    without ever being written explicitly. Do not reorder these loops
    (e.g. column-major, or k descending) without re-verifying that still
    holds.
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


def _preseed_dense(c, n_inputs: int, n_outputs: int,
                   rng: Optional[np.random.Generator] = None,
                   quantize_fn=None) -> int:
    """Fully dense counterpart to `_preseed_random_sparse` -- every
    (input, output) pair connected, loaded straight into block4 via
    `load_dense_codes` (sili__new's `block4_load_dense`, see its own
    docstring for the loading-vs-quantization design split). Added per
    direct request to test whether this project's usual random-SPARSE
    "echo network" preseed is itself a significant source of seed
    -to-seed accuracy variance in toy comparisons (reservoir-computing
    -style connectivity-draw sensitivity), independent of whatever else
    is being compared (base, bits, etc.).

    `quantize_fn`: defaults to `_cpu.fp4_quantize_array` (this class's
    VALUES_TYPE). Kept as a parameter, not hardcoded, so a caller can
    swap in a different quantization scheme (rank-1 scale fit, etc.)
    without this function needing to change -- matches
    block4_load_dense's own loading/quantization separation.

    IMPORTANT, found directly while verifying this (two real bugs, both
    fixed, in order):

    1. Init scale can NOT just be `_preseed_random_sparse`'s own
       `1/sqrt(k)` fan-in scaling carried over naively into the RAW
       quantized value. FP4 (E2M1) has a FIXED absolute zero-rounding
       floor (~0.25, independent of layer width) -- at typical toy
       widths (state_width=128, `1/sqrt(128)~=0.09`), nearly EVERY
       drawn value would quantize to code 0 ("not live"), silently
       collapsing "dense init" back down to mostly-empty regardless of
       what was intended. Confirmed empirically: raw scale=0.3 gave
       only 23/64 live codes on an 8x8 test matrix; scale=1.5-2.0 gave
       ~90%+ live.

    2. Using a FIXED raw scale (1.5) WITHOUT any compensating fan-in
       correction elsewhere is ALSO wrong, just in the opposite
       direction -- confirmed empirically on the real curriculum: every
       one of base=4/6/12/24 collapsed to pure CHANCE (mean_acc~0.094,
       std~0.003 across 3 seeds), identically regardless of base. Root
       cause: `output[c] = sum_r input[r] * weight[r,c]` -- its variance
       scales with the number of ROWS feeding column `c` (that
       column's fan-in, i.e. its live-connection count across rows),
       not the row's own width. Sparse's k=5 connections/row at scale
       `1/sqrt(5)` keep that sum's variance ~1 (properly normalized);
       dense's ~128 connections/column at raw scale 1.5 give variance
       ~128*1.5^2~=288, almost certainly saturating whatever clip/
       RMSNorm sits downstream and destroying the learning signal the
       same way no matter what `base` is -- exactly explaining why
       every dense arm gave identical numbers.

       IMPORTANT CORRECTION (per direct review): the first fix applied
       this correction via per-ROW `value_scale` scaled by `1/sqrt
       (n_outputs)` -- the WRONG axis. It only happened to produce a
       working number in testing because q/k/v/o_proj are square
       (n_inputs==n_outputs), making the wrong-axis correction
       numerically coincide with the right one there; it would have
       been wrong for lm_head (16x10, not square). Also computed from
       the ASSUMED `n_outputs`/`n_inputs`, not the REAL post
       -quantization live count -- wrong whenever problem 1's
       zero-code collisions leave a row/column short of full density
       (as they do here: confirmed ~87.5% actual density at scale=1.5,
       not 100%).

    Fix: after loading, compute the REAL per-column live count
    directly from `weight_codes` (code 0 = not live, same convention
    `block4_load_dense`/every other block4 path already uses -- no
    assumption about density), and apply a fan-in-correcting
    `output_scale` PER COLUMN via `set_output_scale_raw` as METADATA
    (documented pattern: `true_w = stored_w * value_scale[row] *
    output_scale[col]`, "set directly WITHOUT re-encoding stored
    weights") -- output_scale is the per-COLUMN half of that product,
    matching the axis the variance actually depends on. `output_scale[c]
    = 1/(raw_scale * sqrt(col_nnz[c]))` reproduces the same
    effectively-normalized weight distribution `_preseed_random_sparse`
    already produces for a k-wide row, generalized to each column's
    OWN real fan-in rather than an assumed uniform width. `value_scale`
    (per-row) is left at its default 1.0 -- this correction lives
    entirely on the column axis, matching the math above.
    """
    if rng is None:
        rng = np.random.default_rng()
    if quantize_fn is None:
        quantize_fn = _cpu.fp4_quantize_array
    scale = 1.5  # fixed, not fan-in-scaled -- keeps codes representable, see docstring point 1
    dense = (rng.standard_normal((n_inputs, n_outputs)).astype(np.float32) * scale)
    weight_codes = quantize_fn(dense.flatten())
    importance_codes = np.zeros(n_inputs * n_outputs, dtype=np.uint8)
    c.load_dense_codes(weight_codes, importance_codes)
    # Fan-in correction via output_scale metadata, from REAL per-column
    # live counts (code 0 = not live) -- see docstring point 2.
    live = weight_codes.reshape(n_inputs, n_outputs) != 0
    col_nnz = live.sum(axis=0)
    output_scale = 1.0 / (scale * np.sqrt(np.maximum(col_nnz, 1)))
    for col in range(n_outputs):
        c.set_output_scale_raw(col, float(output_scale[col]))
    return n_outputs  # every row is already at max capacity -- nothing left to grow into


def _preseed_empty(c, n_inputs: int, n_outputs: int, max_weights: int) -> int:
    """The genuinely-designed zero-weight-init: NO connections at all
    (nnz=0), not a dense grid pre-loaded with weight=0 (see
    _preseed_random_sparse's own docstring -- "a freshly-constructed
    SparseLinearLayer has zero connections... this is purely an
    optimization for a faster bootstrap, not a requirement"). Real
    synapses are meant to be created by synaptogenesis()
    (build_probes/synap_step/equalizer_step), each starting with a real
    nonzero value the first time it's grown in -- not by gradient
    -nudging a pre-existing zero. Per direct correction: pre-loading
    weight=0/importance=1 into every slot (this project's earlier
    `all_zero_init` mechanism) is a different, synthetic experimental
    arm, not this one -- it has its own failure mode (an eps-shaped
    backprop term can decay that seeded importance back toward 0) that
    doesn't apply here since nothing exists to decay.

    Only reserves per-row growth headroom via equalize_to_capacity --
    safe to call directly on a fresh, never-load_weights'd layer
    (unlike after _preseed_random_sparse's load_weights call, which
    would immediately compact any headroom back away -- see that
    function's own docstring)."""
    per_row = max(2, max_weights // max(1, n_inputs))
    c.equalize_to_capacity(per_row)
    return per_row


# ══════════════════════════════════════════════════════════════════════════════
#  DISLDOLayer — Dense Input, Sparse Linear, Dense Output
# ══════════════════════════════════════════════════════════════════════════════

class DISLDOLayer(_SparseLayerBase):
    """Dense observation → state contribution. No CSR on either side --
    dense in, dense out, via SparseLinearLayer.forward_dense/backward_dense
    (both sides dense, weight update inline in backward_dense -- see module
    docstring). No batch dimension required for online (single-sample) use."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                 num_cpus: int = 4, rng: Optional[np.random.Generator] = None,
                 dense: bool = False, scale_rank: int = 1, empty_init: bool = False):
        self._c = _cpu.SparseLinearLayer(in_features, out_features, max_weights, num_cpus)
        if empty_init:
            self._max_row_weights = _preseed_empty(self._c, in_features, out_features, max_weights)
        elif dense:
            self._max_row_weights = _preseed_dense(self._c, in_features, out_features, rng)
        else:
            self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)
        _seed_scale_rank(self._c, scale_rank, in_features, out_features, rng)

    def forward(self, x, learning_rate: float = 0.0, lr_per_row_nnz: bool = True,
                damp_by_importance: bool = True, min_decay_frac: Optional[float] = None,
                max_abs_delta: Optional[float] = None, max_ci: Optional[float] = None) -> Tensor:
        # min_decay_frac/max_abs_delta/max_ci: None (default) means "use
        # the C++ side's own tuned production defaults" -- kept as
        # Optional here rather than hardcoding the production floats a
        # second time in Python, so there is still exactly one source of
        # truth (cpu_backend.cpp's kSynapsePolicy* constants) even though
        # this wrapper now exposes them for quick per-call experiments
        # (e.g. testing max_abs_delta=huge to effectively disable the
        # clip) without a C++ rebuild.
        #
        # forward_dense/backward_dense always return [batch, cols] --
        # even for a bare 1-D [cols] input, batch is implicitly 1, but
        # the OUTPUT shape stays 2-D regardless. Squeeze that back out
        # when the input was 1-D so a single online sample round-trips
        # to 1-D, matching this class's own no-batch-dimension
        # docstring; leave genuinely 2-D (batched) input alone.
        #
        # lr_per_row_nnz default True preserves prior behavior (was
        # hardcoded) -- exists at all because it was previously
        # unreachable from this wrapper: `learning_rate` silently meant
        # "learning_rate / row_degree", not the literal rate passed in.
        # That normalization exists to keep aggregate row updates
        # comparable when synaptogenesis makes degree vary WITHIN a
        # layer -- at uniform density (no synaptogenesis, or a
        # deliberately dense/parameter-matched layer) it does nothing
        # but silently shrink the effective rate by the row's degree.
        # Pass False for a literal, degree-independent learning rate.
        #
        # forward_dense no longer takes learning_rate at all (see
        # JOURNAL.md) -- it used to run its own gradient-free ADSP-style
        # (Activity-Dependent Structural Plasticity) importance update on
        # every call, independent of whether a backward() would ever
        # follow. Real weight/importance updates now happen ONLY in
        # backward_dense, below.
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float32))
        x_np   = np.asarray(x.data, dtype=np.float32)
        was_1d = x_np.ndim == 1
        out_np = self._c.forward_dense(x_np)
        if was_1d:
            out_np = out_np.squeeze(0)
        out = Tensor(out_np, _children=(x,), _op="disldo", backend=x.backend)

        def _bwd():
            if out.grad is not None:
                dy = np.asarray(out.grad, dtype=np.float32)
                extra = {}
                if min_decay_frac is not None: extra["min_decay_frac"] = min_decay_frac
                if max_abs_delta is not None: extra["max_abs_delta"] = max_abs_delta
                if max_ci is not None: extra["max_ci"] = max_ci
                dx = self._c.backward_dense(dy, learning_rate, lr_per_row_nnz=lr_per_row_nnz,
                                             damp_by_importance=damp_by_importance, **extra)
                if was_1d:
                    dx = dx.squeeze(0)
                _acc(x, dx)

        out._backward = _bwd
        return out


class DISLDOLayerResync(DISLDOLayer):
    """Real DISLDOLayer (true C++ FP4 storage, not a fake-quantize
    simulation) with the DeferredScaleWrite fix applied -- the FP4
    counterpart of DISLDOLayer8Resync. Plain DISLDOLayer/SparseLinearLayer
    used the DEFAULT ScalePolicy/DeferredScaleWrite=false template args
    (the same stale-code path SparseLinearLayer8Resync was built to fix
    for FP8), never updated when that fix landed -- see
    sili_peridot/JOURNAL.md's TrueMultiDigitLayer entry: real FP4
    per-digit training collapsed to chance while an architecturally
    identical fp32-shadow control succeeded by a wide margin, which is
    what prompted checking whether FP4 had this same staleness bug too."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                 num_cpus: int = 4, rng: Optional[np.random.Generator] = None):
        self._c = _cpu.SparseLinearLayerResync(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)


class DISLDOLayerNoScale(DISLDOLayer):
    """Real DISLDOLayer (true C++ FP4 storage) with value_scale/
    output_scale permanently forced to 1.0 -- never trained, nothing to
    go stale, direct hardware test of the "zero trained scale" design
    (sili_peridot's fixed_digit_residual_quantize/TrueMultiDigitLayer
    work) instead of just fixing the staleness bug. See NoScalePolicy's
    docstring, delta_csr_types.hpp."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                 num_cpus: int = 4, rng: Optional[np.random.Generator] = None):
        self._c = _cpu.SparseLinearLayerNoScale(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)


class DISLDOLayerDeterministic(DISLDOLayer):
    """Real DISLDOLayer (true C++ FP4 storage), same RMSprop scale handling
    as plain DISLDOLayer, but the weight/importance store uses deterministic
    nearest-neighbour rounding (fp4_quantize) instead of stochastic dithered
    rounding (fp4_quantize_stochastic). Isolates rounding noise from any
    value_scale mechanism -- built after DISLDOLayerResync/NoScale both
    still collapsed to chance on sili_peridot's out-of-context curriculum,
    to test whether real FP4's per-step stochastic rounding (not value_scale
    staleness) is what the deterministic-rounding fp32-shadow controls
    (fixed_digit_residual_quantize, TrueMultiDigitLayer's simulate_quantize)
    were actually avoiding. See StochasticRounding's docstring,
    linear_disldo.hpp."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                 num_cpus: int = 4, rng: Optional[np.random.Generator] = None,
                 dense: bool = False, scale_rank: int = 1, empty_init: bool = False):
        self._c = _cpu.SparseLinearLayerDeterministic(in_features, out_features, max_weights, num_cpus)
        if empty_init:
            self._max_row_weights = _preseed_empty(self._c, in_features, out_features, max_weights)
        elif dense:
            self._max_row_weights = _preseed_dense(self._c, in_features, out_features, rng)
        else:
            self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)
        _seed_scale_rank(self._c, scale_rank, in_features, out_features, rng)


class DISLDOLayerResyncDeterministic(DISLDOLayer):
    """DeferredScaleWrite fix + deterministic rounding together."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                 num_cpus: int = 4, rng: Optional[np.random.Generator] = None):
        self._c = _cpu.SparseLinearLayerResyncDeterministic(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)


class DISLDOLayerNoScaleDeterministic(DISLDOLayer):
    """value_scale/output_scale forced off + deterministic rounding
    together -- the closest real-hardware match to the zero-trained-scale,
    deterministic-quantize design of fixed_digit_residual_quantize."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                 num_cpus: int = 4, rng: Optional[np.random.Generator] = None):
        self._c = _cpu.SparseLinearLayerNoScaleDeterministic(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)


# ══════════════════════════════════════════════════════════════════════════════
#  DISLDOLayer32 — same DISLDO math, DeltaCSRBiValues<float> (32-bit) instead
#  of FP4BiPacked -- isolates FP4 quantization itself as a variable, not a
#  synaptogenesis/production layer (no equalize_to_capacity, no block4
#  promotion path exercised here). See JOURNAL.md: built specifically to
#  separate "FP4's coarse quantization noise" from "the importance-damping
#  update rule" as candidate explanations for DISLDO's training instability.
# ══════════════════════════════════════════════════════════════════════════════

class DISLDOLayer32(_SparseLayerBase):
    """Same disldo_forward/disldo_backward kernels as DISLDOLayer, generic
    over VALUES_TYPE -- this instantiation uses the 32-bit float fallback
    (_cpu.DISLDOLayerV) instead of 4-bit FP4BiPacked. Same call convention
    as DISLDOLayer (forward(x, learning_rate, lr_per_row_nnz,
    damp_by_importance)).

    NOT a general-purpose production layer: no equalize_to_capacity (no
    growth-headroom use case here), no block4 promotion. Pure diagnostic
    -- same math, no quantization, to isolate what FP4's coarseness
    specifically contributes vs. the update-rule math itself."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                 num_cpus: int = 4, rng: Optional[np.random.Generator] = None):
        self._c = _cpu.DISLDOLayerV(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)

    def forward(self, x, learning_rate: float = 0.0, lr_per_row_nnz: bool = True,
                damp_by_importance: bool = True, min_decay_frac: Optional[float] = None,
                max_abs_delta: Optional[float] = None, max_ci: Optional[float] = None) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float32))
        x_np   = np.asarray(x.data, dtype=np.float32)
        was_1d = x_np.ndim == 1
        out_np = self._c.forward(x_np)
        if was_1d:
            out_np = out_np.squeeze(0)
        out = Tensor(out_np, _children=(x,), _op="disldo32", backend=x.backend)

        def _bwd():
            if out.grad is not None:
                dy = np.asarray(out.grad, dtype=np.float32)
                extra = {}
                if min_decay_frac is not None: extra["min_decay_frac"] = min_decay_frac
                if max_abs_delta is not None: extra["max_abs_delta"] = max_abs_delta
                if max_ci is not None: extra["max_ci"] = max_ci
                dx = self._c.backward(dy, learning_rate, lr_per_row_nnz=lr_per_row_nnz,
                                       damp_by_importance=damp_by_importance, **extra)
                if was_1d:
                    dx = dx.squeeze(0)
                _acc(x, dx)

        out._backward = _bwd
        return out


class DISLDOLayer8(_SparseLayerBase):
    """Same disldo_forward/disldo_backward kernels as DISLDOLayer/
    DISLDOLayer32, generic over VALUES_TYPE -- this instantiation uses
    real 8-bit storage (_cpu.SparseLinearLayer8, OCP MX E4M3 per-value
    codec, fp8quant.hpp) instead of FP4BiPacked or the 32-bit float
    fallback. Same call convention as DISLDOLayer/DISLDOLayer32
    (forward(x, learning_rate, lr_per_row_nnz, damp_by_importance)).

    Combined with the existing per-row `value_scale`/per-col
    `output_scale` (rank-1) mechanism already shared by every
    DISLDOLayer-family class, this is the concrete "8-bit + rank-1
    scale, weight AND importance both quantized" scheme validated in
    sili_peridot's toy-model quantization sweep (three task families,
    ten configs, never lost to native FP4) -- see sili_peridot's
    JOURNAL.md for the full writeup this class is built from.

    dense=True uses the block4 dense-tile SIMD path (Block4Tile8/Store8,
    task #94/#96) via the same load_dense_codes/block4_load_dense
    machinery DISLDOLayer's own dense=True already uses -- this class's
    own docstring previously said "no block4 dense-tile SIMD promotion
    yet" because the PYTHON wrapper never called it, even though the
    C++/pybind side (SparseLinearLayer8.load_dense_codes) has been
    ready since #123."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                 num_cpus: int = 4, rng: Optional[np.random.Generator] = None,
                 dense: bool = False, scale_rank: int = 1):
        self._c = _cpu.SparseLinearLayer8(in_features, out_features, max_weights, num_cpus)
        if dense:
            self._max_row_weights = _preseed_dense(self._c, in_features, out_features, rng,
                                                    quantize_fn=_cpu.fp8_quantize_array)
        else:
            self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)
        # scale_rank was never threaded here before -- confirmed missing
        # (unlike DISLDOLayer/DISLDOLayerDeterministic's own
        # _seed_scale_rank call) while wiring up a real-engine rank1/rank2
        # sweep across fp4/fp4_dual/fp8 (sili_peridot task #247); a rank2
        # FP8 arm would otherwise TypeError immediately.
        _seed_scale_rank(self._c, scale_rank, in_features, out_features, rng)

    def forward(self, x, learning_rate: float = 0.0, lr_per_row_nnz: bool = True,
                damp_by_importance: bool = True, min_decay_frac: Optional[float] = None,
                max_abs_delta: Optional[float] = None, max_ci: Optional[float] = None) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float32))
        x_np   = np.asarray(x.data, dtype=np.float32)
        was_1d = x_np.ndim == 1
        out_np = self._c.forward(x_np)
        if was_1d:
            out_np = out_np.squeeze(0)
        out = Tensor(out_np, _children=(x,), _op="disldo8", backend=x.backend)

        def _bwd():
            if out.grad is not None:
                dy = np.asarray(out.grad, dtype=np.float32)
                extra = {}
                if min_decay_frac is not None: extra["min_decay_frac"] = min_decay_frac
                if max_abs_delta is not None: extra["max_abs_delta"] = max_abs_delta
                if max_ci is not None: extra["max_ci"] = max_ci
                dx = self._c.backward(dy, learning_rate, lr_per_row_nnz=lr_per_row_nnz,
                                       damp_by_importance=damp_by_importance, **extra)
                if was_1d:
                    dx = dx.squeeze(0)
                _acc(x, dx)

        out._backward = _bwd
        return out


class DISLDOLayer8Resync(DISLDOLayer8):
    """Real DISLDOLayer8 (true C++ E4M3 storage, not a fake-quantize
    simulation) with the DeferredScaleWrite fix: touched entries'
    stored codes are written out only after value_scale/output_scale
    are both finalized for the backward() call, instead of immediately
    under the stale pre-update scale. Direct test of whether that
    staleness (not the update RULE) explains the real-vs-simulation
    out-of-context gap found in sili_peridot's toy quantization sweep
    -- see delta_csr_types.hpp's ScalePolicy docstring and
    linear_disldo.hpp's disldo_backward for the mechanism, and
    sili_peridot/JOURNAL.md's 2026-08-09 tile-recurrence entries for
    the investigation this was built to answer."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                 num_cpus: int = 4, rng: Optional[np.random.Generator] = None):
        self._c = _cpu.SparseLinearLayer8Resync(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)


class DISLDOLayer8AdaMax(DISLDOLayer8):
    """Same as DISLDOLayer8Resync (DeferredScaleWrite also on), but
    value_scale/output_scale use an AdaMax-style decayed running-max
    update instead of RMSprop -- see AdaMaxScalePolicy's own docstring,
    delta_csr_types.hpp."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                 num_cpus: int = 4, rng: Optional[np.random.Generator] = None):
        self._c = _cpu.SparseLinearLayer8AdaMax(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)


# ══════════════════════════════════════════════════════════════════════════════
#  SISLDOLayer — Sparse Input, Sparse Linear, Dense Output
# ══════════════════════════════════════════════════════════════════════════════

class SISLDOLayer(_SparseLayerBase):
    """Sparse state → state contribution. Input must be a CSR.

    Forward exploits sparse ACTIVATIONS (forward_sparse, skips inactive
    input rows). Backward exploits a top-k'd sparse GRADIENT
    (backward_sparse) -- these are independent axes on SparseLinearLayer,
    not a matched forward/backward pair (see its class comment in
    cpu_backend.cpp): dx doesn't depend on the input's own sparsity, only
    on weights and dy, so backward_sparse's required dense `x` argument is
    exactly what the C++ side already cached as `last_input` during the
    forward_sparse call above -- not an approximation, just reusing it."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                 num_cpus: int = 4, backprop_p: float = 0.03,
                 rng: Optional[np.random.Generator] = None):
        self._c         = _cpu.SparseLinearLayer(in_features, out_features, max_weights, num_cpus)
        self._max_row_weights = _preseed_random_sparse(self._c, in_features, out_features, max_weights, rng)
        self.backprop_p = backprop_p

    def forward(self, x: Tensor, learning_rate: float = 0.0) -> Tensor:
        """x.data must be a CSR. grad flows back as a dense ndarray.

        `learning_rate` kept in this wrapper's own signature (matches
        every other layer's forward() call convention throughout this
        project) but no longer forwarded to forward_sparse -- it used to
        run its own gradient-free ADSP-style (Activity-Dependent
        Structural Plasticity) importance update on every call,
        independent of whether backward() would ever follow. Real
        weight/importance updates now happen ONLY in backward_sparse."""
        csr    = x.data
        out_np = self._c.forward_sparse(
            csr.ptrs, csr.indices, csr.values, csr.rows).squeeze(0)
        out    = Tensor(out_np, _children=(x,), _op="sisldo", backend=x.backend)

        def _bwd():
            if out.grad is not None:
                dy   = np.asarray(out.grad, dtype=np.float32)[np.newaxis, :]
                k    = max(1, int(dy.shape[1] * self.backprop_p))
                dp, di, dv = _cpu.dense_to_top_k_csr(dy, k, self._c.num_cpus)
                dx = self._c.backward_sparse(
                    self._c.last_input, dp, di, dv,
                    csr.rows, learning_rate, lr_per_row_nnz=True,
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
        "ptrs":             np.array(layer.ptrs),
        "indices":          np.array(layer.indices),
        "weights":          np.array(layer.weights_vals),
        "importance":       np.array(layer.importance),  # NOT restorable -- see docstring
        "value_scale":      np.array([layer.get_value_scale(r) for r in range(n)], dtype=np.float32),
        "importance_scale": np.array([layer.get_importance_scale(r) for r in range(n)], dtype=np.float32),
    }


def _sparse_linear_layer_load_state_dict(layer, d: dict) -> None:
    """Restore weights and per-row value_scale/importance_scale onto an
    already-constructed layer of the matching shape. Does not restore
    importance (see _sparse_linear_layer_state_dict)."""
    layer.load_weights(
        np.asarray(d["ptrs"],    dtype=np.int32),
        np.asarray(d["indices"], dtype=np.int32),
        np.asarray(d["weights"], dtype=np.float32),
    )
    for r in range(layer.n_inputs):
        layer.set_value_scale_raw(r, float(d["value_scale"][r]))
        layer.set_importance_scale_raw(r, float(d["importance_scale"][r]))


def fit_rank1_scale_envelope(row_idx, col_idx, abs_vals, n_rows: int, n_cols: int,
                             n_iters: int = 6):
    """
    Alternating max-fit producing a rank-1 (outer-product) envelope:
    row_scale[r] * col_scale[c] >= |M[r, c]| for every entry, using
    O(rows+cols) parameters instead of O(rows*cols). Each update
    recomputes the exact bound needed against the other side's current
    estimate (Sinkhorn-style, but max- rather than sum-based).

    Takes M's nonzero entries as COO-style triplets (row_idx, col_idx,
    abs_vals), not a dense matrix -- a folded/stacked layer's matrix is
    too large to densify safely (e.g. mlp.gate_proj folds to
    [110592, 1536]; converting that to dense once already risked OOM on
    a real conversion run). Entries not listed are implicitly 0 and
    never bind the max, so this needs no dense intermediate at all.

    Returns (row_scale, col_scale), float32 -- deliberately not float64
    (a max/divide fit has no accumulating-sum precision need, and
    float64 here would double the O(nnz) working set for no benefit).
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
                        bytes_per_row: int = 0,
                        value_scale_mode: str = "per_row",
                        rank1_iters: int = 6,
                        compact_after_build: bool = True) -> "FoldedLayer":
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
          value_scale_mode -- "per_row" (default, exact prior behavior): one
                             value_scale per input row, output_scale left at
                             its default 1.0. "rank1": also fits a
                             per-output-column scale (fit_rank1_scale_envelope)
                             via set_output_scale_raw, gradient-trainable
                             like value_scale. Prefer "rank1" for real
                             pretrained-weight conversion.
          rank1_iters      -- alternating-fit iterations for "rank1" mode
                             (ignored otherwise); see fit_rank1_scale_envelope.
          compact_after_build -- strip equalize_to_capacity's per-row growth
                             headroom right after loading (default True).
                             Measured on a real checkpoint: dropped 7 suffixes'
                             combined RSS from ~1.9x to ~1.08x the 2-bytes/param
                             theoretical minimum. Call expand_headroom_to() on
                             the built layer before synaptogenesis needs room
                             to grow again.
        """
        import numpy as np
        import torch as _torch   # local import: conversion step only -- sili does
        # the compute. Do not use torch in forward/backward paths.
        import warnings; warnings.filterwarnings("ignore")
        _FP4_MAX = 6.0
        if value_scale_mode not in ("per_row", "rank1"):
            raise ValueError(
                f"value_scale_mode must be 'per_row' or 'rank1', got {value_scale_mode!r}")

        layers = {}
        for suffix, csr in descriptor.stacked_weights.items():
            # csr.t() is a metadata-only relabelling into CSC (no densify,
            # no nnz-proportional copy) -- but CSC's own ccol_indices/
            # row_indices are grouped by COLUMN, not the per-ROW grouping
            # load_weights needs. .to_sparse_csr() does the real (but
            # nnz-proportional, not densify-proportional) reorganization
            # into row-major order. NEVER call .to_dense() on a
            # stacked/folded layer's matrix here -- it can be too large to
            # safely materialize (e.g. mlp.gate_proj folds to
            # [110592, 1536]; densifying that alone risked OOM converting
            # a real checkpoint).
            csr_t = csr.t().to_sparse_csr()
            n_in  = int(csr_t.shape[0])
            n_out = int(csr_t.shape[1])
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

            if value_scale_mode == "rank1":
                # One scale per input row AND one per output column,
                # fit directly from the nonzero triplets -- no dense
                # intermediate (see fit_rank1_scale_envelope).
                row_of_nnz = np.repeat(np.arange(n_in, dtype=np.int64), np.diff(ptrs))
                row_env, col_env = fit_rank1_scale_envelope(
                    row_of_nnz, idx.astype(np.int64), np.abs(vals), n_in, n_out, n_iters=rank1_iters)
                row_scales = (row_env / _FP4_MAX).astype(np.float32)
                col_scales = col_env.astype(np.float32)
                combined = row_scales[row_of_nnz] * col_scales[idx]
                nonzero_combined = combined > 0
                vals[nonzero_combined] /= combined[nonzero_combined]
            else:
                # Per-row: map each row's max-abs to FP4_MAX for full
                # quantizer resolution.
                row_scales = np.ones(n_in, dtype=np.float32)
                col_scales = np.ones(n_out, dtype=np.float32)   # left at default (1.0) in this mode
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
            if compact_after_build:
                layer.compact()

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
        this layer automatically. forward_dense(x) [this method] is a pure
        computation (no learning_rate, no side effects -- see JOURNAL.md:
        it used to also update IMPORTANCE via activity correlation on every
        call, independent of whether backward() would ever follow, a real
        footgun now removed). backward_dense(dy, lr) [_backward, called by
        loss.backward()] is the only place weight values AND importance
        ever change, via the task gradient -- that's what enables the
        network to learn tasks.
        """
        x_np = np.asarray(x.data, dtype=np.float32)
        squeezed = x_np.ndim == 1
        if squeezed:
            x_np = x_np[np.newaxis, :]
        batch   = x_np.shape[0]
        out_dim = next(iter(self._out_dims.values()))
        lr      = self.lr  # used by the backward closure below, not forward

        # Single call per suffix -- the full stacked matrix is one layer.
        raw_parts = [layer.forward_dense(x_np)
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
    and pairs it with a `recurrent` layer -- the same input_proj+recurrent
    split SparseRNNCell uses, built on SparseLinearLayer instead of the
    currently-broken DISLDOLayer/SISLDOLayer (see TODO.md).

    in_proj(x) -- the real pretrained per-fold-step matrices (stacked),
    plus a build_input_skip_preseed()-unioned band of zero-valued
    trainable skip connections from input to every fold-depth column.

    recurrent(state) -- build_fold_skip_layer's from-scratch banded
    matrix mapping this layer's own [n_folds*out_dim] output space back to
    itself (fold step i -> i+1 and nearby columns), carrying one step's
    output into the next's input. No pretrained content.

    forward(x, state) = in_proj(x) + recurrent(state), returned as the new
    state to feed back in next call (mirrors SparseRNNCell's convention).
    state defaults to zero when not given.

    Feed forward()'s output to sili.energy.column_averaging_loss, after
    whatever EnergyDynamics gating the model applies (that function must
    run on the energy-gated state, not this layer's raw output).
    """

    @classmethod
    def from_descriptor(cls, descriptor, learning_rate: float = 0.01,
                        num_cpus: int = 4, max_row_weights: int = 0,
                        bytes_per_row: int = 0,
                        recurrent_bandwidth: int = None,
                        existing_recurrent=None,
                        existing_recurrent_prefer: str = "b",
                        input_skip_bandwidth: int = None) -> "FoldedColumnLayer":
        """
        Like FoldedLayer.from_descriptor, plus builds `recurrent` sized to
        this layer's [n_folds*out_dim] output space, and unconditionally
        unions a zero-valued input->column skip pre-seed onto every
        suffix's in_proj weights (see build_input_skip_preseed).

        recurrent_bandwidth: forwarded to build_fold_skip_layer as
        `bandwidth` (None -> that function's own default).

        existing_recurrent / existing_recurrent_prefer: forwarded to
        build_fold_skip_layer as `existing`/`existing_prefer` -- pass a
        previously-saved recurrent CSR (e.g.
        state_dict_to_true_csr(prior_layer.state_dict()["recurrent"])) to
        preserve trained skip-connection weights across re-runs. None
        (default) is the "converting a dense LLM" case.

        input_skip_bandwidth: forwarded to build_input_skip_preseed as
        `bandwidth` per suffix. No `existing` param here -- each suffix's
        real weights ARE the base the fresh pre-seed unions onto, real
        values always winning (see _rebuild_layer_with_preseed).
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

        out_dim = obj.column_width
        for suffix, layer in list(obj._sili_layers.items()):
            preseed_ptrs, preseed_idx, _ = build_input_skip_preseed(
                layer.n_inputs, obj._n_folds, out_dim, bandwidth=input_skip_bandwidth)
            obj._sili_layers[suffix] = _rebuild_layer_with_preseed(
                layer, preseed_ptrs, preseed_idx,
                num_cpus=num_cpus, expected_lr=learning_rate,
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
        lr = self.lr  # used by the backward closure below, not forward

        raw_parts = [layer.forward_dense(x_np)
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
        idx[pos:pos + n] = np.arange(lo[r], hi[r] + 1, dtype=np.int32)
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
        idx[pos:pos + n] = np.arange(lo[r], hi[r] + 1, dtype=np.int32)
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
              n_rows: int, prefer: str = "a", num_cpus: int = 4):
    """
    Merge two CSRs of the SAME shape into one holding the union of their
    nonzero positions. `vals_a`/`vals_b` must already be in true units
    (not raw FP4 levels -- mixing raw units from differently-scaled
    sources would be wrong); rescaling happens after the union, once a
    single per-row scale for the merged row is chosen.

    Where both inputs have an entry at (row, col), `prefer` decides the
    result: 'a' (default) keeps A, 'b' keeps B, 'sum' adds them.

    Construction/loading time only -- never used in the forward/backward
    path. See build_fold_skip_layer's `existing` for the concrete case.

    OpenMP-parallel (_cpu.csr_union, see csr.hpp) -- each row is an
    independent two-pointer merge, so this scales to real model-sized
    weight matrices instead of a per-row Python loop.
    """
    assert prefer in ("a", "b", "sum")
    return _cpu.csr_union(
        np.asarray(ptrs_a, dtype=np.int32), np.asarray(idx_a, dtype=np.int32),
        np.asarray(vals_a, dtype=np.float32),
        np.asarray(ptrs_b, dtype=np.int32), np.asarray(idx_b, dtype=np.int32),
        np.asarray(vals_b, dtype=np.float32),
        int(n_rows), prefer, num_cpus,
    )


def build_fold_skip_layer(n_folds: int, out_dim: int, num_cpus: int = 4,
                          bandwidth: int = None,
                          headroom_fraction: float = 0.5,
                          expected_lr: float = 0.01,
                          existing=None,
                          existing_prefer: str = "b") -> "_cpu.SparseLinearLayer":
    """
    Sparse layer mapping a FoldedColumnLayer's own [n_folds*out_dim]
    output space back to itself: skip connections between virtual
    (fold-depth) layers, pre-seeded as a zero-valued banded pattern (see
    RNNFoldedBlock.forward in conversion/rnn_fold.py for the true fold
    recurrence this approximates).

    bandwidth: connect flat positions r, c whenever abs(r - c) < bandwidth.
    Default (None) uses out_dim, so one hop reaches the adjacent fold
    step at any nearby column.

    expected_lr: sets the fallback per-row value_scale (expected_lr /
    FP4_MAX) for rows whose final values are still all-zero, so gradient
    updates of that magnitude don't round back to zero under FP4. Rows
    with real nonzero values (from `existing`) get max_abs/FP4_MAX
    instead.

    existing: optional (ptrs, idx, vals) CSR, same shape, vals in TRUE
    units (see csr_union) -- unioned with the fresh band before
    construction. Pass a previously-trained recurrent CSR to preserve it
    across re-runs; None (default) is the "converting a dense LLM" case.
    existing_prefer: csr_union's prefer arg for overlapping positions --
    default 'b' (existing's trained value wins over the fresh zero).
    """
    assert n_folds >= 1 and out_dim >= 1
    bw = out_dim if bandwidth is None else bandwidth
    assert bw >= 1
    total = n_folds * out_dim

    ptrs, idx, vals = _build_banded_csr(total, bw)

    if existing is not None:
        ex_ptrs, ex_idx, ex_vals = existing
        ptrs, idx, vals = csr_union(ptrs, idx, vals, ex_ptrs, ex_idx, ex_vals,
                                    total, prefer=existing_prefer, num_cpus=num_cpus)

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


def build_input_skip_preseed(n_in: int, n_folds: int, out_dim: int, bandwidth: int = None):
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


def _rebuild_layer_with_preseed(layer, preseed_ptrs, preseed_idx,
                                num_cpus: int, expected_lr: float,
                                headroom_fraction: float = 0.5):
    """
    Rebuild a SparseLinearLayer, unioning a zero-valued pre-seed CSR's
    structural positions onto the layer's real (already-trained/pretrained)
    weights. Real values always win at any overlap (prefer="a") -- the
    pre-seed only adds new zero-valued positions, never clobbers real data.
    Budget/scale handling mirrors build_fold_skip_layer (nnz-based
    headroom, per-row FP4 rescaling recomputed on the merged row).
    """
    n_in, n_out = layer.n_inputs, layer.n_outputs
    real_ptrs, real_idx, real_vals = state_dict_to_true_csr(
        _sparse_linear_layer_state_dict(layer))
    preseed_vals = np.zeros(len(preseed_idx), dtype=np.float32)
    ptrs, idx, vals = csr_union(real_ptrs, real_idx, real_vals,
                                preseed_ptrs, preseed_idx, preseed_vals,
                                n_in, prefer="a", num_cpus=num_cpus)

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
                 num_cpus: int = 4, percent_active: float = 0.03,
                 dynamic_density_from_branching_ratio: bool = False,
                 branching_tracker: str = "ema",
                 branching_window: int = 200,
                 branching_ema_alpha: float = 0.05):
        assert branching_tracker in ("window", "ema"), \
            f"branching_tracker must be 'window' or 'ema', got {branching_tracker!r}"
        r = percent_active / 0.02
        self.input_proj = DISLDOLayer(n_inputs,   state_size, max_weights, num_cpus)
        self.recurrent  = SISLDOLayer(state_size, state_size, max_weights, num_cpus,
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
        # activation_cost=0.08*r grows unbounded with percent_active (r
        # scales linearly with it) -- fine for the small percent_active
        # this formula was tuned around, but a small state legitimately
        # wants a much higher percent_active to get more than 0-1 active
        # neurons (e.g. state_size=12), which pushes activation_cost past
        # EnergyDynamics's own asserted [0.01, 0.5] range. Clamp rather
        # than let construction fail for that (valid) regime.
        activation_cost = min(0.5, max(0.01, 0.08 * r))
        self.energy     = EnergyDynamics(
            drive          = 0.08*percent_active * r,
            activation_cost= activation_cost,
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

    def forward(self, obs: Tensor, state: Tensor, learning_rate: float = 0.0) -> Tuple[Tensor, Tensor, float]:
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
        recurrent_out = self.recurrent(state, learning_rate)
        recurrent_activity = float(np.sum(
            np.abs(np.asarray(recurrent_out.data, dtype=np.float32))
            > self.energy.activation_threshold
        ))
        self.branching_recurrent.update(recurrent_activity)

        h = self.input_proj(obs, learning_rate) + recurrent_out

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

    def synaptogenesis(self, k: int, importance_cutoff: float, max_weights: int):
        """Structural growth + memory rebalancing for both sub-layers --
        see _SparseLayerBase.synaptogenesis. No lr/step/decay here: weight
        VALUE updates already happened inline during forward()'s
        backward() closures (see module docstring), and there's no
        importance-decay call in this API generation.

        max_weights is the TOTAL per-layer budget (same units as what
        was passed to this cell's own constructor) -- converted to a
        PER-ROW cap for each sub-layer here (synap_step's actual unit),
        using each sub-layer's own in_features, since input_proj and
        recurrent generally have different row counts and a single
        scalar can't be a correct per-row cap for both at once."""
        input_proj_cap = max(1, max_weights // self.input_proj.in_features)
        recurrent_cap  = max(1, max_weights // self.recurrent.in_features)
        self.input_proj.synaptogenesis(k, importance_cutoff, input_proj_cap)
        self.recurrent .synaptogenesis(k, importance_cutoff, recurrent_cap)

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
                 num_cpus: int = 4, percent_active: float = 0.03,
                 lr: float = 1e-3, importance_cutoff: float = 0.01,
                 synaptogenesis_k: int = 4):
        # build_probes(k) generates min(k, n_in) * min(k, n_out) candidate
        # pairs (top-k input rows outer-producted against top-k output
        # columns) -- O(k^2), not O(k). Measured directly: on a 1000x1000
        # layer, k=64 can grow nnz from ~0 to 1000 (full density) in a
        # SINGLE synap_step call, vs. k=4 growing nnz by only ~16 -- since
        # this runs every online step (see class/module docstring), a
        # large k saturates connectivity almost immediately rather than
        # growing gradually. k=4 (default) or smaller is far more sane
        # for continuous per-step growth; only raise this if you've
        # checked the resulting per-step nnz growth against your actual
        # max_weights budget and step rate.
        assert n_actions <= state_size

        self.cell = SparseRNNCell(n_inputs, state_size, max_weights, num_cpus, percent_active)

        self.state = Tensor(np.zeros(state_size, dtype=np.float32))

        self.n_inputs    = n_inputs
        self.n_actions   = n_actions
        self.state_size  = state_size
        self.max_weights = max_weights

        self.lr                = lr
        self.importance_cutoff = importance_cutoff
        self.synaptogenesis_k  = synaptogenesis_k

        self._step_count = 0
        self.aux_loss:   Optional[Tensor] = None
        self._actual_p:  float = 0.0

    def parameters(self) -> list:
        return []

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, obs: Tensor) -> int:
        """Run one step. State stays in the autograd graph (use for multi-step
        BPTT). Threads self.lr into the cell so weight VALUE updates fire
        inline whenever backward() eventually runs (see module docstring) --
        forward() itself never updates anything, only backward() does."""
        h_out, aux_loss, actual_p = self.cell(obs, self.state, self.lr)
        self.state      = h_out
        self.aux_loss   = aux_loss
        self._actual_p  = actual_p
        return int(np.argmax(np.asarray(self.state.data, dtype=np.float32)[:self.n_actions]))

    def train_step(self, obs: Tensor) -> int:
        """
        BPTT=1 convenience wrapper. Detaches state before forward so gradients
        don't flow across steps, then runs aux_loss.backward() (fires the
        inline weight-value updates) and step() (structural growth only --
        weight values are already updated by that point, see module
        docstring).

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
        """Structural growth + memory rebalancing only -- weight VALUE
        updates already happened inline during aux_loss.backward()/
        loss.backward() (see module docstring). Called every step, not
        throttled by an "every N steps" cadence: synaptogenesis/
        equalizer_step are cheap and purpose-built for continuous
        per-step operation -- throttling them would reintroduce the lag
        spikes they exist to avoid."""
        self.cell.synaptogenesis(self.synaptogenesis_k, self.importance_cutoff, self.max_weights)
        self._step_count += 1

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
