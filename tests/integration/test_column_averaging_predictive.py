"""
tests/integration/test_column_averaging_predictive.py
────────────────────────────────────────────────────
Does column-averaging training actually learn to PREDICT the next input
via `recurrent`, or does it just settle for echoing the current input
through `in_proj` alone? Neither existing column-averaging test answers
this: test_column_averaging.py's TestColumnAveragingLossTraining trains
the loss against a free `h` parameter (no layer at all), and this
package's TestColumnAveragingEndToEnd trains against a target unrelated
to the actual input sequence. Both are silent on whether `recurrent` is
doing anything predictive. See sili_peridot/JOURNAL.md for the tuning
process and the (fragile) interaction with EnergyDynamics found here.

Method: train on sequences of increasing "does the next input depend on
more than the current one" complexity and compare `recurrent`'s true-unit
weight magnitude after training. A constant sequence needs no memory at
all; a deterministic cycle (next symbol is a fixed function of the
current one) is solvable by `in_proj` alone; an ambiguous cycle (the same
symbol has different successors at different points in the cycle) is
NOT solvable without tracking position -- only `recurrent`'s accumulated
state can carry that.
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pytest

from sili.tensor import Tensor, combine_losses
from sili.energy import column_averaging_loss, EnergyDynamics
from sili.sparse_rnn import (
    FoldedColumnLayer, state_dict_to_true_csr, _sparse_linear_layer_state_dict,
)
from sili.conversion.rnn_fold import FoldedBlockDescriptor, stack_csr_vertical


def _toy_square_descriptor(n_folds: int, hidden: int, seed: int = 0,
                           density: float = 1.0) -> FoldedBlockDescriptor:
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


# ── Pattern generators ───────────────────────────────────────────────────────

def _make_constant_sequence(T: int, symbols: int, seed: int) -> np.ndarray:
    """One-hot vector repeated every step -- next input always equals the
    current one. Solvable by an identity map; no memory needed at all."""
    rng = np.random.default_rng(seed)
    v = np.zeros(symbols, dtype=np.float32)
    v[rng.integers(0, symbols)] = 1.0
    return np.tile(v, (T, 1)).astype(np.float32)


def _make_sequence(cycle: np.ndarray, T: int, symbols: int) -> np.ndarray:
    """One-hot encode a repeating cycle of symbol indices, tiled to length T."""
    idx = np.tile(cycle, T // len(cycle) + 2)[:T]
    x = np.zeros((T, symbols), dtype=np.float32)
    x[np.arange(T), idx] = 1.0
    return x


def _make_ambiguous_cycle(period: int, symbols: int, seed: int) -> np.ndarray:
    """A repeating cycle where at least one symbol has two DIFFERENT
    successors at different points in the cycle -- predicting the next
    symbol from the current one alone is then impossible in principle;
    only tracking position-in-cycle (state) resolves it."""
    rng = np.random.default_rng(seed)
    for _ in range(500):
        cycle = rng.integers(0, symbols, size=period)
        succ = {}
        ambiguous = False
        for i in range(period):
            s, nxt = int(cycle[i]), int(cycle[(i + 1) % period])
            if s in succ and succ[s] != nxt:
                ambiguous = True
                break
            succ[s] = nxt
        if ambiguous:
            return cycle
    raise RuntimeError("could not find an ambiguous cycle -- try a longer period")


def _recurrent_true_weight_magnitude(layer: FoldedColumnLayer) -> float:
    _, _, vals = state_dict_to_true_csr(_sparse_linear_layer_state_dict(layer.recurrent))
    return float(np.mean(np.abs(vals))) if len(vals) else 0.0


# ── Training harness ─────────────────────────────────────────────────────────
#
# weight=30 amplifies column_averaging_loss's gradient reaching `recurrent`
# well above what it'd normally be (documented elsewhere as "real but tiny")
# so a clear signal shows up within a practical number of epochs for a unit
# test -- not a claim about realistic training hyperparameters.

def _train(seq: np.ndarray, n_folds: int, hidden: int, seed_desc: int,
          epochs: int, weight: float = 30.0, lr: float = 0.02,
          energy: "EnergyDynamics" = None) -> FoldedColumnLayer:
    desc  = _toy_square_descriptor(n_folds, hidden, density=0.6, seed=seed_desc)
    layer = FoldedColumnLayer.from_descriptor(desc, learning_rate=lr, num_cpus=1)
    T = len(seq)
    for _ in range(epochs):
        state = None
        for t in range(T - 1):
            x      = Tensor(seq[t])
            target = Tensor(seq[t + 1])
            raw = layer(x, state)
            if energy is not None:
                h_out, aux_loss, _ = energy.forward(raw)
            else:
                h_out, aux_loss = raw, None
            col_loss = column_averaging_loss(h_out, target, n_folds=n_folds, weight=weight)
            total = combine_losses(aux_loss, col_loss) if aux_loss is not None else col_loss
            total.backward()
            assert np.all(np.isfinite(layer.recurrent.weights_vals)), "diverged mid-training"
            state = h_out.detach()
    return layer


class TestPatternGenerators:
    """Sanity checks on the generators themselves before trusting them to
    drive the actual training tests below."""

    def test_constant_sequence_never_changes(self):
        seq = _make_constant_sequence(20, 4, seed=0)
        assert np.all(seq == seq[0])

    def test_deterministic_cycle_has_one_successor_per_symbol(self):
        cycle = np.arange(6)
        seq = _make_sequence(cycle, 30, 6)
        idx = seq.argmax(axis=1)
        succ = {}
        for i in range(len(idx) - 1):
            s, nxt = int(idx[i]), int(idx[i + 1])
            assert succ.setdefault(s, nxt) == nxt

    def test_ambiguous_cycle_actually_has_conflicting_successors(self):
        cycle = _make_ambiguous_cycle(7, 6, seed=3)
        succ = {}
        found_conflict = False
        for i in range(len(cycle)):
            s, nxt = int(cycle[i]), int(cycle[(i + 1) % len(cycle)])
            if s in succ and succ[s] != nxt:
                found_conflict = True
            succ[s] = nxt
        assert found_conflict


class TestRecurrentRelianceGrowsWithPatternComplexity:
    """Core claim: recurrent's trained weight magnitude tracks how much
    genuine temporal memory the task needs, not just "was training run."
    Fixed seeds/hyperparameters below are empirically tuned (see
    sili_peridot/JOURNAL.md) -- this is a diagnostic tool, not a claim
    that these exact numbers hold for any seed or hyperparameter choice.
    """

    def test_constant_lt_deterministic_lt_ambiguous(self):
        n_folds, hidden, T, epochs = 4, 6, 60, 100

        const_seq   = _make_constant_sequence(T, hidden, seed=1)
        simple_seq  = _make_sequence(np.arange(hidden), T, hidden)
        complex_seq = _make_sequence(_make_ambiguous_cycle(7, hidden, seed=3), T, hidden)

        mag_const   = _recurrent_true_weight_magnitude(
            _train(const_seq,   n_folds, hidden, seed_desc=5, epochs=epochs))
        mag_simple  = _recurrent_true_weight_magnitude(
            _train(simple_seq,  n_folds, hidden, seed_desc=5, epochs=epochs))
        mag_complex = _recurrent_true_weight_magnitude(
            _train(complex_seq, n_folds, hidden, seed_desc=5, epochs=epochs))

        assert mag_const < mag_simple < mag_complex, (
            f"expected recurrent reliance to grow with pattern complexity, got "
            f"constant={mag_const:.4f} simple={mag_simple:.4f} complex={mag_complex:.4f}"
        )


class TestEnergyInteraction:
    """With/without energy, and low/high energy density, to see whether the
    complexity-ordering above survives real energy competition.

    Finding (see sili_peridot/JOURNAL.md): it does NOT reliably survive at
    this toy scale within a practical epoch budget -- energy's stochastic
    top-p gating dominates which pathway gets gradient far more than the
    underlying task structure does. So these tests only assert the weaker,
    robust property (training stays finite/stable under energy gating,
    matching TestColumnAveragingEndToEnd's own established scope), not the
    same strict ordering as the no-energy case above.
    """

    def _energy(self, density: float, p: float) -> EnergyDynamics:
        return EnergyDynamics(drive=0.08, activation_cost=0.08, precision=0.02,
                              density=density, p=p, exploration=0.002, reactivity=0.02)

    @pytest.mark.parametrize("density,p", [(0.15, 0.3), (0.6, 0.9)])
    def test_stays_finite_under_energy_gating(self, density, p):
        n_folds, hidden, T, epochs = 4, 6, 60, 40
        complex_seq = _make_sequence(_make_ambiguous_cycle(7, hidden, seed=3), T, hidden)
        layer = _train(complex_seq, n_folds, hidden, seed_desc=5, epochs=epochs,
                       energy=self._energy(density, p))
        assert np.all(np.isfinite(layer.recurrent.weights_vals))
        sub = next(iter(layer._sili_layers.values()))
        assert np.all(np.isfinite(sub.weights_vals))
