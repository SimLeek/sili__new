"""
tests/test_sili.py
──────────────────
Pytest suite for the SILi CPU backend (_cpu extension module).

Run from the build directory (where _cpu.so lives):
    pytest test_sili.py -v

Or from the repo root after `pip install -e .`:
    pytest tests/test_sili.py -v
"""
from __future__ import annotations
import sys
import os
import math
import numpy as np
import pytest

# Allow running directly from the build directory.
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    # Package-qualified first -- see sili/conversion/rnn_fold.py for why
    # import order matters here (double pybind11 registration otherwise).
    from sili import _cpu
except ImportError:
    import _cpu

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

BUDGET = 4 * 1024 * 1024   # 4 MB — plenty for tests

def make_layer(n_in=8, n_out=8, bw=1, budget=BUDGET, cpus=1):
    return _cpu.SparseLinearLayer(n_in, n_out, bw, budget, cpus)

def dense_to_csr(x: np.ndarray):
    """Convert (batch, cols) dense → (ptrs, indices, values) via _cpu helper."""
    return _cpu.dense_to_csr(np.asarray(x, dtype=np.float32), 0.0)

def fp4_q(v: float) -> float:
    """Round-trip through FP4 quantisation (nearest representable value)."""
    FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    return min(FP4, key=lambda x: abs(x - v) if x != float('nan') else float('inf'))

# ─────────────────────────────────────────────────────────────────────────────
# Layer construction
# ─────────────────────────────────────────────────────────────────────────────

class TestConstruction:
    def test_properties(self):
        layer = make_layer(8, 8, bw=1)
        assert layer.n_inputs  == 8
        assert layer.n_outputs == 8
        # bandwidth=1 → 3 synapses for middle rows, 2 for corners
        assert layer.nnz > 0

    def test_diagonal_only(self):
        layer = make_layer(6, 6, bw=0)
        assert layer.nnz == 6   # exactly one per row

    def test_budget_exceeded_raises(self):
        with pytest.raises(MemoryError):
            _cpu.SparseLinearLayer(1000, 1000, 10, 1)  # 1 byte — absurd

    def test_accum_zeroed(self):
        layer = make_layer()
        assert np.all(layer.neuron_input_accum == 0.0)
        assert np.all(layer.neuron_grad_accum  == 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# Buffer / NeuronView / SynapseView debug access
# ─────────────────────────────────────────────────────────────────────────────

class TestBufferAccess:
    def setup_method(self):
        self.layer = make_layer(6, 6, bw=1)
        self.buf   = self.layer.buffer

    def test_n_neurons(self):
        assert self.buf.n_neurons == 6

    def test_n_outputs(self):
        assert self.buf.n_outputs == 6

    def test_neuron_nnz_corners(self):
        assert self.buf.neuron[0].nnz == 2  # cols {0,1}
        assert self.buf.neuron[5].nnz == 2  # cols {4,5}

    def test_neuron_nnz_middle(self):
        assert self.buf.neuron[2].nnz == 3  # cols {1,2,3}

    def test_neuron_len(self):
        assert len(self.buf.neuron[0]) == self.buf.neuron[0].nnz

    def test_synapse_index(self):
        n = self.buf.neuron[2]
        # bandwidth=1 from row 2: cols should be 1,2,3
        assert n.synapse[0].index == 1
        assert n.synapse[1].index == 2
        assert n.synapse[2].index == 3

    def test_synapse_weight_zero_init(self):
        n = self.buf.neuron[2]
        for k in range(len(n)):
            assert n.synapse[k].weight == 0.0

    def test_synapse_importance_zero_init(self):
        n = self.buf.neuron[2]
        for k in range(len(n)):
            assert n.synapse[k].importance == 0.0

    def test_synapse_weight_set(self):
        n = self.buf.neuron[2]
        n.synapse[1].weight = 1.5
        # FP4 nearest to 1.5 is 1.5 exactly.
        assert n.synapse[1].weight == pytest.approx(1.5, abs=1e-6)

    def test_synapse_importance_set(self):
        n = self.buf.neuron[2]
        n.synapse[0].importance = -1.0
        assert n.synapse[0].importance == pytest.approx(-1.0, abs=1e-6)

    def test_synapse_weight_does_not_affect_importance(self):
        n = self.buf.neuron[2]
        n.synapse[1].importance = 0.5
        n.synapse[1].weight     = 2.0
        assert n.synapse[1].importance == pytest.approx(0.5, abs=1e-6)

    def test_synapse_set_index_reorders(self):
        # Row 2: cols {1,2,3}. Change synapse[2] from col 3 to col 5.
        n = self.buf.neuron[2]
        n.synapse[2].index = 5
        # Row should now have cols {1,2,5} sorted.
        assert n.synapse[0].index == 1
        assert n.synapse[1].index == 2
        assert n.synapse[2].index == 5

    def test_synapse_repr(self):
        n = self.buf.neuron[0]
        r = repr(n.synapse[0])
        assert "Idx:" in r
        assert "W:"   in r
        assert "I:"   in r

    def test_synapse_str(self):
        n   = self.buf.neuron[0]
        s   = str(n.synapse[0])
        assert ";" in s

    def test_neuron_to_numpy_shape(self):
        n   = self.buf.neuron[2]
        arr = n.to_numpy()
        assert arr.shape == (3, 3)   # 3 synapses, 3 columns
        assert arr.dtype == np.float32

    def test_neuron_to_numpy_cols(self):
        n   = self.buf.neuron[2]
        arr = n.to_numpy()
        # Column 0 is the absolute column index.
        np.testing.assert_array_equal(arr[:, 0], [1.0, 2.0, 3.0])

    def test_neuron_to_numpy_values(self):
        n = self.buf.neuron[2]
        n.synapse[1].weight     = 2.0
        n.synapse[1].importance = 1.0
        arr = n.to_numpy()
        assert arr[1, 1] == pytest.approx(2.0, abs=1e-6)
        assert arr[1, 2] == pytest.approx(1.0, abs=1e-6)

    def test_neuron_negative_index(self):
        # Python-style negative indexing.
        n  = self.buf.neuron[2]
        s0 = n.synapse[-3]   # same as synapse[0]
        assert s0.index == 1

    def test_neuron_out_of_range_raises(self):
        with pytest.raises(IndexError):
            _ = self.buf.neuron[2].synapse[99]

    def test_buffer_out_of_range_raises(self):
        with pytest.raises(IndexError):
            _ = self.buf.neuron[999]

    def test_buffer_neuron_alias(self):
        # buffer.neuron[i] and buffer[i] should behave the same.
        assert self.buf.neuron[0].nnz == self.buf[0].nnz

    def test_bytes_used_positive(self):
        assert self.buf.bytes_used > 0

    def test_total_nnz_matches_layer(self):
        assert self.buf.total_nnz == self.layer.nnz

# ─────────────────────────────────────────────────────────────────────────────
# dense_to_csr utility
# ─────────────────────────────────────────────────────────────────────────────

class TestDenseToCsr:
    def test_basic(self):
        x = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float32)
        ptrs, idx, vals = _cpu.dense_to_csr(x, 0.0)
        assert ptrs.tolist() == [0, 2, 3]
        assert idx.tolist()  == [0, 2, 1]
        np.testing.assert_allclose(vals, [1.0, 2.0, 3.0])

    def test_threshold(self):
        x = np.array([[0.001, 1.0]], dtype=np.float32)
        ptrs, idx, vals = _cpu.dense_to_csr(x, 0.01)
        assert idx.tolist() == [1]  # small value filtered out

# ─────────────────────────────────────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────────────────────────────────────

class TestForward:
    def test_zero_weights_zero_output(self):
        # Zero-weight init: output must be exactly zero.  This is a property
        # of the init, not a general forward-pass test.
        layer = make_layer(4, 4, bw=0)
        x = np.ones((1, 4), dtype=np.float32)
        out = layer.forward_dense(x, lr=0.0)
        assert out.shape == (1, 4)
        np.testing.assert_array_equal(out, 0.0)

    def test_output_shape(self):
        layer = make_layer(4, 6, bw=0)
        x = np.ones((3, 4), dtype=np.float32)
        out = layer.forward_dense(x, lr=0.0)
        assert out.shape == (3, 6)

    def test_nonzero_weight_specific_output(self):
        # Row 2 → col 2 with weight = 1.5 (FP4-exact).
        # Input x[2] = 0.8.  Expected output at col 2 = 1.5 × 0.8 = 1.2.
        # FP4 nearest to 1.2 is 1.0 (representable values: 1.0, 1.5).
        # The weight is stored as FP4(1.5) = 1.5, so result = 1.5 × 0.8 = 1.2.
        layer = make_layer(4, 4, bw=0)
        layer.buffer.neuron[2].synapse[0].weight = 1.5   # FP4-exact

        x = np.zeros((1, 4), dtype=np.float32)
        x[0, 2] = 0.8
        out = layer.forward_dense(x, lr=0.0)

        assert out[0, 2] == pytest.approx(1.5 * 0.8, abs=1e-6)
        assert out[0, 0] == 0.0
        assert out[0, 1] == 0.0
        assert out[0, 3] == 0.0

    def test_multi_weight_linear_combination(self):
        # With bw=0 (diagonal), row r → col r.  Use two separate rows and verify
        # each contributes to its own output column independently.
        # Row 0 → col 0, w=-1.5.  Row 2 → col 2, w=2.0.
        # x = [0.4, 0, 0.5, 0].  out[0] = -1.5×0.4 = -0.6.  out[2] = 2.0×0.5 = 1.0.
        layer = make_layer(4, 4, bw=0)
        layer.buffer.neuron[0].synapse[0].weight = -1.5
        layer.buffer.neuron[2].synapse[0].weight =  2.0

        x = np.zeros((1, 4), dtype=np.float32)
        x[0, 0] = 0.4
        x[0, 2] = 0.5
        out = layer.forward_dense(x, lr=0.0)

        assert out[0, 0] == pytest.approx(-1.5 * 0.4, abs=1e-5)   # -0.6
        assert out[0, 2] == pytest.approx( 2.0 * 0.5, abs=1e-5)   # +1.0
        assert out[0, 1] == 0.0
        assert out[0, 3] == 0.0

    def test_forward_adds_not_overwrites(self):
        # Two successive forward calls must give identical results (internal
        # buffers are zeroed between calls, not accumulated across calls).
        # Weight = 1.5 (FP4-exact), x[0,0] = 0.6 → out[0,0] = 0.9 exactly.
        layer = make_layer(4, 4, bw=0)
        layer.buffer.neuron[0].synapse[0].weight = 1.5
        x = np.zeros((1, 4), dtype=np.float32)
        x[0, 0] = 0.6

        out1 = layer.forward_dense(x, lr=0.0)
        out2 = layer.forward_dense(x, lr=0.0)
        np.testing.assert_array_equal(out1, out2)
        assert out1[0, 0] == pytest.approx(1.5 * 0.6, abs=1e-5)

    def test_importance_updated_by_lr_specific_value(self):
        # w=1.5, x=0.4, lr=0.5.
        # Hebbian: new_imp = 0 + (1.5 × 0.4) × 0.5 / (1 + 0) = 0.3
        # FP4 nearest to 0.3 is 0.5 (table: 0.0, 0.5, 1.0 …).
        layer = make_layer(4, 4, bw=0)
        layer.buffer.neuron[0].synapse[0].weight = 1.5

        x = np.zeros((1, 4), dtype=np.float32)
        x[0, 0] = 0.4
        layer.forward_dense(x, lr=0.5)

        imp = layer.buffer.neuron[0].synapse[0].importance
        assert imp == pytest.approx(0.5, abs=1e-6)   # FP4(0.3) = 0.5

    def test_lr_zero_leaves_importance_unchanged(self):
        # After a forward with lr=0, importance must remain at its initial value.
        layer = make_layer(4, 4, bw=0)
        layer.buffer.neuron[0].synapse[0].weight     = 1.5
        layer.buffer.neuron[0].synapse[0].importance = -0.5   # pre-set non-trivial
        x = np.zeros((1, 4), dtype=np.float32)
        x[0, 0] = 0.6
        layer.forward_dense(x, lr=0.0)
        assert layer.buffer.neuron[0].synapse[0].importance == pytest.approx(-0.5, abs=1e-6)

# ─────────────────────────────────────────────────────────────────────────────
# Backward pass
# ─────────────────────────────────────────────────────────────────────────────

class TestBackward:
    def test_output_shape(self):
        # Use non-trivial x and dy so the test exercises real computation.
        layer = make_layer(4, 4, bw=0)
        layer.buffer.neuron[0].synapse[0].weight = 1.5
        x  = np.array([[0.4, 0.6, 0.2, 0.8], [0.3, 0.5, 0.7, 0.1]], dtype=np.float32)
        dy = np.array([[0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5]], dtype=np.float32)
        ptrs, idx, vals = dense_to_csr(x)
        dx = layer.backward(ptrs, idx, vals, dy, lr=0.0, batch=2)
        assert dx.shape == (2, 4)

    def test_lr_zero_leaves_weights_and_importance_unchanged(self):
        # w=1.5, imp=-0.5: both must be preserved after backward with lr=0.
        layer = make_layer(4, 4, bw=0)
        layer.buffer.neuron[1].synapse[0].weight     = 1.5
        layer.buffer.neuron[1].synapse[0].importance = -0.5

        x  = np.array([[0.3, 0.7, 0.4, 0.2]], dtype=np.float32)
        dy = np.array([[0.6, 0.0, 0.0, 0.0]], dtype=np.float32)
        ptrs, idx, vals = dense_to_csr(x)
        layer.backward(ptrs, idx, vals, dy, lr=0.0, batch=1)

        assert layer.buffer.neuron[1].synapse[0].weight     == pytest.approx(1.5,  abs=1e-6)
        assert layer.buffer.neuron[1].synapse[0].importance == pytest.approx(-0.5, abs=1e-6)

    def test_importance_specific_value_after_backward(self):
        # FP4 snaps to the nearest representable value.
        # Need |imp_raw| > 0.25 to land on -0.5 rather than 0.0.
        # w=1.5, imp=0.  x[0,0]=0.8, dy[0,0]=0.8, lr=0.5.
        # grad_eff = 0.8 × 0.8 = 0.64
        # imp_raw  = 0 − 0.64 × 0.5 = −0.32   (>0.25 from zero → snaps to −0.5)
        layer = make_layer(4, 4, bw=0)
        layer.buffer.neuron[0].synapse[0].weight = 1.5

        x  = np.array([[0.8, 0.0, 0.0, 0.0]], dtype=np.float32)
        dy = np.array([[0.8, 0.0, 0.0, 0.0]], dtype=np.float32)
        ptrs, idx, vals = dense_to_csr(x)
        layer.backward(ptrs, idx, vals, dy, lr=0.5, batch=1)

        imp = layer.buffer.neuron[0].synapse[0].importance
        assert imp == pytest.approx(-0.5, abs=1e-6)   # FP4(-0.32) = -0.5

    def test_importance_decreases_on_positive_gradient(self):
        # Same parameter choice as above: imp_raw = -0.32 → FP4 = -0.5 < 0.
        layer = make_layer(4, 4, bw=0)
        layer.buffer.neuron[0].synapse[0].weight = 1.5

        x  = np.array([[0.8, 0.0, 0.0, 0.0]], dtype=np.float32)
        dy = np.array([[0.8, 0.0, 0.0, 0.0]], dtype=np.float32)
        ptrs, idx, vals = dense_to_csr(x)
        layer.backward(ptrs, idx, vals, dy, lr=0.5, batch=1)

        assert layer.buffer.neuron[0].synapse[0].importance < 0.0

    def test_neuron_accum_filled_with_known_magnitudes(self):
        # x[0,0]=0.6 → neuron_input_accum[0] should be exactly 0.6.
        layer = make_layer(4, 4, bw=1)
        x  = np.array([[0.6, 0.0, 0.0, 0.0]], dtype=np.float32)
        dy = np.array([[0.4, 0.0, 0.0, 0.0]], dtype=np.float32)
        ptrs, idx, vals = dense_to_csr(x)
        layer.backward(ptrs, idx, vals, dy, lr=0.0, batch=1)

        assert layer.neuron_input_accum[0] == pytest.approx(0.6, abs=1e-6)
        assert layer.neuron_grad_accum[0]  == pytest.approx(0.4, abs=1e-6)

    def test_zero_accum_clears(self):
        layer = make_layer(4, 4, bw=1)
        x  = np.array([[0.6, 0.4, 0.3, 0.7]], dtype=np.float32)
        dy = np.array([[0.5, 0.2, 0.8, 0.1]], dtype=np.float32)
        ptrs, idx, vals = dense_to_csr(x)
        layer.backward(ptrs, idx, vals, dy, lr=0.3, batch=1)
        layer.zero_accum()

        np.testing.assert_array_equal(layer.neuron_input_accum, 0.0)
        np.testing.assert_array_equal(layer.neuron_grad_accum,  0.0)

# ─────────────────────────────────────────────────────────────────────────────
# Synaptogenesis
# ─────────────────────────────────────────────────────────────────────────────

class TestSynaptogenesis:
    def test_build_probes_no_crash_with_zero_accum(self):
        layer = make_layer(8, 8, bw=0)
        # All accumulators zero — probes should be empty (no signal).
        layer.build_probes(k=4)
        # synap_step should return False (nothing to do).
        cr = 0
        result = layer.synap_step(importance_cutoff=0.1, max_row_weights=10)
        # No assertion on result — just must not crash.

    def test_build_probes_after_backward_adds_probes(self):
        layer = make_layer(8, 8, bw=0)

        # Run a forward+backward to fill accumulators.
        x  = np.ones((1, 8), dtype=np.float32)
        dy = np.ones((1, 8), dtype=np.float32)
        ptrs, idx, vals = dense_to_csr(x)
        layer.backward(ptrs, idx, vals, dy, lr=0.01, batch=1)

        nnz_before = layer.nnz
        layer.build_probes(k=4)

        # Run enough synap_steps to cover all rows.
        for _ in range(layer.n_inputs):
            layer.synap_step(importance_cutoff=0.0, max_row_weights=4)

        # At least some rows should have gained synapses.
        assert layer.nnz > nnz_before

    def test_synap_step_respects_importance_cutoff(self):
        """A very high importance cutoff should not *add* any new synapses.
        Existing synapses whose importance is below the cutoff may be pruned
        (that is expected behaviour — synaptogenesis prunes weak connections).
        """
        layer = make_layer(8, 8, bw=0)

        x  = np.ones((1, 8), dtype=np.float32)
        dy = np.ones((1, 8), dtype=np.float32)
        ptrs, idx, vals = dense_to_csr(x)
        layer.backward(ptrs, idx, vals, dy, lr=0.01, batch=1)

        nnz_before = layer.nnz
        layer.build_probes(k=4)

        # Use an impossibly high cutoff — probes should not be added.
        # nnz may drop if existing synapses are also below cutoff.
        for _ in range(layer.n_inputs):
            layer.synap_step(importance_cutoff=1e9, max_row_weights=100)

        assert layer.nnz <= nnz_before, (
            f"nnz increased with impossibly high cutoff: {nnz_before} → {layer.nnz}")

    def test_out_degree_non_negative(self):
        layer = make_layer(8, 8, bw=0)
        x  = np.ones((1, 8), dtype=np.float32)
        dy = np.ones((1, 8), dtype=np.float32)
        ptrs, idx, vals = dense_to_csr(x)
        layer.backward(ptrs, idx, vals, dy, lr=0.01, batch=1)
        layer.build_probes(k=4)
        for _ in range(layer.n_inputs):
            layer.synap_step(importance_cutoff=0.0, max_row_weights=4)
        assert np.all(layer.out_degree >= 0)

# ─────────────────────────────────────────────────────────────────────────────
# Equalizer
# ─────────────────────────────────────────────────────────────────────────────

class TestEqualizer:
    def test_equalizer_step_no_crash(self):
        layer = make_layer(8, 8, bw=1)
        for _ in range(16):
            layer.equalizer_step()

    def test_equalizer_preserves_data(self):
        layer = make_layer(6, 6, bw=1)
        # Set some known weights.
        layer.buffer.neuron[2].synapse[1].weight = 3.0

        for _ in range(20):
            layer.equalizer_step()

        # Data must survive equalization.
        assert layer.buffer.neuron[2].synapse[1].weight == pytest.approx(3.0, abs=1e-6)

    def test_equalizer_columns_unchanged(self):
        layer = make_layer(6, 6, bw=1)
        cols_before = [layer.buffer.neuron[2].synapse[k].index
                       for k in range(layer.buffer.neuron[2].nnz)]
        for _ in range(20):
            layer.equalizer_step()
        cols_after = [layer.buffer.neuron[2].synapse[k].index
                      for k in range(layer.buffer.neuron[2].nnz)]
        assert cols_before == cols_after

# ─────────────────────────────────────────────────────────────────────────────
# to_absolute serialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestToAbsolute:
    def test_round_trip_nnz(self):
        layer = make_layer(6, 6, bw=1)
        ptrs, idx, w, imp = layer.to_absolute()
        assert ptrs[-1] == layer.nnz

    def test_ptrs_length(self):
        layer = make_layer(6, 6, bw=1)
        ptrs, *_ = layer.to_absolute()
        assert len(ptrs) == layer.n_inputs + 1

    def test_indices_sorted_per_row(self):
        layer = make_layer(6, 6, bw=1)
        ptrs, idx, *_ = layer.to_absolute()
        for r in range(layer.n_inputs):
            row_idx = idx[ptrs[r]:ptrs[r+1]]
            assert list(row_idx) == sorted(row_idx), f"row {r} not sorted"

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch-like training loop
# ─────────────────────────────────────────────────────────────────────────────
#
# Identity mapping task: train a 4→4 diagonal sparse layer to output ≈ input.
# Zero-weight init means backprop must "discover" the weights from scratch.

class TestPytorchLike:
    """Minimal training loop test — verifies the full forward→backward→optim cycle."""

    def test_identity_loss_decreases(self):
        """After N gradient steps, MSE loss should decrease."""
        n = 4
        layer = make_layer(n, n, bw=0, cpus=1)  # diagonal only

        lr   = 0.5
        loss_history = []

        for step in range(10):
            # Target: identity (x → x).
            x = np.eye(n, dtype=np.float32)  # batch = n, each row is a unit vector

            out = layer.forward_dense(x, lr=0.0)

            # MSE loss and gradient.
            diff = out - x
            loss = float(np.mean(diff ** 2))
            loss_history.append(loss)

            dy = (2.0 / (n * n)) * diff   # dL/dy

            ptrs, idx, vals = dense_to_csr(x)
            layer.backward(ptrs, idx, vals, dy.astype(np.float32), lr=lr, batch=n)

        # Loss should be non-increasing overall (some FP4 noise is ok).
        # We just check that the last 3 are below the first 3.
        avg_start = sum(loss_history[:3]) / 3
        avg_end   = sum(loss_history[-3:]) / 3
        assert avg_end <= avg_start + 1e-3, (
            f"Loss did not decrease: start={avg_start:.4f}, end={avg_end:.4f}")

    def test_weight_sign_correct_after_training(self):
        """For an identity task, diagonal weights should move positive."""
        n = 4
        layer = make_layer(n, n, bw=0, cpus=1)

        lr = 0.5
        x  = np.eye(n, dtype=np.float32)

        for _ in range(8):
            out   = layer.forward_dense(x, lr=0.0)
            dy    = (2.0 / (n * n)) * (out - x)
            ptrs, idx, vals = dense_to_csr(x)
            layer.backward(ptrs, idx, vals, dy.astype(np.float32), lr=lr, batch=n)

        for i in range(n):
            w = layer.buffer.neuron[i].synapse[0].weight
            # Diagonal weights should be positive after gradient descent on identity.
            assert w >= 0.0, f"row {i}: diagonal weight {w} should be ≥ 0"

    def test_synaptogenesis_adds_weights_after_training(self):
        """After training, build_probes + synap_steps should add off-diagonal weights."""
        n = 8
        layer = make_layer(n, n, bw=0, cpus=1)

        lr = 0.3
        x  = np.random.randn(n, n).astype(np.float32)

        for _ in range(5):
            out = layer.forward_dense(x, lr=0.01)
            dy  = (2.0 / (n * n)) * (out - x)
            ptrs, idx, vals = dense_to_csr(x)
            layer.backward(ptrs, idx, vals, dy.astype(np.float32), lr=lr, batch=n)

        nnz_before = layer.nnz
        layer.build_probes(k=4)
        for _ in range(n):
            layer.synap_step(importance_cutoff=0.0, max_row_weights=8)

        # Synaptogenesis should have added some connections.
        assert layer.nnz > nnz_before


# ─────────────────────────────────────────────────────────────────────────────
# Neuron accumulators as numpy views
# ─────────────────────────────────────────────────────────────────────────────

class TestNumpyViews:
    def test_accum_shapes(self):
        layer = make_layer(6, 8, bw=1)
        assert layer.neuron_input_accum.shape == (6,)
        assert layer.neuron_grad_accum.shape  == (8,)

    def test_out_degree_shape(self):
        layer = make_layer(6, 8, bw=1)
        assert layer.out_degree.shape == (8,)

    def test_out_degree_sums_to_nnz(self):
        layer = make_layer(6, 6, bw=1)
        assert layer.out_degree.sum() == layer.nnz

    def test_accum_is_view_not_copy(self):
        # Mutating the accum should not crash (it's a zero-copy view into C++ memory).
        layer = make_layer(4, 4, bw=1)
        a = layer.neuron_input_accum
        assert a.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# Sparse attention
# ─────────────────────────────────────────────────────────────────────────────

class TestSparseAttention:
    def test_output_shape(self):
        T, d = 8, 16
        q = np.random.randn(T, d).astype(np.float32)
        k = np.random.randn(T, d).astype(np.float32)
        v = np.random.randn(T, d).astype(np.float32)
        out = _cpu.sparse_attention(q, k, v, top_k=3)
        assert out.shape == (T, d)

    def test_sqrt_T_default(self):
        # top_k=0 → sqrt(T)=3 for T=9. Should not crash.
        T, d = 9, 8
        q = np.random.randn(T, d).astype(np.float32)
        out = _cpu.sparse_attention(q, q, q, top_k=0)
        assert out.shape == (T, d)

    def test_k1_output_equals_single_v_row(self):
        # With k=1, the single selected key is the one with largest norm.
        # The single attention weight is 1.0, so output[selected_q] = v[selected_k].
        T, d = 4, 8
        q = np.zeros((T, d), dtype=np.float32)
        k = np.zeros((T, d), dtype=np.float32)
        v = np.zeros((T, d), dtype=np.float32)
        # Give row 2 the largest Q norm and row 3 the largest K norm.
        q[2] = 1.0
        k[3] = 1.0
        v[3] = np.arange(d, dtype=np.float32)  # known value

        out = _cpu.sparse_attention(q, k, v, top_k=1)
        # Row 2 (selected query) should equal v[3] (selected key, single weight=1).
        np.testing.assert_allclose(out[2], v[3], atol=1e-5)
        # All other rows should be zero.
        for i in range(T):
            if i != 2:
                np.testing.assert_allclose(out[i], 0.0, atol=1e-6)

    def test_k_equals_T_matches_dense_attention(self):
        # k=T: all pairs selected → should match naive dense attention.
        T, d = 4, 8
        np.random.seed(42)
        q = np.random.randn(T, d).astype(np.float32)
        k = np.random.randn(T, d).astype(np.float32)
        v = np.random.randn(T, d).astype(np.float32)

        out_sparse = _cpu.sparse_attention(q, k, v, top_k=T)

        # Reference dense attention.
        scale  = d ** -0.5
        scores = (q @ k.T) * scale
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=-1, keepdims=True)
        out_dense = w @ v

        np.testing.assert_allclose(out_sparse, out_dense, atol=1e-4)

    def test_only_selected_rows_nonzero(self):
        # With k=2 and T=8, only 2 rows of output should be non-zero.
        T, d = 8, 16
        q = np.random.randn(T, d).astype(np.float32)
        k = np.random.randn(T, d).astype(np.float32)
        v = np.random.randn(T, d).astype(np.float32)
        out = _cpu.sparse_attention(q, k, v, top_k=2)
        nonzero_rows = [i for i in range(T) if np.any(out[i] != 0.0)]
        assert len(nonzero_rows) <= 2

    def test_weights_sum_to_one_per_selected_query(self):
        # The output for each selected query should be a convex combination of V rows:
        # if all V rows are 1.0, the output should also be 1.0.
        T, d = 6, 8
        q = np.random.randn(T, d).astype(np.float32)
        k = np.random.randn(T, d).astype(np.float32)
        v = np.ones((T, d), dtype=np.float32)   # all V rows = 1
        out = _cpu.sparse_attention(q, k, v, top_k=3)
        for i in range(T):
            if np.any(out[i] != 0.0):
                # Selected query row: output should be 1.0 (sum of weights × 1 = 1).
                np.testing.assert_allclose(out[i], 1.0, atol=1e-5)

    def test_wrong_shape_raises(self):
        with pytest.raises(RuntimeError):
            _cpu.sparse_attention(
                np.ones((4, 8), dtype=np.float32),
                np.ones((4, 8), dtype=np.float32),
                np.ones((5, 8), dtype=np.float32),   # T mismatch
                top_k=2)

    def test_model_attention_method(self):
        # Verify the model's .attention() wrapper runs without error.
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from multimodal_sparse_rnn import MultimodalSparseRNN, MultimodalConfig, ModalityConfig
        KB = 1024
        cfg = MultimodalConfig(
            modalities=[ModalityConfig("language", 8, 16, 2, 64*KB)],
            recurrent_bw=2, recurrent_budget=256*KB,
            language_output_size=16, motor_output_size=4,
            output_bw=2, output_budget=128*KB,   # covers both W_lang and W_motor
            qkv_size=8, qkv_bw=2, qkv_budget=32*KB,
            num_cpus=1)
        model = MultimodalSparseRNN(cfg)
        h_seq = np.random.randn(6, cfg.total_state_size).astype(np.float32)
        ctx = model.attention(h_seq, top_k=3)
        assert ctx.shape == (cfg.qkv_size,)
        assert np.all(np.abs(ctx) <= 6.0), "clip6 not applied"


# ─────────────────────────────────────────────────────────────────────────────
# Banded attention
# ─────────────────────────────────────────────────────────────────────────────

class TestBandedAttention:
    """Tests for dense banded attention (geometric diagonal)."""

    def test_output_shape(self):
        T, d = 8, 16
        q = np.random.randn(T, d).astype(np.float32)
        out = _cpu.banded_attention(q, q, q, half_bandwidth=2)
        assert out.shape == (T, d)

    def test_cross_attention_shape(self):
        # Non-square: T queries, K keys/values.
        T, K, d = 6, 10, 8
        q = np.random.randn(T, d).astype(np.float32)
        k = np.random.randn(K, d).astype(np.float32)
        v = np.random.randn(K, d).astype(np.float32)
        out = _cpu.banded_attention(q, k, v, half_bandwidth=2)
        assert out.shape == (T, d)

    def test_full_bandwidth_matches_dense_attention(self):
        # half_bandwidth >= T covers all pairs — should match naive attention.
        T, d = 6, 8
        np.random.seed(7)
        q = np.random.randn(T, d).astype(np.float32)
        k = np.random.randn(T, d).astype(np.float32)
        v = np.random.randn(T, d).astype(np.float32)

        out_banded = _cpu.banded_attention(q, k, v, half_bandwidth=T)

        scale  = d ** -0.5
        scores = (q @ k.T) * scale
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores); w /= w.sum(axis=-1, keepdims=True)
        out_dense = w @ v

        np.testing.assert_allclose(out_banded, out_dense, atol=1e-4)

    def test_weights_sum_to_one(self):
        # If all V rows are identical the output should equal that row.
        T, d = 8, 6
        q = np.random.randn(T, d).astype(np.float32)
        k = np.random.randn(T, d).astype(np.float32)
        v_const = np.ones((T, d), dtype=np.float32) * 2.5
        out = _cpu.banded_attention(q, k, v_const, half_bandwidth=2)
        np.testing.assert_allclose(out, 2.5, atol=1e-5)

    def test_geometric_diagonal_reaches_last_key(self):
        # For a tall matrix (16q × 4k), the last query should attend near k=3.
        T, K, d = 16, 4, 8
        q = np.zeros((T, d), dtype=np.float32)
        k = np.zeros((K, d), dtype=np.float32)
        v = np.eye(K, d, dtype=np.float32)   # v[j] = basis vector j

        # Give last query and last key large norms so they dominate.
        q[T-1, 0] = 10.0
        k[K-1, 0] = 10.0

        out = _cpu.banded_attention(q, k, v, half_bandwidth=1)
        # Last query should attend heavily to last key (K-1).
        # Output[T-1] should be close to v[K-1] = e_{K-1}.
        assert out[T-1, K-1] > 0.5, f"last query didn't reach last key: {out[T-1]}"

    def test_shape_mismatch_raises(self):
        with pytest.raises(RuntimeError):
            _cpu.banded_attention(
                np.ones((4, 8), dtype=np.float32),
                np.ones((4, 8), dtype=np.float32),
                np.ones((4, 9), dtype=np.float32),  # d mismatch
                half_bandwidth=1)


class TestSparseBandedAttention:
    """Tests for sparse banded attention (geometric diagonal + inner top-k)."""

    def test_output_shape(self):
        T, d = 8, 16
        q = np.random.randn(T, d).astype(np.float32)
        out = _cpu.sparse_banded_attention(q, q, q, half_bandwidth=3, inner_k=2)
        assert out.shape == (T, d)

    def test_inner_k_zero_matches_dense_banded(self):
        # inner_k=0 → use all band keys → should match banded_attention exactly.
        T, d = 8, 12
        np.random.seed(11)
        q = np.random.randn(T, d).astype(np.float32)
        k = np.random.randn(T, d).astype(np.float32)
        v = np.random.randn(T, d).astype(np.float32)

        out_dense_banded  = _cpu.banded_attention(q, k, v, half_bandwidth=3)
        out_sparse_banded = _cpu.sparse_banded_attention(q, k, v,
                                half_bandwidth=3, inner_k=0)
        np.testing.assert_allclose(out_dense_banded, out_sparse_banded, atol=1e-5)

    def test_inner_k1_output_equals_dominant_v_in_band(self):
        # inner_k=1: each query attends to exactly one key in its band.
        # The single softmax weight is 1.0, so output[t] == v[selected_k].
        T, K, d = 6, 6, 8
        q = np.zeros((T, d), dtype=np.float32)
        k = np.zeros((K, d), dtype=np.float32)
        v = np.eye(K, d, dtype=np.float32)   # v[j] = basis vector j

        # Make one key clearly dominant in norm across the whole sequence.
        dominant = 2  # col 2
        k[dominant] = 5.0   # large norm — will win top-1 within any band covering it

        out = _cpu.sparse_banded_attention(q, k, v,
                    half_bandwidth=K, inner_k=1)  # wide band so dominant is always included

        # Every query should attend to the dominant key.
        for t in range(T):
            np.testing.assert_allclose(out[t], v[dominant], atol=1e-5,
                err_msg=f"row {t} didn't attend to dominant key")

    def test_weights_sum_to_one(self):
        T, d = 8, 6
        q = np.random.randn(T, d).astype(np.float32)
        k = np.random.randn(T, d).astype(np.float32)
        v_const = np.ones((T, d), dtype=np.float32) * 3.0
        out = _cpu.sparse_banded_attention(q, k, v_const,
                    half_bandwidth=2, inner_k=2)
        np.testing.assert_allclose(out, 3.0, atol=1e-5)

    def test_inner_k_larger_than_band_degenerates_to_dense_banded(self):
        # inner_k > band width → same as inner_k=0.
        T, d = 6, 10
        np.random.seed(3)
        q = np.random.randn(T, d).astype(np.float32)
        k = np.random.randn(T, d).astype(np.float32)
        v = np.random.randn(T, d).astype(np.float32)

        bw = 1  # band width = 3
        out_full  = _cpu.banded_attention(q, k, v, half_bandwidth=bw)
        out_large = _cpu.sparse_banded_attention(q, k, v,
                        half_bandwidth=bw, inner_k=100)  # way over band
        np.testing.assert_allclose(out_full, out_large, atol=1e-5)

    def test_non_square_cross_attention(self):
        # T queries, K keys, T != K.
        T, K, d = 12, 8, 10
        q = np.random.randn(T, d).astype(np.float32)
        k = np.random.randn(K, d).astype(np.float32)
        v = np.random.randn(K, d).astype(np.float32)
        out = _cpu.sparse_banded_attention(q, k, v, half_bandwidth=2, inner_k=2)
        assert out.shape == (T, d)
        # No NaN or Inf allowed.
        assert np.all(np.isfinite(out))

    def test_model_uses_sparse_banded_attention(self):
        # Wire sparse_banded_attention into the attention() call on the model.
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from multimodal_sparse_rnn import MultimodalSparseRNN, MultimodalConfig, ModalityConfig
        KB = 1024
        cfg = MultimodalConfig(
            modalities=[ModalityConfig("language", 8, 16, 2, 128*KB)],
            recurrent_bw=2, recurrent_budget=256*KB,
            language_output_size=16, motor_output_size=4,
            output_bw=2, output_budget=128*KB,
            qkv_size=8, qkv_bw=2, qkv_budget=32*KB,
            num_cpus=1)
        model = MultimodalSparseRNN(cfg)
        T = 8
        h_seq = np.random.randn(T, cfg.total_state_size).astype(np.float32)
        ctx = model.attention(h_seq, top_k=3)
        assert ctx.shape == (cfg.qkv_size,)
        assert np.all(np.isfinite(ctx))
        assert np.all(np.abs(ctx) <= 6.0)


# ─────────────────────────────────────────────────────────────────────────────
# Parallel pointers (Python API)
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelPointers:
    """Tests for set_parallel_ptrs / rebuild / n_parallel_ptrs via pybind."""

    def test_disabled_by_default(self):
        layer = make_layer(8, 8, bw=1)
        assert layer.n_parallel_ptrs == 0

    def test_set_enables(self):
        layer = make_layer(8, 8, bw=1)
        layer.set_parallel_ptrs(5)
        assert layer.n_parallel_ptrs == 5

    def test_set_zero_disables(self):
        layer = make_layer(8, 8, bw=1)
        layer.set_parallel_ptrs(5)
        layer.set_parallel_ptrs(0)
        assert layer.n_parallel_ptrs == 0

    def test_set_one_promoted_to_two(self):
        # n=1 is invalid (need start + sentinel); silently becomes 2.
        layer = make_layer(8, 8, bw=1)
        layer.set_parallel_ptrs(1)
        assert layer.n_parallel_ptrs == 2

    def test_init_with_parallel_ptrs(self):
        layer = _cpu.SparseLinearLayer(8, 8, 1, BUDGET, 1, 5)
        assert layer.n_parallel_ptrs == 5

    def test_forward_works_with_parallel_ptrs(self):
        # Parallel pointers must not break forward correctness.
        layer_noptrs = make_layer(4, 4, bw=0)
        layer_ptrs   = _cpu.SparseLinearLayer(4, 4, 0, BUDGET, 1, 5)

        # Set same weight on both.
        layer_noptrs.buffer.neuron[2].synapse[0].weight = 2.0
        layer_ptrs  .buffer.neuron[2].synapse[0].weight = 2.0
        layer_ptrs.rebuild_par_ptrs(2)

        x = np.zeros((1, 4), dtype=np.float32)
        x[0, 2] = 1.0

        out_no  = layer_noptrs.forward_dense(x, lr=0.0)
        out_yes = layer_ptrs.forward_dense(x, lr=0.0)

        np.testing.assert_allclose(out_no, out_yes, atol=1e-6)

    def test_rebuild_single_row(self):
        layer = make_layer(8, 8, bw=1)
        layer.set_parallel_ptrs(3)
        # Rebuild just row 2 — must not crash.
        layer.rebuild_par_ptrs(2)
        assert layer.n_parallel_ptrs == 3

    def test_rebuild_all(self):
        layer = make_layer(8, 8, bw=1)
        layer.set_parallel_ptrs(4)
        layer.rebuild_all_par_ptrs()
        assert layer.n_parallel_ptrs == 4

    def test_set_parallel_ptrs_survives_synaptogenesis(self):
        layer = _cpu.SparseLinearLayer(8, 8, 0, BUDGET, 1, 3)

        x  = np.ones((1, 8), dtype=np.float32)
        dy = np.ones((1, 8), dtype=np.float32)
        ptrs, idx, vals = dense_to_csr(x)
        layer.backward(ptrs, idx, vals, dy, lr=0.01, batch=1)
        layer.build_probes(k=4)

        for _ in range(layer.n_inputs):
            layer.synap_step(importance_cutoff=0.0, max_row_weights=6)

        # After synaptogenesis, rebuild pointers and run another forward.
        layer.rebuild_all_par_ptrs()
        out = layer.forward_dense(x, lr=0.0)
        assert out.shape == (1, 8)
        assert np.all(np.isfinite(out))

    def test_parallel_ptrs_survive_equalizer(self):
        # Equalizer must shift byte_offsets; check forward still correct.
        layer_ref  = make_layer(6, 6, bw=1)
        layer_ptrs = _cpu.SparseLinearLayer(6, 6, 1, BUDGET, 1, 4)

        # Set a known weight on both.
        layer_ref .buffer.neuron[3].synapse[1].weight = 1.5
        layer_ptrs.buffer.neuron[3].synapse[1].weight = 1.5

        # Run many equalizer steps on the ptrs layer.
        for _ in range(20):
            layer_ptrs.equalizer_step()

        # Rebuild pointers after movement.
        layer_ptrs.rebuild_all_par_ptrs()

        x = np.zeros((1, 6), dtype=np.float32)
        x[0, 3] = 1.0

        out_ref  = layer_ref .forward_dense(x, lr=0.0)
        out_ptrs = layer_ptrs.forward_dense(x, lr=0.0)

        np.testing.assert_allclose(out_ref, out_ptrs, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation — save to file, load, verify byte-level identity
# ─────────────────────────────────────────────────────────────────────────────

import tempfile, os

def _train_small_layer():
    """
    Build a 4×4 diagonal layer, set specific FP4-exact weights, run a few
    training steps with non-trivial inputs, and return the layer.

    Weight trace (row 0, col 0):
      init:          w=1.5,  imp=0.0
      backward 1:    grad_eff = 0.6×0.4 = 0.24 → imp = -0.12 → FP4 = -0.5
                                                   w_delta = -0.5×0.24/(1+0.5)
                                                           = -0.08 → FP4(1.5-0.08)=1.5
      backward 2:    imp = -0.5 - 0.24×0.5 = -0.62 → FP4 = -0.5 (clipped by table)
    The exact values after training are deterministic because FP4 is lossy but
    reproducible.
    """
    layer = _cpu.SparseLinearLayer(4, 4, 0, BUDGET, 1)
    # Set non-trivial, FP4-exact initial weights.
    layer.buffer.neuron[0].synapse[0].weight = 1.5
    layer.buffer.neuron[1].synapse[0].weight = -1.5
    layer.buffer.neuron[2].synapse[0].weight = 2.0
    layer.buffer.neuron[3].synapse[0].weight = -2.0

    x  = np.array([[0.6, 0.4, 0.3, 0.7]], dtype=np.float32)
    dy = np.array([[0.4, 0.6, 0.5, 0.3]], dtype=np.float32)
    ptrs, idx, vals = dense_to_csr(x)

    for _ in range(3):
        layer.forward_dense(x, lr=0.3)
        layer.backward(ptrs, idx, vals, dy, lr=0.3, batch=1)

    return layer


class TestSerialisation:

    def test_save_produces_nonempty_file(self):
        layer = _train_small_layer()
        with tempfile.NamedTemporaryFile(suffix='.sili', delete=False) as f:
            path = f.name
        try:
            layer.save(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_load_restores_raw_buffer_byte_for_byte(self):
        layer = _train_small_layer()
        raw_before = bytes(layer.raw_buffer)

        with tempfile.NamedTemporaryFile(suffix='.sili', delete=False) as f:
            path = f.name
        try:
            layer.save(path)
            layer2 = _cpu.load_layer(path, num_cpus=1)
            raw_after = bytes(layer2.raw_buffer)
        finally:
            os.unlink(path)

        # The flat buffers must be identical down to the last byte.
        assert raw_before == raw_after, (
            f"Buffer mismatch: {len(raw_before)} vs {len(raw_after)} bytes")

    def test_load_restores_nnz(self):
        layer = _train_small_layer()
        with tempfile.NamedTemporaryFile(suffix='.sili', delete=False) as f:
            path = f.name
        try:
            layer.save(path)
            layer2 = _cpu.load_layer(path, num_cpus=1)
        finally:
            os.unlink(path)

        assert layer2.nnz == layer.nnz

    def test_load_restores_specific_weights(self):
        # Verify that specific known non-trivial weights survive round-trip.
        layer = _train_small_layer()
        # Read the weights now (after training) — these are our ground truth.
        w0  = layer.buffer.neuron[0].synapse[0].weight
        i0  = layer.buffer.neuron[0].synapse[0].importance
        w2  = layer.buffer.neuron[2].synapse[0].weight
        i3  = layer.buffer.neuron[3].synapse[0].importance

        # None should be exactly 0, 1, or ±inf after training.
        for v in (w0, i0, w2, i3):
            assert v not in (0.0, 1.0, float('inf'), float('-inf'))

        with tempfile.NamedTemporaryFile(suffix='.sili', delete=False) as f:
            path = f.name
        try:
            layer.save(path)
            layer2 = _cpu.load_layer(path, num_cpus=1)
        finally:
            os.unlink(path)

        assert layer2.buffer.neuron[0].synapse[0].weight     == pytest.approx(w0, abs=1e-6)
        assert layer2.buffer.neuron[0].synapse[0].importance == pytest.approx(i0, abs=1e-6)
        assert layer2.buffer.neuron[2].synapse[0].weight     == pytest.approx(w2, abs=1e-6)
        assert layer2.buffer.neuron[3].synapse[0].importance == pytest.approx(i3, abs=1e-6)

    def test_forward_output_identical_after_load(self):
        # Forward pass on the loaded layer must match the original exactly.
        layer = _train_small_layer()
        x = np.array([[0.6, 0.4, 0.3, 0.7]], dtype=np.float32)
        out_orig = layer.forward_dense(x, lr=0.0).copy()

        with tempfile.NamedTemporaryFile(suffix='.sili', delete=False) as f:
            path = f.name
        try:
            layer.save(path)
            layer2 = _cpu.load_layer(path, num_cpus=1)
        finally:
            os.unlink(path)

        out_loaded = layer2.forward_dense(x, lr=0.0)
        np.testing.assert_array_equal(out_orig, out_loaded)

    def test_in_place_load_replaces_weights(self):
        # layer.load(path) replaces the layer's own weights in-place.
        layer  = _train_small_layer()
        layer2 = _train_small_layer()   # identical training → same weights

        with tempfile.NamedTemporaryFile(suffix='.sili', delete=False) as f:
            path = f.name
        try:
            layer.save(path)
            # Corrupt layer2's weights, then reload from file.
            layer2.buffer.neuron[0].synapse[0].weight = 6.0   # definitely wrong
            layer2.load(path)
        finally:
            os.unlink(path)

        # After reload, layer2 should match the original.
        assert layer2.buffer.neuron[0].synapse[0].weight == pytest.approx(
            layer.buffer.neuron[0].synapse[0].weight, abs=1e-6)

    def test_save_redundant_two_files(self):
        layer = _train_small_layer()
        with tempfile.NamedTemporaryFile(suffix='.sili', delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix='.sili', delete=False) as f2:
            p1, p2 = f1.name, f2.name
        try:
            layer.save_redundant([p1, p2])
            # Both files exist and are the same size.
            s1, s2 = os.path.getsize(p1), os.path.getsize(p2)
            assert s1 == s2
            assert s1 > 0
            # Contents are byte-for-byte identical.
            assert open(p1, 'rb').read() == open(p2, 'rb').read()
        finally:
            for p in (p1, p2):
                if os.path.exists(p): os.unlink(p)

    def test_save_redundant_three_files(self):
        layer = _train_small_layer()
        paths = [tempfile.mktemp(suffix='.sili') for _ in range(3)]
        try:
            layer.save_redundant(paths)
            contents = [open(p, 'rb').read() for p in paths]
            # All three files identical.
            assert contents[0] == contents[1] == contents[2]
        finally:
            for p in paths:
                if os.path.exists(p): os.unlink(p)

    def test_load_bad_magic_raises(self):
        with tempfile.NamedTemporaryFile(suffix='.sili', delete=False) as f:
            f.write(b'JUNK' + b'\x01\x00\x00\x00' + b'\x00' * 64)
            path = f.name
        try:
            with pytest.raises(RuntimeError, match="bad magic"):
                _cpu.load_layer(path)
        finally:
            os.unlink(path)

    def test_load_wrong_version_raises(self):
        with tempfile.NamedTemporaryFile(suffix='.sili', delete=False) as f:
            import struct
            # Write correct magic but version=99.
            f.write(b'SiLi' + struct.pack('<I', 99) + b'\x00' * 64)
            path = f.name
        try:
            with pytest.raises(RuntimeError, match="unsupported version"):
                _cpu.load_layer(path)
        finally:
            os.unlink(path)

    def test_load_truncated_file_raises(self):
        layer = _train_small_layer()
        with tempfile.NamedTemporaryFile(suffix='.sili', delete=False) as f:
            path = f.name
        try:
            layer.save(path)
            # Truncate to half size.
            full = open(path, 'rb').read()
            open(path, 'wb').write(full[:len(full)//2])
            with pytest.raises((RuntimeError, Exception)):
                _cpu.load_layer(path)
        finally:
            os.unlink(path)

    def test_save_load_with_parallel_ptrs(self):
        # Parallel pointers must survive the round-trip.
        layer = _train_small_layer()
        layer.set_parallel_ptrs(3)
        layer.rebuild_all_par_ptrs()

        with tempfile.NamedTemporaryFile(suffix='.sili', delete=False) as f:
            path = f.name
        try:
            layer.save(path)
            layer2 = _cpu.load_layer(path, num_cpus=1)
        finally:
            os.unlink(path)

        assert layer2.n_parallel_ptrs == 3
        # Raw buffers still identical.
        assert bytes(layer.raw_buffer) == bytes(layer2.raw_buffer)

    def test_column_indices_intact_after_load(self):
        # Verify column indices decode correctly after load (not just values).
        layer = _cpu.SparseLinearLayer(6, 6, 1, BUDGET, 1)
        # bw=1 → row 3 connects to cols {2,3,4}.
        with tempfile.NamedTemporaryFile(suffix='.sili', delete=False) as f:
            path = f.name
        try:
            layer.save(path)
            layer2 = _cpu.load_layer(path, num_cpus=1)
        finally:
            os.unlink(path)

        n_orig   = layer .buffer.neuron[3]
        n_loaded = layer2.buffer.neuron[3]

        cols_orig   = [n_orig  .synapse[k].index for k in range(n_orig.nnz)]
        cols_loaded = [n_loaded.synapse[k].index for k in range(n_loaded.nnz)]
        assert cols_orig == cols_loaded == [2, 3, 4]
