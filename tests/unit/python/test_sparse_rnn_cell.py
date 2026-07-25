"""
tests/unit/python/test_sparse_rnn_cell.py
──────────────────────────────────────────
Dedicated tests for SparseRNNCell, CSR, BranchingRatioTracker, and the
Tensor CSR-delegation added by the A6 identifiability fix
(see sili_peridot/todolist.md Phase A6 for the design this implements).

SparseRNNCell previously had NO dedicated test coverage of its own --
SparseRNNAgent (which owns the only SparseRNNCell in the tree) is only
exercised by examples/atari_run.py, which needs gymnasium/ROMs and isn't a
practical unit test. These tests exercise SparseRNNCell/SparseRNNAgent
directly, standalone.

KNOWN PRE-EXISTING BUG (found while writing these, unrelated to the A6
changes themselves -- do not "fix" by weakening these tests):
sili.sparse_rnn.DISLDOLayer/SISLDOLayer call _cpu.DISLDOLayer(...)/
_cpu.SISLDOLayer(...), plus _SparseLayerBase.step()/.decay()/
.synaptogenesis() call self._c.optim_weights(lr)/.decay_importance(rate)/
.optim_synaptogenesis(...) -- NONE of these names exist on any C++ class
currently bound in cpu_backend.cpp. Only DISLDOLayerV is bound (a
different, incompatible API: forward(x, learning_rate)/backward(dy,
learning_rate) instead of forward(x)/backward(dy), synap_row_step instead
of a build_probes+optim_synaptogenesis pair, no optim_weights/
decay_importance at all). SparseLinearLayer (what FoldedLayer actually
uses, and what MiniCPM5 conversion + Mandelbrot's SparseCore/MistralCore
both build on -- see make_grown_sparse_layer in test_mandelbrot_rl.py) is
a THIRD, separate, currently-WORKING API: forward_dense(x, learning_rate)/
backward_dense(dy, learning_rate) apply the weight update inline, no
separate step()/decay() call at all. DISLDOLayer/SISLDOLayer/
SparseRNNCell/SparseRNNAgent are therefore stale relics of an even older
API generation that predates all three of these, not something the A6
changes broke -- SparseRNNCell could not have been constructed successfully
before this session either. Tests below that need actual C++ execution are
marked xfail(strict=True) with this same reason, so they'll flag loudly
(as an unexpected pass) once DISLDOLayer/SISLDOLayer are rebuilt on
SparseLinearLayer's real API rather than needing to be remembered by hand.
"""
from __future__ import annotations
import numpy as np
import pytest

from sili.tensor import Tensor
from sili.sparse_rnn import CSR, SparseRNNCell, SparseRNNAgent
from sili.energy import EnergyDynamics, BranchingRatioTracker

_DISLDO_BROKEN_REASON = (
    "Pre-existing, unrelated bug: _cpu.DISLDOLayer/_cpu.SISLDOLayer were "
    "never bound (only DISLDOLayerV exists, a different+incompatible API), "
    "and _SparseLayerBase.step()/.decay()/.synaptogenesis() call C++ methods "
    "(optim_weights/decay_importance/optim_synaptogenesis) that don't exist "
    "on any currently-bound class. See this file's module docstring."
)


def _small_cell(**kwargs):
    kwargs.setdefault("percent_active", 0.25)  # small state -> want several active
    return SparseRNNCell(n_inputs=6, state_size=12, max_weights=512,
                         num_cpus=1, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# CSR.nnz / CSR.from_kept_indices
# ─────────────────────────────────────────────────────────────────────────────

class TestCSRNnzAndFromKeptIndices:
    def test_nnz_matches_indices_length(self):
        csr = CSR.from_dense(np.random.randn(16).astype(np.float32), p=0.25, num_cpus=1)
        assert csr.nnz == len(csr.indices)

    def test_from_kept_indices_pulls_pregating_values(self):
        cols = 10
        values_source = np.arange(cols, dtype=np.float32) * 10.0  # [0,10,20,...,90]
        kept = np.array([2, 5, 7], dtype=np.int32)
        csr = CSR.from_kept_indices(kept, values_source, cols=cols)
        assert csr.rows == 1
        assert csr.cols == cols
        assert csr.nnz == 3
        np.testing.assert_array_equal(np.asarray(csr.indices), kept)
        np.testing.assert_allclose(np.asarray(csr.values), [20.0, 50.0, 70.0])

    def test_from_kept_indices_roundtrips_through_to_dense(self):
        cols = 8
        values_source = np.random.randn(cols).astype(np.float32)
        kept = np.array([0, 3, 6], dtype=np.int32)
        csr = CSR.from_kept_indices(kept, values_source, cols=cols)
        dense = csr.to_dense()[0]
        for i in kept:
            assert dense[i] == pytest.approx(values_source[i])
        for i in set(range(cols)) - set(kept.tolist()):
            assert dense[i] == 0.0

    def test_from_kept_indices_empty(self):
        csr = CSR.from_kept_indices(np.array([], dtype=np.int32),
                                    np.zeros(5, dtype=np.float32), cols=5)
        assert csr.nnz == 0
        assert csr.to_dense().shape == (1, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Tensor CSR delegation
# ─────────────────────────────────────────────────────────────────────────────

class TestTensorCSRDelegation:
    def test_dense_tensor_is_dense_not_csr(self):
        t = Tensor(np.zeros(4, dtype=np.float32))
        assert t.is_dense is True
        assert t.is_csr is False

    def test_csr_tensor_is_csr_not_dense(self):
        csr = CSR.from_dense(np.random.randn(8).astype(np.float32), p=0.25, num_cpus=1)
        t = csr.as_tensor()
        assert t.is_csr is True
        assert t.is_dense is False

    def test_getattr_delegates_nnz_rows_cols(self):
        csr = CSR.from_dense(np.random.randn(8).astype(np.float32), p=0.25, num_cpus=1)
        t = csr.as_tensor()
        assert t.nnz == csr.nnz
        assert t.rows == csr.rows
        assert t.cols == csr.cols

    def test_getattr_raises_for_unknown_attr_on_dense(self):
        t = Tensor(np.zeros(4, dtype=np.float32))
        with pytest.raises(AttributeError):
            _ = t.nnz  # dense data has no nnz -- should not silently return something

    def test_getattr_raises_for_attr_not_on_csr_either(self):
        csr = CSR.from_dense(np.random.randn(8).astype(np.float32), p=0.25, num_cpus=1)
        t = csr.as_tensor()
        with pytest.raises(AttributeError):
            _ = t.totally_made_up_attribute_xyz


# ─────────────────────────────────────────────────────────────────────────────
# BranchingRatioTracker
# ─────────────────────────────────────────────────────────────────────────────

class TestBranchingRatioTracker:
    def test_none_before_enough_history(self):
        t = BranchingRatioTracker(window=50)
        assert t.branching_ratio() is None
        t.update(1.0); t.update(2.0)
        assert t.branching_ratio() is None  # only 2 points

    def test_recovers_known_branching_ratio(self):
        # a[t+1] = m*a[t] + h, deterministic -- OLS should recover m closely.
        m, h, a = 0.9, 1.0, 5.0
        t = BranchingRatioTracker(window=200)
        for _ in range(100):
            t.update(a)
            a = m * a + h
        est = t.branching_ratio()
        assert est is not None
        assert est == pytest.approx(m, abs=0.02)

    def test_flat_history_returns_none(self):
        t = BranchingRatioTracker(window=50)
        for _ in range(10):
            t.update(3.0)  # zero variance
        assert t.branching_ratio() is None

    def test_window_caps_history_length(self):
        t = BranchingRatioTracker(window=5)
        for i in range(20):
            t.update(float(i))
        assert len(t._history) == 5
        assert t._history == [15.0, 16.0, 17.0, 18.0, 19.0]

    def test_reset_clears_history(self):
        t = BranchingRatioTracker(window=50)
        for i in range(10):
            t.update(float(i))
        t.reset()
        assert t.branching_ratio() is None
        assert t.avalanche_sizes() == []

    def test_avalanche_sizes_extracts_runs(self):
        t = BranchingRatioTracker(window=50)
        for a in [0, 1, 1, 0, 0, 1, 1, 1, 0, 1]:
            t.update(float(a))
        assert t.avalanche_sizes() == [2, 3, 1]

    def test_window_too_small_rejected(self):
        with pytest.raises(AssertionError):
            BranchingRatioTracker(window=2)


# ─────────────────────────────────────────────────────────────────────────────
# EnergyDynamics kept_indices
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyDynamicsKeptIndices:
    def test_kept_indices_count_matches_actual_p(self):
        ed = EnergyDynamics(drive=0.1, activation_cost=0.05, precision=0.01,
                            density=0.05, p=0.3)
        h = Tensor((np.random.randn(40) * 0.5).astype(np.float32))
        h_out, aux_loss, actual_p = ed.forward(h)
        assert ed.kept_indices is not None
        n = 40
        assert len(ed.kept_indices) == pytest.approx(actual_p * n, abs=1)

    def test_kept_indices_sorted_unique_in_range(self):
        ed = EnergyDynamics(drive=0.1, activation_cost=0.05, precision=0.01,
                            density=0.05, p=0.3)
        h = Tensor((np.random.randn(64) * 0.5).astype(np.float32))
        ed.forward(h)
        idx = ed.kept_indices
        assert np.all(idx >= 0) and np.all(idx < 64)
        assert np.array_equal(idx, np.sort(idx))
        assert len(np.unique(idx)) == len(idx)

    def test_density_override_does_not_mutate_stored_density(self):
        ed = EnergyDynamics(drive=0.1, activation_cost=0.05, precision=0.01,
                            density=0.05, p=0.3)
        h = Tensor((np.random.randn(20) * 0.5).astype(np.float32))
        ed.forward(h, density_override=0.2)
        assert ed.density == 0.05  # unchanged -- override is call-scoped only

    def test_p_density_inversion_rejected(self):
        with pytest.raises(AssertionError):
            EnergyDynamics(drive=0.1, activation_cost=0.05, precision=0.01,
                           density=0.5, p=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# SparseRNNCell
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.xfail(reason=_DISLDO_BROKEN_REASON, strict=True)
class TestSparseRNNCellConstruction:
    def test_density_below_p_by_design(self):
        cell = _small_cell()
        assert cell.energy.density <= cell.energy.p * 0.8 + 1e-9

    def test_has_branching_tracker(self):
        cell = _small_cell()
        assert isinstance(cell.branching_recurrent, BranchingRatioTracker)

    def test_dynamic_density_default_off(self):
        cell = _small_cell()
        assert cell.dynamic_density_from_branching_ratio is False


@pytest.mark.xfail(reason=_DISLDO_BROKEN_REASON, strict=True)
class TestSparseRNNCellForward:
    def test_step0_dense_fallback_produces_finite_output(self):
        cell = _small_cell()
        obs   = Tensor(np.random.randn(6).astype(np.float32))
        state = Tensor(np.zeros(12, dtype=np.float32))
        new_state, aux_loss, actual_p = cell(obs, state)
        assert new_state.is_dense
        assert np.all(np.isfinite(new_state.data))
        assert np.isfinite(float(aux_loss.data))
        assert 0.0 < actual_p <= 1.0

    def test_first_call_caches_kept_indices_and_pregate_h(self):
        cell = _small_cell()
        obs   = Tensor(np.random.randn(6).astype(np.float32))
        state = Tensor(np.zeros(12, dtype=np.float32))
        assert cell._prev_kept_indices is None
        cell(obs, state)
        assert cell._prev_kept_indices is not None
        assert cell._prev_h_dense is not None
        assert cell._prev_h_dense.shape == (12,)

    def test_second_call_uses_cached_kept_indices_not_from_dense(self, monkeypatch):
        cell = _small_cell()
        obs   = Tensor(np.random.randn(6).astype(np.float32))
        state = Tensor(np.zeros(12, dtype=np.float32))
        new_state, _, _ = cell(obs, state)

        calls = {"from_dense": 0, "from_kept_indices": 0}
        orig_from_dense = CSR.from_dense
        orig_from_kept  = CSR.from_kept_indices

        def spy_from_dense(*a, **k):
            calls["from_dense"] += 1
            return orig_from_dense(*a, **k)

        def spy_from_kept(*a, **k):
            calls["from_kept_indices"] += 1
            return orig_from_kept(*a, **k)

        monkeypatch.setattr(CSR, "from_dense", staticmethod(spy_from_dense))
        monkeypatch.setattr(CSR, "from_kept_indices", staticmethod(spy_from_kept))

        obs2 = Tensor(np.random.randn(6).astype(np.float32))
        cell(obs2, new_state.detach())

        assert calls["from_kept_indices"] == 1
        assert calls["from_dense"] == 0

    def test_reset_falls_back_to_from_dense_again(self, monkeypatch):
        cell = _small_cell()
        obs   = Tensor(np.random.randn(6).astype(np.float32))
        state = Tensor(np.zeros(12, dtype=np.float32))
        new_state, _, _ = cell(obs, state)
        cell.reset()
        assert cell._prev_kept_indices is None

        calls = {"from_dense": 0}
        orig_from_dense = CSR.from_dense
        def spy_from_dense(*a, **k):
            calls["from_dense"] += 1
            return orig_from_dense(*a, **k)
        monkeypatch.setattr(CSR, "from_dense", staticmethod(spy_from_dense))

        obs2 = Tensor(np.random.randn(6).astype(np.float32))
        cell(obs2, new_state.detach())
        assert calls["from_dense"] == 1

    def test_recurrent_activity_tracked_before_combining_with_input(self):
        cell = _small_cell()
        obs   = Tensor(np.random.randn(6).astype(np.float32))
        state = Tensor(np.zeros(12, dtype=np.float32))
        assert len(cell.branching_recurrent._history) == 0
        cell(obs, state)
        assert len(cell.branching_recurrent._history) == 1

    def test_backward_reaches_input_proj_and_recurrent_weights(self):
        # Gradient should flow: aux_loss -> h_out -> h -> input_proj/recurrent.
        # Weights live in C++ (parameters() == []), so verify indirectly via
        # neuron_grad_accum, which forward/backward populate through the
        # DISLDOLayer/SISLDOLayer _backward closures.
        cell = _small_cell()
        obs   = Tensor(np.random.randn(6).astype(np.float32))
        state = Tensor(np.zeros(12, dtype=np.float32))
        _, aux_loss, _ = cell(obs, state)
        aux_loss.backward()
        assert np.any(cell.input_proj.neuron_grad_accum != 0.0) or \
               np.any(cell.recurrent.neuron_grad_accum != 0.0)

    def test_multi_step_rollout_stays_finite(self):
        cell = _small_cell()
        state = Tensor(np.zeros(12, dtype=np.float32))
        for _ in range(15):
            obs = Tensor(np.random.randn(6).astype(np.float32))
            state, aux_loss, actual_p = cell(obs, state)
            assert np.all(np.isfinite(state.data))
            assert np.isfinite(float(aux_loss.data))
            state = state.detach()

    def test_dynamic_density_opt_in_stays_finite(self):
        cell = _small_cell(dynamic_density_from_branching_ratio=True, branching_window=20)
        state = Tensor(np.zeros(12, dtype=np.float32))
        for _ in range(10):
            obs = Tensor(np.random.randn(6).astype(np.float32))
            state, aux_loss, actual_p = cell(obs, state)
            assert np.all(np.isfinite(state.data))
            state = state.detach()
        # After enough steps the tracker should have a real estimate.
        assert cell.branching_recurrent.branching_ratio() is not None


@pytest.mark.xfail(reason=_DISLDO_BROKEN_REASON, strict=True)
class TestSparseRNNAgentSaveLoad:
    def test_state_stays_dense_across_steps(self, tmp_path):
        agent = SparseRNNAgent(n_inputs=6, n_actions=2, state_size=12,
                               max_weights=512, num_cpus=1, percent_active=0.25)
        for _ in range(5):
            obs = Tensor(np.random.randn(6).astype(np.float32))
            agent.train_step(obs)
        assert agent.state.is_dense  # save()/argmax rely on this staying true

    def test_save_load_roundtrip_resets_cache(self, tmp_path):
        agent = SparseRNNAgent(n_inputs=6, n_actions=2, state_size=12,
                               max_weights=512, num_cpus=1, percent_active=0.25)
        for _ in range(5):
            obs = Tensor(np.random.randn(6).astype(np.float32))
            agent.train_step(obs)
        assert agent.cell._prev_kept_indices is not None

        path = str(tmp_path / "agent.npz")
        agent.save(path)

        agent2 = SparseRNNAgent(n_inputs=6, n_actions=2, state_size=12,
                                max_weights=512, num_cpus=1, percent_active=0.25)
        agent2.load(path)
        # A freshly loaded agent's cell cache must be invalidated -- the
        # loaded state didn't come from THIS cell's own last forward call.
        assert agent2.cell._prev_kept_indices is None

        # And it should still be able to take a step without error.
        obs = Tensor(np.random.randn(6).astype(np.float32))
        action = agent2.train_step(obs)
        assert action in (0, 1)
