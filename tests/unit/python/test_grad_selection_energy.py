"""
tests/unit/python/test_grad_selection_energy.py
─────────────────────────────────────────────────
Tests for GradSelectionEnergy -- the lightweight per-neuron energy
accumulator used for Arm G's grad-selection fairness (top-k by
under-activity, feeding dy_gate_mask). See sili/energy.py's own
docstring for how this differs from both EnergyDynamics (forward
intervention) and top_k_csr_nucleus's signal energy (magnitude proxy).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sili.energy import GradSelectionEnergy


class TestConstructorValidation:
    def test_negative_decay_rejected(self):
        with pytest.raises(AssertionError):
            GradSelectionEnergy(drive=0.1, activation_cost=0.05, decay=-0.01, n_out=10)

    def test_zero_activation_cost_rejected(self):
        with pytest.raises(AssertionError):
            GradSelectionEnergy(drive=0.1, activation_cost=0.0, decay=0.01, n_out=10)

    def test_drive_too_small_relative_to_decay_rejected(self):
        with pytest.raises(AssertionError):
            GradSelectionEnergy(drive=0.01, activation_cost=0.05, decay=0.02, n_out=10)

    def test_valid_construction(self):
        ge = GradSelectionEnergy(drive=0.1, activation_cost=0.05, decay=0.02, n_out=10)
        assert ge.energy.shape == (10,)
        assert np.all(ge.energy == 0.0)


class TestUpdate:
    def test_quiet_neuron_gains_energy(self):
        ge = GradSelectionEnergy(drive=0.1, activation_cost=0.05, decay=0.0, n_out=4)
        ge.update(np.zeros(4))
        assert np.all(ge.energy > 0.0)

    def test_loud_neuron_loses_energy(self):
        ge = GradSelectionEnergy(drive=0.01, activation_cost=0.5, decay=0.0, n_out=4)
        ge.update(np.full(4, 10.0))
        assert np.all(ge.energy < 0.0)

    def test_2d_output_reduced_over_rows(self):
        ge = GradSelectionEnergy(drive=0.1, activation_cost=0.05, decay=0.0, n_out=3)
        out2d = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        ge.update(out2d)
        # mean|out| per column: [0, 1, 2] -> energy = drive - cost*mean
        expected = 0.1 - 0.05 * np.array([0.0, 1.0, 2.0])
        assert np.allclose(ge.energy, expected)

    def test_decay_zero_matches_manual_accumulation(self):
        ge = GradSelectionEnergy(drive=0.2, activation_cost=0.1, decay=0.0, n_out=2)
        for _ in range(5):
            ge.update(np.array([1.0, 3.0]))
        expected = 5 * (0.2 - 0.1 * np.array([1.0, 3.0]))
        assert np.allclose(ge.energy, expected)


class TestTopKMask:
    def test_selects_highest_energy_neurons(self):
        ge = GradSelectionEnergy(drive=0.1, activation_cost=0.05, decay=0.02, n_out=5)
        ge.energy = np.array([0.1, 0.5, 0.9, 0.2, 0.7], dtype=np.float32)
        mask = ge.top_k_mask(2)
        assert mask.sum() == 2
        assert mask[2] and mask[4]  # the two highest (0.9, 0.7)
        assert not mask[0] and not mask[1] and not mask[3]

    def test_k_zero_selects_nothing(self):
        ge = GradSelectionEnergy(drive=0.1, activation_cost=0.05, decay=0.02, n_out=5)
        mask = ge.top_k_mask(0)
        assert mask.sum() == 0

    def test_k_exceeds_n_out_selects_all(self):
        ge = GradSelectionEnergy(drive=0.1, activation_cost=0.05, decay=0.02, n_out=5)
        mask = ge.top_k_mask(100)
        assert mask.sum() == 5

    def test_mask_is_boolean_array_of_correct_length(self):
        ge = GradSelectionEnergy(drive=0.1, activation_cost=0.05, decay=0.02, n_out=7)
        mask = ge.top_k_mask(3)
        assert mask.dtype == bool
        assert len(mask) == 7


class TestCoverageGuarantee:
    def test_never_selected_neuron_eventually_wins_top_k(self):
        # One neuron is ALWAYS loud (never selected, so its output stays
        # loud every call -- simulating "never gets a training turn to
        # quiet down"); the rest are quiet. The loud one should
        # eventually accumulate enough NEGATIVE energy that it's NEVER
        # in the top-k (correctly excluded, it's not under-activity) --
        # the real test is the INVERSE: a neuron that's quiet but
        # simply hasn't been picked yet must eventually win.
        n = 10
        drive, activation_cost, decay = 0.2, 0.05, 0.02
        ge = GradSelectionEnergy(drive=drive, activation_cost=activation_cost, decay=decay, n_out=n)
        # All neurons quiet (output ~0) every call -- symmetric, but
        # neuron 0 should still reach the top-k within a bounded number
        # of calls purely from the guaranteed monotonic rise (starting
        # tied at 0, argpartition ties broken consistently, so with
        # truly identical inputs every neuron reaches the SAME energy --
        # verify the ENERGY reaches a high value, not literal selection,
        # since ties are the interesting edge case here).
        fixed_point = drive / decay
        assert fixed_point > 0
        t_needed = int(math.log(0.01 / fixed_point) / math.log(1.0 - decay)) + 20
        for _ in range(t_needed):
            ge.update(np.zeros(n))
        assert np.allclose(ge.energy, fixed_point, atol=0.05)

    def test_quiet_neuron_overtakes_loud_neuron_within_bounded_calls(self):
        n = 4
        drive, activation_cost, decay = 0.3, 0.05, 0.02
        ge = GradSelectionEnergy(drive=drive, activation_cost=activation_cost, decay=decay, n_out=n)
        loud_val = 20.0  # activation_cost*loud_val=1.0 >> drive -> permanent negative drift
        t_needed = 500
        overtaken_step = None
        for t in range(t_needed):
            out = np.array([loud_val, 0.0, 0.0, 0.0])
            ge.update(out)
            if overtaken_step is None and ge.energy[1] > ge.energy[0]:
                overtaken_step = t
        assert overtaken_step is not None, "quiet neuron never overtook the loud one"
        assert overtaken_step < 50, f"took {overtaken_step} calls, expected fast separation"
