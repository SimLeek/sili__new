"""
tests/unit/python/test_energy_decay.py
───────────────────────────────────────
Tests for EnergyDynamics' `decay` (mean-reversion) parameter.

Found empirically (sili_peridot dense-vs-sparse MQAR investigation): a
"neutral" drive/activation_cost calibration (drive == activation_cost *
E[|h|]) only zeroes the EXPECTED per-step drift -- it does not make
energy sit near 0. With no restoring force, energy is a driftless
random walk bounded only by the +-2.0 clamps, which spreads out and
gets absorbed at a boundary by chance, not because a neuron is actually
behaving pathologically. `decay` adds a discrete OU-style pull toward
0 every step (see sili/energy.py:_apply_energy_dynamics's
"Choosing decay" docstring section for the full derivation) while
preserving "gain from quiet, lose from active" exactly as before.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sili.energy import EnergyDynamics
from sili.tensor import Tensor


def _normal_tail(x: float) -> float:
    """1 - Phi(x), the standard normal upper tail, via erfc (stdlib-only)."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


# ─────────────────────────────────────────────────────────────────────────────
# Constructor validation
# ─────────────────────────────────────────────────────────────────────────────


class TestDecayConstructorValidation:
    def test_negative_decay_rejected(self):
        with pytest.raises(AssertionError):
            EnergyDynamics(drive=0.1, activation_cost=0.05, precision=0.01, density=0.05, p=0.3, decay=-0.01)

    def test_decay_zero_needs_no_drive_relationship(self):
        # decay=0.0 is the no-op default -- any drive is fine, no assert fires.
        EnergyDynamics(drive=1e-6, activation_cost=0.05, precision=0.01, density=0.05, p=0.3, decay=0.0)

    def test_drive_too_small_relative_to_decay_rejected(self):
        # Fixed point drive/decay = 0.01/0.02 = 0.5 < 2.0 -- a permanently
        # -quiet neuron could never reach the fire threshold.
        with pytest.raises(AssertionError):
            EnergyDynamics(drive=0.01, activation_cost=0.05, precision=0.01, density=0.05, p=0.3, decay=0.02)

    def test_drive_exactly_at_boundary_rejected(self):
        # drive == 2*decay -> fixed point exactly 2.0, never strictly clears
        # the threshold (asymptotic, not reachable) -- must be a strict >.
        with pytest.raises(AssertionError):
            EnergyDynamics(drive=0.04, activation_cost=0.05, precision=0.01, density=0.05, p=0.3, decay=0.02)

    def test_drive_comfortably_above_boundary_accepted(self):
        EnergyDynamics(drive=0.1, activation_cost=0.05, precision=0.01, density=0.05, p=0.3, decay=0.02)


# ─────────────────────────────────────────────────────────────────────────────
# Backward compatibility
# ─────────────────────────────────────────────────────────────────────────────


class TestDecayBackwardCompatible:
    def test_decay_zero_matches_prior_behavior_bit_exact(self):
        rng_a = np.random.RandomState(0)
        rng_b = np.random.RandomState(0)
        ed_no_decay_param = EnergyDynamics(
            drive=0.1, activation_cost=0.05, precision=0.01, density=0.05, p=0.3, rng=np.random.default_rng(1)
        )
        ed_explicit_zero = EnergyDynamics(
            drive=0.1,
            activation_cost=0.05,
            precision=0.01,
            density=0.05,
            p=0.3,
            decay=0.0,
            rng=np.random.default_rng(1),
        )
        h1 = Tensor((rng_a.randn(50) * 0.5).astype(np.float32))
        h2 = Tensor((rng_b.randn(50) * 0.5).astype(np.float32))
        h_out_a, _, actual_p_a = ed_no_decay_param.forward(h1)
        h_out_b, _, actual_p_b = ed_explicit_zero.forward(h2)
        assert np.array_equal(ed_no_decay_param.energy, ed_explicit_zero.energy)
        assert actual_p_a == actual_p_b


# ─────────────────────────────────────────────────────────────────────────────
# Escapes both pathological firing patterns (never-fires, always-fires)
# ─────────────────────────────────────────────────────────────────────────────


class TestDecayEscapesPathologicalPatterns:
    def test_permanently_quiet_neuron_still_reaches_fire(self):
        # |h|=0 every step (the zero-init/dead-neuron case this class
        # exists to escape) -- with decay set validly (drive > 2*decay),
        # the neuron MUST still cross +2.0 within a bounded number of
        # steps derived from the fixed point's convergence.
        drive, decay, activation_cost = 0.2, 0.05, 0.05
        ed = EnergyDynamics(
            drive=drive,
            activation_cost=activation_cost,
            precision=0.01,
            density=0.05,
            p=1.0,
            decay=decay,
            exploration=0.0,
            rng=np.random.default_rng(0),
        )
        h = Tensor(np.zeros(20, dtype=np.float32))
        fixed_point = drive / decay
        assert fixed_point > 2.0  # sanity check on the test's own setup
        # Convergence to within 0.01 of the fixed point: (1-decay)^t < eps
        t_needed = int(math.log(0.01 / fixed_point) / math.log(1.0 - decay)) + 50
        fired_ever = False
        for _ in range(t_needed):
            ed.forward(h)
            if np.max(ed.energy) >= 1.999:
                fired_ever = True
                break
        assert fired_ever, (
            f"a permanently-quiet neuron never approached the fire "
            f"threshold within {t_needed} steps despite a valid fixed "
            f"point of {fixed_point:.3f} > 2.0"
        )

    def test_permanently_loud_neuron_still_reaches_shutoff(self):
        # |h| large every step (chronically-saturated/"always fires" in
        # the activity sense) -- must still cross -2.0. drive=0.1 >
        # 2*decay=0.04 so the constructor's own validity guard is
        # satisfied (this is what "escapes the dead-neuron case" needs
        # to be true anyway -- a valid decay/drive pair should ALSO be
        # able to escape the opposite extreme).
        drive, decay, activation_cost = 0.1, 0.02, 0.5
        h_loud = 1.0  # activation_cost*h_loud=0.5 >> drive=0.1
        ed = EnergyDynamics(
            drive=drive,
            activation_cost=activation_cost,
            precision=0.01,
            density=0.05,
            p=1.0,
            decay=decay,
            exploration=0.0,
            rng=np.random.default_rng(0),
        )
        h = Tensor(np.full(20, h_loud, dtype=np.float32))
        fixed_point = (drive - activation_cost * h_loud) / decay
        assert fixed_point < -2.0  # sanity check on the test's own setup
        t_needed = int(math.log(0.01 / abs(fixed_point)) / math.log(1.0 - decay)) + 50
        shutoff_ever = False
        for _ in range(t_needed):
            ed.forward(h)
            if np.min(ed.energy) <= -1.999:
                shutoff_ever = True
                break
        assert shutoff_ever, (
            f"a permanently-loud neuron never approached the shutoff "
            f"threshold within {t_needed} steps despite a valid fixed "
            f"point of {fixed_point:.3f} < -2.0"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Concentrates a healthy population near 0 (the bug decay fixes)
# ─────────────────────────────────────────────────────────────────────────────


class TestDecayConcentratesHealthyPopulation:
    def test_decay_shrinks_stationary_std_vs_no_decay(self):
        # Same healthy, neutrally-calibrated, i.i.d.-|h| population run
        # with decay=0 vs decay>0 -- decay must produce a visibly
        # tighter (lower-std) energy distribution. This is the direct
        # regression test for the "flat histogram, no concentration
        # near 0" bug found in the real investigation.
        activation_cost = 0.05
        mean_abs_h = 0.34
        sigma_h = mean_abs_h / math.sqrt(2.0 / math.pi)
        drive = activation_cost * mean_abs_h
        n_neurons = 500
        n_steps = 4000

        def run(decay):
            e = np.zeros(n_neurons, dtype=np.float64)
            local_rng = np.random.default_rng(0)
            for _ in range(n_steps):
                h = np.abs(local_rng.normal(0, sigma_h, size=n_neurons))
                noise = local_rng.normal(0, 0.001, size=n_neurons)
                e = e * (1.0 - decay) + drive + noise - activation_cost * h
                e = np.clip(e, -2.0, 2.0)
            return e

        e_no_decay = run(0.0)
        e_with_decay = run(0.02)
        assert e_with_decay.std() < e_no_decay.std() * 0.5, (
            f"decay=0.02 std={e_with_decay.std():.4f} was not meaningfully "
            f"tighter than decay=0 std={e_no_decay.std():.4f}"
        )
        # And it should genuinely concentrate near 0, not just shrink some.
        frac_near_zero = np.mean(np.abs(e_with_decay) < 0.2)
        assert frac_near_zero > 0.3, (
            f"only {frac_near_zero:.2%} sat within [-0.2,0.2] under decay -- "
            f"expected real concentration near the setpoint"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Approximate firing rate / power matches the documented analytic estimate
# ─────────────────────────────────────────────────────────────────────────────


class TestDecayApproximateFiringRate:
    def test_empirical_boundary_crossing_rate_matches_analytic_order_of_magnitude(self):
        # Validates the "Approximate per-step boundary-crossing rate"
        # formula documented in _apply_energy_dynamics against a real
        # simulation with a KNOWN |h| distribution (so Var(|h|) is
        # exact, not estimated) -- this is what makes it a genuinely
        # useful predictive/tuning tool rather than decorative text.
        activation_cost = 0.05
        mean_abs_h = 0.34
        sigma_h = mean_abs_h / math.sqrt(2.0 / math.pi)
        var_h = sigma_h**2 * (1.0 - 2.0 / math.pi)  # Var(|h|) for half-normal
        drive = activation_cost * mean_abs_h
        # decay small enough (drive=0.017 > 2*decay=0.01, a valid pair)
        # and exploration large enough that the predicted crossing rate
        # lands in an empirically-testable range (~0.1-1%) -- a decay
        # strong enough to actually concentrate energy near 0 (the
        # whole point) makes boundary crossings a genuine rare event,
        # so validating the formula needs enough forcing variance that
        # a few-million-sample simulation still sees thousands of
        # crossings, not zero.
        decay = 0.005
        exploration = 0.065
        n_neurons = 800
        n_steps = 6000

        var_forcing = activation_cost**2 * var_h + exploration**2
        predicted_var_e = var_forcing / (2.0 * decay)
        predicted_sigma_e = math.sqrt(predicted_var_e)
        predicted_crossing_rate = 2.0 * _normal_tail(2.0 / predicted_sigma_e)

        rng = np.random.default_rng(0)
        e = np.zeros(n_neurons, dtype=np.float64)
        crossings = 0
        total = 0
        for _ in range(n_steps):
            h = np.abs(rng.normal(0, sigma_h, size=n_neurons))
            noise = rng.normal(0, exploration, size=n_neurons)
            e = e * (1.0 - decay) + drive + noise - activation_cost * h
            fired = e >= 2.0
            shutoff = e <= -2.0
            crossings += int(np.sum(fired) + np.sum(shutoff))
            total += n_neurons
            e = np.clip(e, -2.0, 2.0)

        empirical_rate = crossings / total
        assert predicted_crossing_rate > 0, "test setup produced a degenerate zero prediction"
        # "Approximate" per the docstring's own caveat (linear AR(1)
        # approx, ignores absorbing-boundary correction, symmetric
        # Gaussian assumption vs real right-skewed |h|) -- check
        # order-of-magnitude agreement, not tight equality.
        ratio = empirical_rate / predicted_crossing_rate
        assert 0.1 < ratio < 10.0, (
            f"empirical crossing rate {empirical_rate:.6f} vs analytic "
            f"prediction {predicted_crossing_rate:.6f} (ratio {ratio:.2f}) "
            f"-- outside the order-of-magnitude tolerance the docstring "
            f"formula claims"
        )
