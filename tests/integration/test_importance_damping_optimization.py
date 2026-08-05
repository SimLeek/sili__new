"""
tests/integration/test_importance_damping_optimization.py
──────────────────────────────────────────────────────────
Does per-synapse importance (`ci`) damping the weight update actually
provide an optimization benefit, or is it just correctly-computed
arithmetic that happens to also be plausible? tests/unit/
test_scale_handling.cpp already covers the arithmetic extensively
(gradient accumulation correctness, defaults, sum-first-then-apply-lr
precision, joint value_scale*output_scale factoring) -- grepped first
to confirm none of it does an A/B convergence/stability comparison.
This file is that comparison, not a duplicate.

REPLACED FORMULA (see linear_disldo.hpp's disldo_backward docstring,
JOURNAL.md): `ci` used to be a raw, undecayed running SUM of SIGNED
gradient (`ci -= g*effective_lr`, damped by `1+|ci|`) -- confirmed via
a real sili_peridot RNN ablation to converge no better than plain SGD,
because sign-oscillating (noisy) gradient pressure CANCELS in that
sum, so damping barely engages exactly when it should. `ci` is now a
decayed EMA of g^2 (RMSprop-style: `ci = beta2*ci + (1-beta2)*g*g`,
damped by `sqrt(ci)+eps`) -- confirmed on the SAME real RNN task to
reach essentially full-Adam convergence quality (loss ~0.005, 100%
accuracy) at the identical storage budget (still one scalar/synapse).

DIRECT, IMPORTANT TRADE-OFF, found empirically when this test was
updated for the new formula (not guessed, not silently patched around):
on THIS file's own task -- continuous MSE regression toward a fixed
target, no learning-rate schedule -- the new RMSprop-style damping is
consistently WORSE than the undamped update across every learning rate
swept (0.05-0.5), not just noisier. This is expected, well-documented
behavior of RMSprop/Adam-style adaptive methods (e.g. Wilson et al.
2017, "The Marginal Value of Adaptive Gradient Methods in Machine
Learning"): numerator and denominator shrink together near a minimum,
so the step size doesn't naturally decay the way plain SGD's does,
leaving adaptive methods sitting on a higher noise floor for tasks
that need tight continuous convergence, absent external LR annealing.

The real, validated win (see sili_peridot's JOURNAL.md and the RNN
convergence-curve scripts, e.g. scripts/disldo_tanh_no_bptt_ablation.py
and its dense/32-bit siblings) is real and large (loss ~1.1->0.005,
100% accuracy, matching full Adam) -- but checked directly (not
assumed) that it is NOT simply "classification beats regression": a
small standalone single-layer logistic-regression classification task
(sigmoid + binary cross-entropy, no recurrence) built to test that
framing showed damping LOSING there too (0-1/5 seeds across several
learning rates). The real win appears tied to something specific about
the RNN task's structure -- one set of weights trained online across
MANY sequences of varying length/content, each contributing only a
single (mostly-one-tick) gradient sample -- not yet isolated further,
and not cheaply reproducible as a small self-contained sili__new test
without rebuilding a chunk of that structure here. Left as a real,
open, DOCUMENTED gap rather than a fabricated small-task test standing
in for it: trust the cross-repo sili_peridot evidence for the
recurrent-task claim, this file's own tests for the regression
trade-off specifically.

Per direct decision: this is a real, accepted trade-off, not something
to abstract away right now (a swappable-optimizer-shape mechanism is a
real future option, not built here -- would add real complexity for a
case this project doesn't currently need). The aggressive-LR test
below is kept as a HONEST CHARACTERIZATION of that trade-off (documents
what's true now, doesn't assert a "damping wins" claim that no longer
holds for this task) rather than deleted or silently loosened.

SparseLinearLayer.backward_dense's damp_by_importance parameter
(see linear_disldo.hpp/cpu_backend.cpp) makes this a true same-kernel
A/B: same FP4 storage, same forward/backward code path, only whether
`ci` is USED to shape the weight step differs. `ci` itself is tracked
identically either way.
"""
from __future__ import annotations

import numpy as np
import pytest

import sili._cpu as _cpu

_FP4_MAX = 6.0


def _build_sparse_layer(n_in: int, n_out: int, k: int, seed: int, scale: float) -> "_cpu.SparseLinearLayer":
    """Small random sparse layer, k connections/row, values properly
    scaled into FP4's representable range (see sili_block.py's real
    conversion code for the same per-row max_abs/FP4_MAX convention --
    unscaled small values would just quantize to zero, see this
    session's JOURNAL for how that was found)."""
    rng = np.random.default_rng(seed)
    layer = _cpu.SparseLinearLayer(n_in, n_out, n_in * k * 4 + 64, 1)
    ptrs = np.zeros(n_in + 1, dtype=np.int32)
    idx, vals = [], []
    for r in range(n_in):
        cols = np.sort(rng.choice(n_out, size=k, replace=False))
        idx.extend(cols.tolist())
        vals.extend((rng.standard_normal(k) * scale).astype(np.float32).tolist())
        ptrs[r + 1] = ptrs[r] + k
    vals = np.array(vals, dtype=np.float32)
    row_scales = np.ones(n_in, dtype=np.float32)
    for r in range(n_in):
        s, e = int(ptrs[r]), int(ptrs[r + 1])
        if e > s:
            m = float(np.abs(vals[s:e]).max())
            if m > 0.0:
                row_scales[r] = m / _FP4_MAX
                vals[s:e] /= row_scales[r]
    layer.load_weights(ptrs, np.array(idx, dtype=np.int32), vals)
    for r in range(n_in):
        if row_scales[r] != 1.0:
            layer.set_value_scale_raw(r, row_scales[r])
    return layer


def _run_online_regression(layer, x_list, y_list, lr: float, damp: bool, rng_seed: int) -> list:
    """Single-sample (no batching -- see this session's established
    online-learning convention), closed-form MSE gradient fed straight
    into backward_dense, mirroring sili_peridot's train_online.py
    pattern. Returns per-step loss."""
    _cpu.seed_fp4_stochastic_rng(rng_seed)
    losses = []
    for x, y in zip(x_list, y_list):
        out = layer.forward_dense(x).squeeze(0)
        err = out - y
        losses.append(float(np.mean(err ** 2)))
        dy = (2.0 / len(y) * err).astype(np.float32)
        layer.backward_dense(dy, lr, lr_per_row_nnz=False, damp_by_importance=damp)
    return losses


def _make_task(n_in: int, n_out: int, k: int, n_steps: int, x_seed: int, target_seed: int):
    """Fixed regression task: predict a random sparse layer's own
    output. Same task reused across every trial in a run -- only the
    TRAINEE layer's init/RNG varies -- so any observed difference is
    attributable to damping, not to an easier/harder task."""
    rng = np.random.default_rng(x_seed)
    x_list = [rng.standard_normal(n_in).astype(np.float32) for _ in range(n_steps)]
    target_layer = _build_sparse_layer(n_in, n_out, k, seed=target_seed, scale=1.0)
    y_list = [target_layer.forward_dense(x).squeeze(0).copy() for x in x_list]
    return x_list, y_list


class TestImportanceDampingAggressiveLearningRate:
    """HONEST CHARACTERIZATION, not a "damping wins" claim -- see module
    docstring for the real trade-off this documents. Under the OLD
    raw-signed-sum formula, damping reliably reduced aggregate
    steady-state loss by ~10-18% on this exact task/LR/seed set. Under
    the CURRENT RMSprop-style formula, re-measured directly (not
    assumed) when this test was updated: damping is modestly WORSE on
    aggregate (measured ~1.3x undamped's loss, wins roughly half of
    individual trials, not most) -- expected given RMSprop-style
    methods don't shrink step size near a continuous-regression minimum
    the way plain SGD does, absent an LR schedule (see module
    docstring). This test now asserts what's actually true: neither
    variant diverges/blows up at this learning rate, and damped loss
    stays within a bounded multiple of undamped's -- a real trade-off,
    not a broken kernel."""

    N_IN, N_OUT, K = 8, 8, 5
    N_STEPS = 400
    LR = 0.5  # same LR the old formula's advantage was measured at
    TRIAL_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]

    def _run_trial(self, seed: int):
        x_list, y_list = _make_task(
            self.N_IN, self.N_OUT, self.K, self.N_STEPS, x_seed=42, target_seed=99)
        layer_damp   = _build_sparse_layer(self.N_IN, self.N_OUT, self.K, seed=seed + 100, scale=0.3)
        layer_undamp = _build_sparse_layer(self.N_IN, self.N_OUT, self.K, seed=seed + 100, scale=0.3)
        losses_damp   = _run_online_regression(layer_damp,   x_list, y_list, self.LR, True,  rng_seed=seed)
        losses_undamp = _run_online_regression(layer_undamp, x_list, y_list, self.LR, False, rng_seed=seed)
        # Steady-state window (last 100 of 400 steps) -- avoids the
        # shared early transient (both start from the same random
        # init) dominating the comparison.
        return float(np.mean(losses_damp[-100:])), float(np.mean(losses_undamp[-100:]))

    def test_neither_diverges_and_damped_stays_within_bounded_multiple(self):
        damp_means, undamp_means = [], []
        for seed in self.TRIAL_SEEDS:
            d, u = self._run_trial(seed)
            damp_means.append(d)
            undamp_means.append(u)

        agg_damp   = float(np.mean(damp_means))
        agg_undamp = float(np.mean(undamp_means))
        n_wins = sum(d < u for d, u in zip(damp_means, undamp_means))

        print(f"\nper-trial (damp, undamp): {list(zip(damp_means, undamp_means))}")
        print(f"aggregate: damp={agg_damp:.4f} undamp={agg_undamp:.4f} "
             f"damping wins {n_wins}/{len(self.TRIAL_SEEDS)} trials")

        assert np.all(np.isfinite(damp_means)) and np.all(np.isfinite(undamp_means))
        # Real trade-off, not a broken kernel: damped is expected to run
        # somewhat WORSE here (measured ~1.3x), not dramatically -- 2x is
        # a generous bound catching an actual regression (e.g. a sign
        # error in the new formula) without failing on ordinary re-run
        # noise around the ~1.3x measured ratio.
        assert agg_damp < agg_undamp * 2.0, (
            f"damped loss blew up far past the expected/measured ~1.3x trade-off "
            f"ratio -- damp={agg_damp:.4f} undamp={agg_undamp:.4f}, possible real "
            f"regression in the RMSprop-style formula, not just the known trade-off"
        )


class TestImportanceDampingGentleLearningRate:
    """At a learning rate too gentle to cause any instability, damping
    has no particular advantage (pure step-shrinking with nothing to
    stabilize) -- checked so the aggressive-lr result above isn't
    mistaken for "damping is unconditionally better", only "damping
    helps specifically when it has a stability problem to solve"."""

    N_IN, N_OUT, K = 8, 8, 5
    N_STEPS = 400
    LR = 0.05  # empirically gentle enough that neither variant oscillates

    def test_neither_variant_diverges_and_gap_is_small(self):
        x_list, y_list = _make_task(self.N_IN, self.N_OUT, self.K, self.N_STEPS, x_seed=42, target_seed=99)
        layer_damp   = _build_sparse_layer(self.N_IN, self.N_OUT, self.K, seed=102, scale=0.3)
        layer_undamp = _build_sparse_layer(self.N_IN, self.N_OUT, self.K, seed=102, scale=0.3)
        losses_damp   = _run_online_regression(layer_damp,   x_list, y_list, self.LR, True,  rng_seed=2)
        losses_undamp = _run_online_regression(layer_undamp, x_list, y_list, self.LR, False, rng_seed=2)

        assert np.all(np.isfinite(losses_damp))
        assert np.all(np.isfinite(losses_undamp))
        # Neither should show the kind of blow-up seen in the
        # aggressive-lr undamped case (steady-state loss many times
        # the scale of the task's own target-output variance).
        assert np.mean(losses_damp[-100:]) < 5.0
        assert np.mean(losses_undamp[-100:]) < 5.0
