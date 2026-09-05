"""Homeostatic energy dynamics for sili neural networks.

energy is plain numpy — running state, not a learned parameter. The
caller owns batch iteration; do not pass batched tensors here. See
PROOFS.md for the full mathematical treatment, CITATIONS.md (repo
root) for external work referenced by name below, and
docs/research/energy.rst for the design-tradeoff research narrative.
"""

from __future__ import annotations

import numpy as np

from sili.module import Module
from sili.tensor import Tensor, gather

# Core function


def _apply_energy_dynamics(
    h: Tensor,
    energy: np.ndarray,
    drive: float,
    activation_cost: float,
    precision: float,
    density: float,
    exploration: float = 0.001,
    setpoint: float = 1.0,
    activation_threshold: float = 1e-4,
    reactivity: float = 0.01,
    p: float = 0.05,
    fire_reset_to_zero: bool = False,
    fire_cost: float | None = None,
    fire_wake_gradient: float | None = None,
    wake_sign: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    wake_gate_steps: int | None = None,
    steps_since_fired: np.ndarray | None = None,
) -> tuple[Tensor, np.ndarray, Tensor, float, np.ndarray, np.ndarray]:
    """Apply continuous energy dynamics, returning an updated Tensor in
    the autograd graph. See docs/research/energy.rst for the full design
    rationale behind every non-obvious parameter below (anchors named in
    parens).

    Parameters
    ----------
    h : hidden state Tensor, any shape. energy : per-neuron energy,
    plain numpy, same shape as h. drive : baseline energy drift
    (metabolic tempo). activation_cost : energy drain per unit |h|.
    precision : KL sparsity strength (lambda_kl), population-level loop
    (``apply_energy_dynamics.precision_vs_reactivity``). density :
    target density (beta), must stay below p
    (``apply_energy_dynamics.p_vs_density``). exploration : per-neuron
    noise std, must stay < drive/2, breaks top-p-tie symmetry. setpoint
    : comfort zone target (tau). activation_threshold : dead zone
    threshold. reactivity : per-neuron correction gain (alpha)
    (``apply_energy_dynamics.precision_vs_reactivity``). p : hard
    ceiling on active fraction, a compute limit, not a sparsity target
    (``apply_energy_dynamics.p_vs_density``). fire_reset_to_zero :
    opt-in, e<-0.0 on fire instead of refractory drain
    (``apply_energy_dynamics.fire_reset_to_zero``), exclusive with
    fire_cost. fire_cost : opt-in fixed fire-drain amount. fire_wake_gradient
    : opt-in guaranteed-magnitude gradient at fired positions
    (``apply_energy_dynamics.fire_wake_gradient``). wake_sign : required
    if fire_wake_gradient set, precomputed +-1 array. rng : opt-in
    seeded source for exploration noise (``apply_energy_dynamics.rng_seeding_bug``).
    wake_gate_steps : opt-in recency threshold gating the WHOLE
    mechanism (``apply_energy_dynamics.wake_gate_steps``).
    stagger_wake_init : opt-in, randomizes initial steps_since_fired
    (``apply_energy_dynamics.stagger_wake_init``). steps_since_fired :
    caller-owned recency counter, required if wake_gate_steps is set.

    Returns
    -------
    h_out, new_energy, aux_loss (see
    ``apply_energy_dynamics.aux_loss_zero_init_escape``), actual_p,
    kept_indices (see ``apply_energy_dynamics.kept_indices``),
    new_steps_since_fired.
    """

    original_shape = h.shape
    n = int(np.prod(original_shape))
    dtype = np.float32

    # Pull h into numpy for all mask / gate decisions.
    # CPU backend: h.data is already numpy.
    h_np = np.asarray(h.data, dtype=dtype).ravel()
    energy_flat = np.asarray(energy, dtype=dtype).ravel().copy()

    # ── 1. Dead zone ──────────────────────────────────────────────────
    alive = np.abs(h_np) > activation_threshold
    h_dz = h_np * alive

    # ── 2. Energy update ─────────────────────────────────────────────
    noise_src = rng if rng is not None else np.random
    noise = noise_src.normal(0.0, exploration, size=(n,)).astype(dtype)

    # steps_since_fired defaults to all-zeros (everyone "awake") so a
    # fresh EnergyDynamics with wake_gate_steps set starts masked-off,
    # not universally stale. See
    # docs/research/energy.rst:apply_energy_dynamics.wake_gate_steps.
    ssf_flat = (
        np.asarray(steps_since_fired, dtype=np.int64).ravel()
        if steps_since_fired is not None
        else np.zeros(n, dtype=np.int64)
    )
    stale = (ssf_flat >= wake_gate_steps) if wake_gate_steps is not None else np.ones(n, dtype=bool)
    new_energy = energy_flat + drive + noise - activation_cost * np.abs(h_dz)

    # ── 3. Hard thresholds (integrate-and-fire) ──────────────────────
    # Shutoff resolved first -- frees budget slots before fire claims
    # them. See PROOFS.md Theorems 2 and 6, and
    # docs/research/energy.rst:apply_energy_dynamics.wake_gate_steps
    # for why `stale` gates the whole mechanism below, not just drive.
    fire_mask = (new_energy >= 2.0) & stale
    shutoff_mask = (new_energy <= -2.0) & stale

    shutoff_values = np.zeros(n, dtype=dtype)
    if shutoff_mask.any():
        shutoff_values[shutoff_mask] = (energy_flat + 2.0)[shutoff_mask]
        new_energy[shutoff_mask] = -2.0

    if fire_mask.any():
        if fire_reset_to_zero:
            new_energy[fire_mask] = 0.0
        elif fire_cost is not None:
            new_energy[fire_mask] -= float(fire_cost)
        else:
            new_energy[fire_mask] -= 2.0 * activation_cost  # refractory drain (default, unchanged)

    # Awake neurons' energy stays exactly at its input value -- frozen,
    # not even subject to drive/noise/drain, until they go stale.
    new_energy = np.where(stale, new_energy, energy_flat)

    # ── 4. Hard-ceiling top_p gate ────────────────────────────────────
    # p is a HARD CEILING -- see
    # docs/research/energy.rst:apply_energy_dynamics.p_vs_density.
    # Priority: highest-energy fired neurons, then highest-|h| others.
    # Suppressed fired neurons keep elevated energy, queue for next step.
    n_stale = int(stale.sum())
    k = max(1, round(p * n_stale)) if n_stale > 0 else 0

    fire_idx = np.where(fire_mask)[0]
    non_fire_idx = np.where(stale & ~fire_mask)[0]
    n_fired = len(fire_idx)

    if n_fired >= k:
        top_order = np.argpartition(new_energy[fire_idx], -k)[-k:]
        kept_fire = fire_idx[top_order]
        kept_nfire = np.empty(0, dtype=int)
    else:
        kept_fire = fire_idx
        remaining = k - n_fired
        if remaining > 0 and len(non_fire_idx) > 0:
            fill_k = min(remaining, len(non_fire_idx))
            scores = np.abs(h_dz[non_fire_idx])
            top_local = np.argpartition(scores, -fill_k)[-fill_k:]
            kept_nfire = non_fire_idx[top_local]
        else:
            kept_nfire = np.empty(0, dtype=int)

    # Build differentiable h_out: normal kept -> h*gate (gradient flows);
    # fired/shutoff kept -> additive const (threshold event, not h value);
    # suppressed -> zero. Gate is a straight-through estimator; _coerce
    # handles numpy operands.

    shutoff_in_kept = (
        np.intersect1d(kept_nfire, np.where(shutoff_mask)[0]) if len(kept_nfire) > 0 else np.empty(0, dtype=int)
    )
    normal_kept = np.setdiff1d(kept_nfire, shutoff_in_kept)
    normal_kept = normal_kept[alive[normal_kept]]

    gate_np = np.zeros(n, dtype=dtype)
    if len(normal_kept) > 0:
        gate_np[normal_kept] = 1.0
    gate_np[~stale] = 1.0  # awake neurons: pure passthrough, h_out=h

    const_np = np.zeros(n, dtype=dtype)
    if len(kept_fire) > 0:
        const_np[kept_fire] = 2.0
    if len(shutoff_in_kept) > 0:
        const_np[shutoff_in_kept] = shutoff_values[shutoff_in_kept]

    h_out = h * gate_np.reshape(original_shape) + const_np.reshape(original_shape)

    # ── 6. aux_loss as a Tensor ────────────────────────────────────────
    # Setting loss = aux_loss for every layer is sufficient for training.
    # See docs/research/energy.rst:apply_energy_dynamics.aux_loss_zero_init_escape
    # for the full proof and the two independent zero-init-escape paths.
    n_active = (
        len(normal_kept)
        + len(kept_fire)
        + (
            int(np.sum(np.abs(const_np.ravel()[shutoff_in_kept]) > activation_threshold))
            if len(shutoff_in_kept) > 0
            else 0
        )
    )
    actual_p = float(n_active / n_stale) if n_stale > 0 else 0.0

    rho = float(np.clip(actual_p, 1e-5, 1.0 - 1e-5))
    kl_val = float(precision * (rho * np.log(rho / density) + (1.0 - rho) * np.log((1.0 - rho) / (1.0 - density))))

    # Loss tensor uses h directly (linear surrogate), not |h| -- removes
    # the sign(h) factor that kills gradients at zero init. Restricted to
    # STALE positions via gather() (true topological exclusion), NOT
    # `expr * stale_mask` -- a plain multiplicative mask does NOT protect
    # against inf*0.0==nan contamination from an overflowed awake-position
    # term. See docs/research/energy.rst:apply_energy_dynamics.gather_not_mask.
    stale_idx = np.where(stale)[0]
    if len(stale_idx) > 0:
        c_stale_np = (energy_flat + drive + noise)[stale_idx]
        c_stale_t = Tensor(c_stale_np.astype(dtype), backend=h.backend)
        h_stale = gather(h, stale_idx)
        new_energy_t_stale = c_stale_t - activation_cost * abs(h_stale)
        energy_loss = (reactivity / 2.0) * ((new_energy_t_stale - setpoint) ** 2).sum()
    else:
        energy_loss = Tensor(np.float32(0.0), backend=h.backend)
    aux_loss = kl_val + energy_loss  # float + Tensor -> Tensor via _coerce

    # Guaranteed-magnitude gradient at kept-fired positions -- see
    # docs/research/energy.rst:apply_energy_dynamics.fire_wake_gradient.
    if fire_wake_gradient is not None and len(kept_fire) > 0:
        assert wake_sign is not None, "wake_sign is required when fire_wake_gradient is set"
        wake_sign_flat = np.asarray(wake_sign, dtype=dtype).ravel()
        wake_gate_np = np.zeros(n, dtype=dtype)
        wake_gate_np[kept_fire] = fire_wake_gradient * wake_sign_flat[kept_fire]
        wake_gate_t = Tensor(wake_gate_np.reshape(original_shape), backend=h.backend)
        aux_loss = aux_loss + (h * wake_gate_t).sum()

    # kept_indices: the gate decision energy already made. See
    # docs/research/energy.rst:apply_energy_dynamics.kept_indices.
    kept_indices = np.sort(
        np.concatenate(
            [
                normal_kept,
                kept_fire,
                shutoff_in_kept,
            ]
        )
    ).astype(np.int32)

    # Recency counter for wake_gate_steps -- reset wherever a neuron
    # actually fired-and-was-kept this call, incremented everywhere else.
    new_ssf = ssf_flat + 1
    if len(kept_fire) > 0:
        new_ssf[kept_fire] = 0
    new_steps_since_fired = new_ssf.reshape(energy.shape)

    return (h_out, new_energy.reshape(energy.shape), aux_loss, actual_p, kept_indices, new_steps_since_fired)


# Module wrapper


class EnergyDynamics(Module):
    """
    Per-region homeostatic energy dynamics.

    energy is plain numpy (running state, not a learned parameter).
    parameters() returns [] — nothing here is trained by the optimizer.

    forward() returns (h_out, aux_loss, actual_p).
    Add aux_loss to any task loss before calling backward(), or call
    aux_loss.backward() directly when it is the only signal.

    Example
    -------
    >>> ed = EnergyDynamics(drive=0.15, activation_cost=0.08,
    ...                     precision=0.04, density=0.25)
    >>> h_out, aux_loss, actual_p = ed(h)
    >>> (task_loss(h_out) + aux_loss).backward()
    """

    def __init__(
        self,
        drive: float,
        activation_cost: float,
        precision: float,
        density: float,
        exploration: float = 0.001,
        setpoint: float = 1.0,
        activation_threshold: float = 1e-4,
        reactivity: float = 0.01,
        p: float = 0.05,
        fire_reset_to_zero: bool = False,
        fire_cost: float | None = None,
        fire_wake_gradient: float | None = None,
        wake_seed: int = 0,
        rng: np.random.Generator | None = None,
        wake_gate_steps: int | None = None,
        stagger_wake_init: bool = False,
    ):
        """Same parameters as _apply_energy_dynamics (drive=delta,
        activation_cost=gamma, precision=lambda_kl, density=beta,
        exploration=sigma, setpoint=tau, reactivity=alpha) -- see that
        function's docstring for full semantics."""
        assert np.finfo(np.float32).eps * 2 <= activation_cost <= 4.0, (
            "activation_cost (gamma) must be positive and <= 4.0"
        )
        assert 0.0 < density < 1.0, "density (beta) must be in (0, 1)"
        assert 0.0 < p <= 1.0, "p must be in (0, 1]"
        # See docs/research/energy.rst:apply_energy_dynamics.p_vs_density.
        assert density <= p * 0.8, (
            f"density ({density}) must stay comfortably below p ({p}) -- "
            f"p is a hardware/telemetry compute-limit ceiling that should "
            f"rarely bind, not the thing that shapes learned sparsity. Got "
            f"density > p*0.8 ({p * 0.8}); e.g. p=0.05 with density=0.01 is "
            f"the intended relationship, not p=density or p<density."
        )

        self._energy_start = max(
            0.0, 2.0 - drive * 10
        )  # allow 10 steps for noise, but don't wait forever for more noise

        self.drive = float(drive)
        self.activation_cost = float(activation_cost)
        self.precision = float(precision)
        self.density = float(density)
        self.exploration = float(exploration)
        self.setpoint = float(setpoint)
        self.activation_threshold = float(activation_threshold)
        self.reactivity = float(reactivity)
        self.p = float(p)
        self.fire_reset_to_zero = bool(fire_reset_to_zero)
        self.fire_cost = None if fire_cost is None else float(fire_cost)
        self.fire_wake_gradient = None if fire_wake_gradient is None else float(fire_wake_gradient)
        self.wake_seed = int(wake_seed)
        self.rng = rng
        self.wake_gate_steps = None if wake_gate_steps is None else int(wake_gate_steps)
        self.stagger_wake_init = bool(stagger_wake_init)

        # Running state -- numpy, not a Tensor, not a learned parameter
        self.energy: np.ndarray | None = None
        self.steps_since_fired: np.ndarray | None = None
        # Seeded (not global RNG), lazily built on first forward().
        self._wake_sign: np.ndarray | None = None

        # Cached for inspection / logging
        self.aux_loss: Tensor | None = None
        self.actual_p: float = 0.0
        # See docs/research/energy.rst:apply_energy_dynamics.kept_indices.
        self.kept_indices: np.ndarray | None = None

    def parameters(self) -> list:
        return []

    def state_dict(self) -> dict:
        return {
            "energy": (
                np.array(self.energy, dtype=np.float32) if self.energy is not None else np.zeros(0, dtype=np.float32)
            ),
        }

    def load_state_dict(self, d: dict):
        e = d["energy"]
        self.energy = e.copy() if e.size > 0 else None

    def forward(self, h: Tensor, density_override: float | None = None) -> tuple[Tensor, Tensor, float]:
        """
        Parameters
        ----------
        h : Tensor, any shape -- no batch dimension, caller iterates batches.
        density_override : used in place of self.density for this call
            only, e.g. a dynamic KL target derived from a measured
            branching ratio (see BranchingRatioTracker). None (default)
            reproduces the fixed-density behavior of every existing caller.

        Returns
        -------
        h_out : updated Tensor, in autograd graph.
        aux_loss : Tensor scalar -- add to task loss or .backward() directly.
        actual_p : achieved active fraction -- feed to PFC / thermal / battery.
        """
        if self.energy is None or self.energy.shape != h.shape:
            # Reset energy on shape change (e.g. body switch, region resize)
            self.energy = np.ones(h.shape, dtype=np.float32) * self._energy_start
            self._wake_sign = None
            if self.wake_gate_steps is not None and self.stagger_wake_init:
                if self.rng is not None:
                    self.steps_since_fired = self.rng.integers(0, self.wake_gate_steps, size=h.shape).astype(np.int64)
                else:
                    self.steps_since_fired = (
                        np.random.RandomState(self.wake_seed)
                        .randint(0, self.wake_gate_steps, size=h.shape)
                        .astype(np.int64)
                    )
            else:
                self.steps_since_fired = np.zeros(h.shape, dtype=np.int64)

        if self.fire_wake_gradient is not None and self._wake_sign is None:
            n = int(np.prod(h.shape))
            self._wake_sign = np.random.RandomState(self.wake_seed).choice([-1.0, 1.0], size=n).astype(np.float32)

        density = self.density if density_override is None else float(density_override)

        (h_out, self.energy, self.aux_loss, self.actual_p, self.kept_indices, self.steps_since_fired) = (
            _apply_energy_dynamics(
                h,
                self.energy,
                self.drive,
                self.activation_cost,
                self.precision,
                density,
                self.exploration,
                self.setpoint,
                self.activation_threshold,
                self.reactivity,
                self.p,
                self.fire_reset_to_zero,
                self.fire_cost,
                self.fire_wake_gradient,
                self._wake_sign,
                self.rng,
                self.wake_gate_steps,
                self.steps_since_fired,
            )
        )
        return h_out, self.aux_loss, self.actual_p


class BranchingRatioTracker:
    """Estimate the branching ratio m of a self-propagating activity
    process from a scalar activity count fed in once per step. Feed it
    recurrent(state)'s OWN activity, not combined activity (see
    SparseRNNCell.forward). See
    docs/research/energy.rst:branching_ratio_tracker.model for the
    branching-with-immigration model, the identifiability rationale,
    and the OLS estimator's relationship to the MR estimator.
    """

    def __init__(self, window: int = 200):
        assert window >= 3, "window must be >= 3 -- OLS needs at least a few points"
        self.window = int(window)
        self._history: list = []

    def update(self, activity: float) -> None:
        """Record one step's activity count (e.g. count of |h| above the
        region's activation_threshold)."""
        self._history.append(float(activity))
        if len(self._history) > self.window:
            self._history.pop(0)

    def branching_ratio(self) -> float | None:
        """OLS slope of a[t+1] on a[t] over the current window, or None if
        there isn't enough history yet or activity has had zero variance
        (a flat sequence carries no information about self-propagation)."""
        if len(self._history) < 3:
            return None
        a = np.asarray(self._history[:-1], dtype=np.float64)
        b = np.asarray(self._history[1:], dtype=np.float64)
        a_mean = a.mean()
        denom = float(np.sum((a - a_mean) ** 2))
        if denom <= 1e-12:
            return None
        b_mean = b.mean()
        m = float(np.sum((a - a_mean) * (b - b_mean)) / denom)
        return m

    def avalanche_sizes(self) -> list:
        """Consecutive-nonzero-activity run lengths -- see
        docs/research/energy.rst:branching_ratio_tracker.model for why a
        power-law tail here is the falsifiable SOC signature. Log/plot
        externally; this just extracts the runs."""
        sizes, current = [], 0
        for a in self._history:
            if a > 0:
                current += 1
            elif current > 0:
                sizes.append(current)
                current = 0
        if current > 0:
            sizes.append(current)
        return sizes

    def reset(self) -> None:
        self._history.clear()


class EMABranchingRatioTracker:
    """EMA-based streaming variant of BranchingRatioTracker -- same
    OLS-slope estimator, but regression statistics are exponentially-
    weighted running estimates instead of a hard sliding window: O(1)
    memory instead of O(window). Does NOT implement avalanche_sizes()
    (needs a retained raw sequence an EMA state can't reconstruct) --
    run a BranchingRatioTracker alongside if that's needed. See
    docs/research/energy.rst:branching_ratio_tracker.model for the
    `alpha` vs. `window` tradeoff and the underlying model.
    """

    def __init__(self, alpha: float = 0.05):
        assert 0.0 < alpha <= 1.0, "alpha must be in (0, 1]"
        self.alpha = float(alpha)
        self._prev: float | None = None
        self._mean_a: float | None = None
        self._mean_b: float | None = None
        self._mean_aa: float | None = None
        self._mean_ab: float | None = None
        self._n_pairs = 0

    def update(self, activity: float) -> None:
        """Record one step's activity count (e.g. count of |h| above the
        region's activation_threshold)."""
        activity = float(activity)
        if self._prev is not None:
            a, b = self._prev, activity
            a2, ab = a * a, a * b
            if self._mean_a is None:
                # First pair -- seed the running means directly rather
                # than blending against an undefined prior.
                self._mean_a, self._mean_b = a, b
                self._mean_aa, self._mean_ab = a2, ab
            else:
                al = self.alpha
                self._mean_a = (1.0 - al) * self._mean_a + al * a
                self._mean_b = (1.0 - al) * self._mean_b + al * b
                self._mean_aa = (1.0 - al) * self._mean_aa + al * a2
                self._mean_ab = (1.0 - al) * self._mean_ab + al * ab
            self._n_pairs += 1
        self._prev = activity

    def branching_ratio(self) -> float | None:
        """OLS slope implied by the current EMA statistics, or None if
        there aren't at least two pairs yet or activity has had zero
        variance under the EMA (mirrors BranchingRatioTracker's guards)."""
        if self._n_pairs < 2 or self._mean_a is None:
            return None
        var = self._mean_aa - self._mean_a**2
        if var <= 1e-12:
            return None
        cov = self._mean_ab - self._mean_a * self._mean_b
        return float(cov / var)

    def reset(self) -> None:
        self._prev = None
        self._mean_a = self._mean_b = self._mean_aa = self._mean_ab = None
        self._n_pairs = 0


# Column-averaging loss


def column_averaging_loss(h_out: Tensor, target: Tensor, n_folds: int, weight: float = 1.0, indices=None) -> Tensor:
    """weight * mean_i( (mean_t(column_i[t]) - target[i])^2 ), where
    column_i is the set of n_folds neurons tracking input index i. Takes
    h_out (energy-gated), not the pre-gating h, so a column member
    EnergyDynamics suppresses hurts this loss rather than being
    invisible to it. Combine with EnergyDynamics's aux_loss via
    sili.tensor.combine_losses.

    indices: optional, len n_folds*input_size, selects which flat
    positions of a larger h_out form the column block. None (default)
    requires h_out to be exactly that size.
    """
    input_size = target.shape[0]
    expected = n_folds * input_size
    if indices is None:
        assert h_out.shape[0] == expected, (
            f"h_out has {h_out.shape[0]} elements, expected n_folds*input_size "
            f"= {n_folds}*{input_size} = {expected} (or pass indices to "
            f"select a subset of a larger state)"
        )
        column_block = h_out
    else:
        indices = np.asarray(indices)
        assert indices.shape == (expected,), f"indices has shape {indices.shape}, expected ({expected},)"
        column_block = gather(h_out, indices)

    col_mean = column_block.reshape((n_folds, input_size)).sum(axis=0) * (1.0 / n_folds)
    diff = col_mean - target
    return (diff**2).sum() * (weight / input_size)
