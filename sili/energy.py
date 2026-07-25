"""
energy_dynamics.py
──────────────────
Homeostatic energy dynamics for sili neural networks.

h is kept in the autograd graph throughout via straight-through gating:
  - Normal kept neurons:  h * gate        gradient flows through normally
  - Fired neurons:        constant 2.0    no gradient through h_out for this
                                          path — aux_loss reaches h directly
                                          via new_energy_t = c - gamma*h (step 6)
  - Shutoff neurons:      constant (e+2)  same as fired — aux_loss path active
  - Suppressed neurons:   zero            no gradient

energy is plain numpy — running state, not a learned parameter.
The caller owns batch iteration; do not pass batched tensors here.

See PROOFS.md for the full mathematical treatment, and CITATIONS.md (repo
root) for the external work referenced by name in this file's docstrings.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

from sili.module import Module
from sili.tensor import Tensor


# ══════════════════════════════════════════════════════════════════════════════
#  Core function
# ══════════════════════════════════════════════════════════════════════════════

def _apply_energy_dynamics(
        h: Tensor,
        energy: np.ndarray,
        drive: float,                # delta  — baseline energy accumulation rate
        activation_cost: float,      # gamma  — energy drain per unit |h|
        precision: float,            # lambda_kl — sparsity enforcement strength
        density: float,              # beta   — target activation density
        exploration: float = 0.001,  # sigma  — per-neuron energy noise std
        setpoint: float = 1.0,       # tau    — homeostatic comfort zone target
        activation_threshold: float = 1e-4,  # dead zone threshold (architectural)
        reactivity: float = 0.01,    # alpha  — homeostatic correction gain
        p: float = 0.05,             # HARD CEILING on active neuron fraction
) -> Tuple[Tensor, np.ndarray, Tensor, float, np.ndarray]:
    """
    Apply continuous energy dynamics, returning an updated Tensor in the
    autograd graph.

    Parameters
    ----------
    h             : hidden state Tensor, any shape
    energy        : per-neuron energy, plain numpy, same shape as h
    drive         : baseline energy drift — sets metabolic tempo
    activation_cost: energy drain per unit |h| — neural efficiency
    precision     : KL sparsity enforcement strength (lambda_kl). Population
                    -level control loop: pushes the ACHIEVED active fraction
                    (rho, a population statistic) toward density (beta) via
                    a KL term with no gradient to h (discrete mask) — this is
                    a "how many are active" loop, blind to which neurons.
                    Distinct from reactivity below; the two are not
                    interchangeable knobs on the same thing.
    density       : target activation density / sparsity setpoint (beta) —
                    the actual LEARNED sparsity should usually land below p
                    on its own, driven down by this term (plus shutoff/forced
                    -firing) competing against whatever task objective is
                    training the network — see p's docstring for why this
                    must stay clearly below p, not close to or above it.
    exploration   : per-neuron energy noise std.
                    Must stay < drive/2 during waking — crossing this boundary
                    enters the hallucination / REM regime (intentional in DREAM,
                    forbidden during waking operation). Also, independent of
                    that regime boundary: exploration is what breaks symmetry
                    between otherwise-identical neurons (same drive, same
                    activation_cost, same initial energy) — without it,
                    ties in the top-p competition are resolved by array order
                    rather than by any signal, so every "identical" neuron
                    would fire in lockstep instead of the population
                    differentiating over time.
    setpoint      : homeostatic comfort zone target
    activation_threshold : activation dead zone — values at or below this are
                    zeroed for energy-accounting purposes (named for what it
                    protects — the boundary between "active" and "not" — not
                    for the KL mechanism, which is a separate, population
                    -level pressure that happens to also consume this value).
    reactivity    : homeostatic correction gain (alpha). Per-neuron control
                    loop: pushes each neuron's own new_energy_t toward
                    setpoint (tau) via a quadratic loss WITH a gradient to h
                    (the linear surrogate) — a "how comfortable is this
                    specific neuron" loop. See precision above for the
                    contrasting population-level loop; both land in the same
                    aux_loss scalar but are not the same mechanism and don't
                    substitute for each other.
    p             : HARD CEILING on active neuron fraction — a hardware/
                    telemetry-driven compute-limit ceiling (thermal, battery,
                    update-rate), NOT a tuning knob for learning quality and
                    NOT the thing that should shape learned sparsity. Never
                    exceeded under any condition — including fire events and
                    pain signals. Exceeding p risks GPU thermal runaway,
                    battery overdraw, and update-rate collapse (30-60hz ->
                    2-5hz) that causes physical instability in motor-control
                    regions. Set jointly by:
                      - PFC (cognitive load target)
                      - GPU/CPU temperature monitors (thermal throttle)
                      - Battery level + distance to charger (power reserve)
                    MUST sit clearly above density (roughly 5-10x, e.g.
                    p=0.05 with density=0.01) so KL/shutoff/forced-firing —
                    not this hard ceiling — are what the network actually
                    has to satisfy while also chasing its task objective.
                    Set p this low (or lower) only when genuinely resource
                    -constrained; see EnergyDynamics.__init__'s
                    `density <= p * 0.8` assertion, which exists because this
                    relationship was previously (silently) inverted.

    Returns
    -------
    h_out      : updated Tensor, same shape, still in autograd graph
    new_energy : updated energy array, same shape as input energy
    aux_loss   : Tensor scalar. Call .backward() directly when this is the only
                 signal (early training, DREAM mode), or add to task loss first.
                 Gradient flows to h via energy_loss -> new_energy_t -> h
                 (linear surrogate, no abs — see step 6 for full proof).
    actual_p   : fraction of neurons active after all gating. May differ
                 slightly from p at small region sizes due to integer rounding.
                 Return to PFC / thermal / battery for closed-loop feedback.
    kept_indices : int32 array, flat (raveled) indices into h's original
                 shape of every position gating kept alive (normal + fired +
                 shutoff-but-kept), sorted ascending. This IS the gate
                 decision energy already made — the intended source for
                 building a sparse (CSR) representation of h for the next
                 consumer, using h's PRE-gating values at these indices
                 (not h_out's post-gating fire/shutoff constants), rather
                 than re-deriving sparsity via an independent top-k pass
                 that could disagree with what energy actually decided.
    """

    b              = h.backend
    original_shape = h.shape
    n              = int(np.prod(original_shape))
    dtype          = np.float32

    # Pull h into numpy for all mask / gate decisions.
    # CPU backend: h.data is already numpy.
    h_np        = np.asarray(h.data, dtype=dtype).ravel()
    energy_flat = np.asarray(energy, dtype=dtype).ravel().copy()

    # ── 1. Dead zone ─────────────────────────────────────────────────────────
    alive = np.abs(h_np) > activation_threshold
    h_dz  = h_np * alive           # zeroed copy for energy computation

    # ── 2. Energy update ─────────────────────────────────────────────────────
    # drive     — deterministic drift (every neuron, every step)
    # noise     — decorrelated per-neuron stochastic variation
    # |h_dz|   — active representation drains energy
    # Constraint: exploration must stay below drive/2 during waking
    #             to remain in curiosity regime, not hallucination regime.
    noise      = np.random.normal(0.0, exploration, size=(n,)).astype(dtype)
    new_energy = energy_flat + drive + noise - activation_cost * np.abs(h_dz)

    # ── 3. Hard thresholds (integrate-and-fire) ──────────────────────────────
    # Shutoff resolved first — frees budget slots before fire claims them.
    # See PROOFS.md Theorems 2 and 6.
    fire_mask    = new_energy >= 2.0
    shutoff_mask = new_energy <= -2.0

    shutoff_values = np.zeros(n, dtype=dtype)
    if shutoff_mask.any():
        shutoff_values[shutoff_mask] = (energy_flat + 2.0)[shutoff_mask]
        new_energy[shutoff_mask]     = -2.0

    if fire_mask.any():
        new_energy[fire_mask] -= 2.0 * activation_cost   # refractory drain

    # ── 4. Hard-ceiling top_p gate ───────────────────────────────────────────
    # p is a HARD CEILING — no condition may exceed it.
    # Priority: highest-energy fired neurons first, then highest-|h| others.
    # Suppressed fired neurons keep elevated energy and queue for next step.
    k = max(1, int(round(p * n)))

    fire_idx     = np.where(fire_mask)[0]
    non_fire_idx = np.where(~fire_mask)[0]
    n_fired      = len(fire_idx)

    if n_fired >= k:
        top_order  = np.argpartition(new_energy[fire_idx], -k)[-k:]
        kept_fire  = fire_idx[top_order]
        kept_nfire = np.empty(0, dtype=int)
    else:
        kept_fire = fire_idx
        remaining = k - n_fired
        if remaining > 0 and len(non_fire_idx) > 0:
            fill_k     = min(remaining, len(non_fire_idx))
            scores     = np.abs(h_dz[non_fire_idx])
            top_local  = np.argpartition(scores, -fill_k)[-fill_k:]
            kept_nfire = non_fire_idx[top_local]
        else:
            kept_nfire = np.empty(0, dtype=int)

    # ── 5. Build differentiable h_out ────────────────────────────────────────
    #
    # Normal kept  ->  h * gate        gradient flows through
    # Fired        ->  + 2.0 const     constant: threshold event, not h value
    # Shutoff      ->  + (e+2) const   constant: threshold event, not h value
    # Suppressed   ->  zero            no contribution
    #
    # Gate is a straight-through estimator over the kept positions.
    # _coerce in Tensor handles numpy arrays so operators work directly.

    shutoff_in_kept = (
        np.intersect1d(kept_nfire, np.where(shutoff_mask)[0])
        if len(kept_nfire) > 0 else np.empty(0, dtype=int)
    )
    normal_kept = np.setdiff1d(kept_nfire, shutoff_in_kept)
    normal_kept = normal_kept[alive[normal_kept]]

    gate_np = np.zeros(n, dtype=dtype)
    if len(normal_kept) > 0:
        gate_np[normal_kept] = 1.0

    const_np = np.zeros(n, dtype=dtype)
    if len(kept_fire) > 0:
        const_np[kept_fire] = 2.0
    if len(shutoff_in_kept) > 0:
        const_np[shutoff_in_kept] = shutoff_values[shutoff_in_kept]

    h_out = h * gate_np.reshape(original_shape) + const_np.reshape(original_shape)

    # ── 6. aux_loss as a Tensor ───────────────────────────────────────────────
    #
    # Setting loss = aux_loss for every layer is sufficient for training.
    # Proof for the chain  x ->[ED]-> x_out ->[W]-> y ->[ED]:
    #
    #   c_j          = e_j + drive + noise_j       (independent of h)
    #   new_energy_t = c_j - activation_cost * y_j (linear surrogate)
    #   energy_loss  = (alpha/2) * sum((new_energy_t_j - tau)^2)
    #
    #   d(loss)/d(W_ji) = alpha * (new_energy_t_j - tau) * (-gamma) * x_out_i
    #
    # Zero-init escape — two independent mechanisms:
    #   1. Each layer's own aux_loss: d(loss)/d(h) = alpha*(c-gamma*h-tau)*(-gamma)
    #      c ~ drive at init, so (drive-tau) ~ -0.85 != 0 from step 1 onward.
    #      This reaches upstream weights immediately, before any firing.
    #   2. Fire-together-wire-together: once neurons fire (~2.0/drive ~ 14 steps),
    #      x_out_i = 2.0 makes d(aux_loss_y)/d(W_ji) non-zero for the NEXT layer.
    #      These two mechanisms are complementary, not the same path.
    #
    # Zero-loss target is new_energy_t = tau (the homeostatic setpoint).
    # Push direction and magnitude depend on running energy history — not
    # just sign(h) — producing genuine homeostatic drive rather than simple
    # weight decay. In RNNs/agents the network can learn to take actions that
    # bring future inputs into the comfort zone.
    #
    # Learning rule: x_out_i fires AND y_j needs more drain -> W_ji grows.
    # (fire together, wire together)
    #
    # Physical energy update still uses np.abs(h_dz) — only the gradient
    # path changes. kl_loss has no h-gradient (discrete mask); added as a
    # plain float, _coerce promotes it when summed with energy_loss.

    n_active = len(normal_kept) + len(kept_fire) + (
        int(np.sum(np.abs(const_np.ravel()[shutoff_in_kept]) > activation_threshold))
        if len(shutoff_in_kept) > 0 else 0
    )
    actual_p = float(n_active / n)

    rho    = float(np.clip(actual_p, 1e-5, 1.0 - 1e-5))
    kl_val = float(precision * (
        rho * np.log(rho / density) +
        (1.0 - rho) * np.log((1.0 - rho) / (1.0 - density))
    ))

    # Physical energy uses |h| (always positive drain, unchanged).
    # Loss tensor uses h directly (linear surrogate) — removes the sign(h)
    # factor that kills gradients at zero init.
    # Quadratic form targets exactly setpoint (tau) for zero loss:
    #
    #   L_j = (alpha/2) * (new_energy_t_j - tau)^2
    #
    #   d(L)/d(W_ji) = alpha * (new_energy_t_j - tau) * (-gamma) * x_out_i
    #
    # At zero init: new_energy_t_j ~ drive, so (drive - tau) is a non-zero
    # constant (e.g. 0.15 - 1.0 = -0.85), giving non-zero gradients immediately.
    # The zero-crossing is at new_energy_t = tau — the homeostatic setpoint.
    # Push direction depends on history (running energy) and upstream activity,
    # not just on the sign of h, which produces richer dynamics in RNNs/agents.

    # Human note: this actually pulls e to t, not |e| to t. To pull |e| to t, we would need
    # energy_loss = (reactivity / 2.0) * (new_energy_t.abs() - setpoint).pow(2).sum()
    # but that's complicated. Doing that would allow neurons at e=t to fire rapidly
    # and go to e=-t quickly, but there doesn't seem to be much actual benefit to that.

    c_np = (energy_flat + drive + noise).reshape(original_shape)
    c_t = Tensor(c_np.astype(dtype), backend=h.backend)
    # Abs backward in sili is defined in every backend with `grad[a == 0.0] = 1.0`
    # in pytorch, you would use `h_abs = torch.where(h == 0, h, torch.abs(h))` and then use h_abs instead of abs(h)
    new_energy_t = c_t - activation_cost * abs(h)
    energy_loss  = (reactivity / 2.0) * ((new_energy_t - setpoint)**2).sum()
    aux_loss     = kl_val + energy_loss     # float + Tensor -> Tensor via _coerce

    # kept_indices: the gate decision energy already made, exposed so a
    # caller can build a CSR straight from it (indices here, values from
    # PRE-gating h_np at these same positions) instead of re-deriving
    # sparsity via an independent top-k pass that could disagree with what
    # energy actually decided. Sorted ascending -- CSR requires sorted
    # per-row indices.
    kept_indices = np.sort(np.concatenate([
        normal_kept, kept_fire, shutoff_in_kept,
    ])).astype(np.int32)

    return h_out, new_energy.reshape(energy.shape), aux_loss, actual_p, kept_indices


# ══════════════════════════════════════════════════════════════════════════════
#  Module wrapper
# ══════════════════════════════════════════════════════════════════════════════

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
            drive: float,                # delta  — metabolic tempo
            activation_cost: float,      # gamma  — neural efficiency
            precision: float,            # lambda_kl — sparsity enforcement
            density: float,              # beta   — target activation density
            exploration: float = 0.001,  # sigma  — per-neuron noise
            setpoint: float   = 1.0,     # tau    — comfort zone target
            activation_threshold: float = 1e-4,  # dead zone threshold
            reactivity: float = 0.01,    # alpha  — homeostatic gain
            p: float          = 0.05,    # hard ceiling on active fraction
    ):
        assert 0.01 <= activation_cost <= 0.5, "activation_cost (gamma) must be in [0.01, 0.5]"
        assert 0.0  <  density         < 1.0,  "density (beta) must be in (0, 1)"
        assert 0.0  <  p               <= 1.0, "p must be in (0, 1]"
        # p is a hardware/telemetry compute-limit ceiling, not a sparsity
        # target -- it must sit clearly above density so KL/shutoff/forced
        # -firing (not this hard ceiling) are what actually shapes learned
        # sparsity. This relationship was previously (silently) invertible;
        # this assert exists specifically to catch that, not just document it.
        assert density <= p * 0.8, (
            f"density ({density}) must stay comfortably below p ({p}) -- "
            f"p is a hardware/telemetry compute-limit ceiling that should "
            f"rarely bind, not the thing that shapes learned sparsity. Got "
            f"density > p*0.8 ({p * 0.8}); e.g. p=0.05 with density=0.01 is "
            f"the intended relationship, not p=density or p<density."
        )

        self._energy_start = max(0.0,2.0-drive*10)  # allow 10 steps for noise, but don't wait forever for more noise

        self.drive                = float(drive)
        self.activation_cost      = float(activation_cost)
        self.precision            = float(precision)
        self.density              = float(density)
        self.exploration          = float(exploration)
        self.setpoint             = float(setpoint)
        self.activation_threshold = float(activation_threshold)
        self.reactivity           = float(reactivity)
        self.p                    = float(p)

        # Running state — numpy, not a Tensor, not a learned parameter
        self.energy: Optional[np.ndarray] = None

        # Cached for inspection / logging
        self.aux_loss: Optional[Tensor] = None
        self.actual_p: float = 0.0
        # The gate decision from the most recent forward() -- flat indices
        # into h's shape, sorted ascending. See _apply_energy_dynamics's
        # kept_indices docstring: the intended source for building a CSR of
        # h for a downstream consumer (indices here, values from that
        # forward's PRE-gating h), rather than a separate independent top-k
        # pass that could disagree with this decision.
        self.kept_indices: Optional[np.ndarray] = None

    def parameters(self) -> list:
        return []

    def state_dict(self) -> dict:
        return {
            "energy": (np.array(self.energy, dtype=np.float32)
                       if self.energy is not None
                       else np.zeros(0, dtype=np.float32)),
        }

    def load_state_dict(self, d: dict):
        e = d["energy"]
        self.energy = e.copy() if e.size > 0 else None

    def forward(self, h: Tensor, density_override: Optional[float] = None) -> Tuple[Tensor, Tensor, float]:
        """
        Parameters
        ----------
        h : Tensor, any shape — no batch dimension, caller iterates batches
        density_override : if given, used in place of self.density for this
                    call only (self.density is left unchanged). Intended for
                    a caller-computed dynamic KL target -- e.g. derived from
                    a measured recurrent-only branching ratio (see
                    sili.energy.BranchingRatioTracker) rather than a fixed
                    config value. EnergyDynamics itself has no opinion on how
                    this value is computed; passing None (the default)
                    reproduces the exact fixed-density behavior of every
                    existing caller.

        Returns
        -------
        h_out    : updated Tensor, in autograd graph
        aux_loss : Tensor scalar — add to task loss or .backward() directly
        actual_p : achieved active fraction — feed to PFC / thermal / battery
        """
        if self.energy is None or self.energy.shape != h.shape:
            # Reset energy on shape change (e.g. body switch, region resize)
            self.energy = np.ones(h.shape, dtype=np.float32)*self._energy_start

        density = self.density if density_override is None else float(density_override)

        h_out, self.energy, self.aux_loss, self.actual_p, self.kept_indices = _apply_energy_dynamics(
            h, self.energy,
            self.drive, self.activation_cost, self.precision, density,
            self.exploration, self.setpoint, self.activation_threshold, self.reactivity, self.p,
        )
        return h_out, self.aux_loss, self.actual_p


# ══════════════════════════════════════════════════════════════════════════════
#  BranchingRatioTracker
# ══════════════════════════════════════════════════════════════════════════════

class BranchingRatioTracker:
    """
    Estimate the branching ratio m of a self-propagating activity process
    from a scalar activity count fed in once per step.

    Background: a[t+1] = m*a[t] + h models activity as a branching process
    with immigration -- m is the self-propagation (recurrent) factor, h an
    external drive. m in [0.97, 0.99] ("near-critical") is the intended
    healthy operating band for a genuinely self-sustaining recurrent
    pathway; m near 0 means activity is being carried entirely by fresh
    external drive each step, with no real recurrent memory.

    Identifiability note (why this exists as its own tracker, not a stat
    computed on SparseRNNCell's combined h): if fed the COMBINED activity
    of input_proj(obs) + recurrent(state), this estimate cannot distinguish
    "the recurrent pathway genuinely self-propagates" from "fresh input
    alone keeps activity in-band while the recurrent branching factor is
    silently 0" -- both produce statistically identical activity sequences
    once mixed. Feed this tracker recurrent(state)'s OWN activity, measured
    BEFORE it's summed with input_proj(obs)'s contribution, to get a
    meaningful answer. See SparseRNNCell.forward for the split measurement
    this is meant to be used with.

    Estimator: single-lag OLS slope of a[t+1] on a[t] over a sliding window
    -- a simplified, single-lag version of the multistep-regression (MR)
    estimator (Wilting & Priesemann) that method is otherwise named after;
    this implementation does not do the multi-lag extrapolation MR uses to
    correct for subsampling bias, which matters more at large scale than it
    does for the per-region window sizes this is intended for. Treat
    branching_ratio() as a useful trend/regime indicator, not a
    publication-grade MR estimate.
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

    def branching_ratio(self) -> Optional[float]:
        """OLS slope of a[t+1] on a[t] over the current window, or None if
        there isn't enough history yet or activity has had zero variance
        (a flat sequence carries no information about self-propagation)."""
        if len(self._history) < 3:
            return None
        a = np.asarray(self._history[:-1], dtype=np.float64)
        b = np.asarray(self._history[1:],  dtype=np.float64)
        a_mean = a.mean()
        denom  = float(np.sum((a - a_mean) ** 2))
        if denom <= 1e-12:
            return None
        b_mean = b.mean()
        m = float(np.sum((a - a_mean) * (b - b_mean)) / denom)
        return m

    def avalanche_sizes(self) -> list:
        """Consecutive-nonzero-activity run lengths from the current
        history window -- a power-law-tail in this distribution is the
        actual falsifiable signature of self-organized criticality (Bak,
        Tang & Wiesenfeld 1987; Beggs & Plenz 2003), independent of whether
        branching_ratio() alone looks healthy. Log/plot externally; this
        just extracts the runs."""
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
    """
    EMA-based streaming variant of BranchingRatioTracker -- same OLS-slope
    estimator (see that class's docstring for the branching-with-
    immigration model and the identifiability reason this needs
    recurrent-only activity, not combined activity), but its regression
    statistics (mean(a), mean(a_next), mean(a*a_next), mean(a^2)) are
    exponentially-weighted running estimates instead of a hard sliding
    window. O(1) memory (no history buffer) instead of O(window).

    The `alpha` decay rate is the direct tradeoff BranchingRatioTracker's
    `window` only offers indirectly: large alpha = fast response to a
    changing regime but noisy/short effective memory; small alpha = smooth
    long-term trend but slow to notice a real regime shift. Unlike a hard
    window, older observations are smoothly discounted rather than
    dropped outright at a fixed cutoff. Want BOTH a fast and a long-term
    read at once? Run two instances side by side at different alphas
    (e.g. alpha=0.2 "fast" + alpha=0.02 "slow") rather than looking for a
    single-instance dual-timescale mode -- composition over a wider
    single-class API, and it keeps each instance's semantics simple.

    Does NOT implement avalanche_sizes() -- that check inherently needs a
    retained sequence of raw activity values (consecutive-nonzero run
    lengths), which an O(1) EMA state cannot reconstruct by construction.
    Use BranchingRatioTracker (run one alongside, if wanted) for that
    check -- this is exactly the "have both EMA and windowed" case, not a
    reason to avoid EMA for the branching-ratio estimate itself.
    """

    def __init__(self, alpha: float = 0.05):
        assert 0.0 < alpha <= 1.0, "alpha must be in (0, 1]"
        self.alpha = float(alpha)
        self._prev:    Optional[float] = None
        self._mean_a:  Optional[float] = None
        self._mean_b:  Optional[float] = None
        self._mean_aa: Optional[float] = None
        self._mean_ab: Optional[float] = None
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
                self._mean_a, self._mean_b   = a, b
                self._mean_aa, self._mean_ab = a2, ab
            else:
                al = self.alpha
                self._mean_a  = (1.0 - al) * self._mean_a  + al * a
                self._mean_b  = (1.0 - al) * self._mean_b  + al * b
                self._mean_aa = (1.0 - al) * self._mean_aa + al * a2
                self._mean_ab = (1.0 - al) * self._mean_ab + al * ab
            self._n_pairs += 1
        self._prev = activity

    def branching_ratio(self) -> Optional[float]:
        """OLS slope implied by the current EMA statistics, or None if
        there aren't at least two pairs yet or activity has had zero
        variance under the EMA (mirrors BranchingRatioTracker's guards)."""
        if self._n_pairs < 2 or self._mean_a is None:
            return None
        var = self._mean_aa - self._mean_a ** 2
        if var <= 1e-12:
            return None
        cov = self._mean_ab - self._mean_a * self._mean_b
        return float(cov / var)

    def reset(self) -> None:
        self._prev = None
        self._mean_a = self._mean_b = self._mean_aa = self._mean_ab = None
        self._n_pairs = 0