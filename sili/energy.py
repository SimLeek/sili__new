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

See PROOFS.md for the full mathematical treatment.
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
        kl_eps: float = 1e-4,        # dead zone threshold (architectural)
        reactivity: float = 0.01,    # alpha  — homeostatic correction gain
        p: float = 0.02,             # HARD CEILING on active neuron fraction
) -> Tuple[Tensor, np.ndarray, Tensor, float]:
    """
    Apply continuous energy dynamics, returning an updated Tensor in the
    autograd graph.

    Parameters
    ----------
    h             : hidden state Tensor, any shape
    energy        : per-neuron energy, plain numpy, same shape as h
    drive         : baseline energy drift — sets metabolic tempo
    activation_cost: energy drain per unit |h| — neural efficiency
    precision     : KL sparsity enforcement strength
    density       : target activation density / sparsity setpoint
    exploration   : per-neuron energy noise std.
                    Must stay < drive/2 during waking — crossing this boundary
                    enters the hallucination / REM regime (intentional in DREAM,
                    forbidden during waking operation).
    setpoint      : homeostatic comfort zone target
    kl_eps        : activation dead zone — values at or below this are zeroed
    reactivity    : homeostatic correction gain
    p             : HARD CEILING on active neuron fraction. Never exceeded under
                    any condition — including fire events and pain signals.
                    Exceeding p risks GPU thermal runaway, battery overdraw, and
                    update-rate collapse (30-60hz -> 2-5hz) that causes physical
                    instability in motor-control regions.
                    Set jointly by:
                      - PFC (cognitive load target)
                      - GPU/CPU temperature monitors (thermal throttle)
                      - Battery level + distance to charger (power reserve)

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
    alive = np.abs(h_np) > kl_eps
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
        int(np.sum(np.abs(const_np.ravel()[shutoff_in_kept]) > kl_eps))
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

    return h_out, new_energy.reshape(energy.shape), aux_loss, actual_p


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
            kl_eps: float     = 1e-4,    # dead zone threshold
            reactivity: float = 0.01,    # alpha  — homeostatic gain
            p: float          = 0.02,    # hard ceiling on active fraction
    ):
        assert 0.01 <= activation_cost <= 0.5, "activation_cost (gamma) must be in [0.01, 0.5]"
        assert 0.0  <  density         < 1.0,  "density (beta) must be in (0, 1)"
        assert 0.0  <  p               <= 1.0, "p must be in (0, 1]"

        self._energy_start = max(0.0,2.0-drive*10)  # allow 10 steps for noise, but don't wait forever for more noise

        self.drive           = float(drive)
        self.activation_cost = float(activation_cost)
        self.precision       = float(precision)
        self.density         = float(density)
        self.exploration     = float(exploration)
        self.setpoint        = float(setpoint)
        self.kl_eps          = float(kl_eps)
        self.reactivity      = float(reactivity)
        self.p               = float(p)

        # Running state — numpy, not a Tensor, not a learned parameter
        self.energy: Optional[np.ndarray] = None

        # Cached for inspection / logging
        self.aux_loss: Optional[Tensor] = None
        self.actual_p: float = 0.0

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

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor, float]:
        """
        Parameters
        ----------
        h : Tensor, any shape — no batch dimension, caller iterates batches

        Returns
        -------
        h_out    : updated Tensor, in autograd graph
        aux_loss : Tensor scalar — add to task loss or .backward() directly
        actual_p : achieved active fraction — feed to PFC / thermal / battery
        """
        if self.energy is None or self.energy.shape != h.shape:
            # Reset energy on shape change (e.g. body switch, region resize)
            self.energy = np.ones(h.shape, dtype=np.float32)*self._energy_start

        h_out, self.energy, self.aux_loss, self.actual_p = _apply_energy_dynamics(
            h, self.energy,
            self.drive, self.activation_cost, self.precision, self.density,
            self.exploration, self.setpoint, self.kl_eps, self.reactivity, self.p,
        )
        return h_out, self.aux_loss, self.actual_p