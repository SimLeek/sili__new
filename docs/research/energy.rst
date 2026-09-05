``energy.py`` research notes
==============================

Background for ``sili/energy.py``'s homeostatic energy dynamics. This
document holds the investigation narrative, measured numbers, and
design-tradeoff reasoning that used to live inline in the module's
docstrings; the source file keeps concise functional docs (parameter
types, brief purpose) and points here for the "why." See ``PROOFS.md``
for the full mathematical treatment and ``CITATIONS.md`` (repo root)
for external work referenced by name below.

``h`` is kept in the autograd graph throughout via straight-through
gating:

- Normal kept neurons: ``h * gate`` -- gradient flows through normally.
- Fired neurons: constant ``2.0`` -- no gradient through ``h_out`` on
  this path; ``aux_loss`` reaches ``h`` directly via
  ``new_energy_t = c - gamma*h`` (see ``aux_loss.zero_init_escape``
  below).
- Shutoff neurons: constant ``(e+2)`` -- same as fired, ``aux_loss``
  path active.
- Suppressed neurons: zero -- no gradient.

``energy`` is plain numpy -- running state, not a learned parameter.
The caller owns batch iteration; this module does not accept batched
tensors.

.. _apply_energy_dynamics.precision_vs_reactivity:

``precision`` (population loop) vs. ``reactivity`` (per-neuron loop)
-----------------------------------------------------------------------

*ID:* ``apply_energy_dynamics.precision_vs_reactivity``

``precision`` (lambda_kl) is a population-level control loop: it pushes
the ACHIEVED active fraction (rho, a population statistic) toward
``density`` (beta) via a KL term with no gradient to ``h`` (the mask is
discrete) -- a "how many are active" loop, blind to which specific
neurons.

``reactivity`` (alpha) is a per-neuron control loop: it pushes each
neuron's own ``new_energy_t`` toward ``setpoint`` (tau) via a quadratic
loss WITH a gradient to ``h`` (the linear surrogate) -- a "how
comfortable is this specific neuron" loop.

Both land in the same ``aux_loss`` scalar but are not the same
mechanism and don't substitute for each other.

.. _apply_energy_dynamics.p_vs_density:

``p`` (hard ceiling) vs. ``density`` (target) -- why they must stay far apart
--------------------------------------------------------------------------------

*ID:* ``apply_energy_dynamics.p_vs_density``

``p`` is a HARD CEILING on active neuron fraction -- a hardware/
telemetry-driven compute-limit ceiling (thermal, battery, update-rate),
NOT a tuning knob for learning quality and NOT the thing that should
shape learned sparsity. Never exceeded under any condition, including
fire events and pain signals. Exceeding ``p`` risks GPU thermal
runaway, battery overdraw, and update-rate collapse (30-60hz ->
2-5hz) that causes physical instability in motor-control regions. Set
jointly by PFC (cognitive load target), GPU/CPU temperature monitors
(thermal throttle), and battery level + distance to charger (power
reserve).

``density`` is the target activation density / sparsity setpoint
(beta) -- the actual LEARNED sparsity should usually land below ``p``
on its own, driven down by ``precision`` (plus shutoff/forced-firing)
competing against whatever task objective is training the network.

``p`` MUST sit clearly above ``density`` (roughly 5-10x, e.g.
``p=0.05`` with ``density=0.01``) so KL/shutoff/forced-firing -- not
the hard ceiling -- are what the network actually has to satisfy while
also chasing its task objective. This relationship was previously
(silently) invertible; ``EnergyDynamics.__init__``'s
``density <= p * 0.8`` assertion exists specifically to catch that, not
just document it.

.. _apply_energy_dynamics.fire_reset_to_zero:

``fire_reset_to_zero``: a hard reset beats a small refractory drain
------------------------------------------------------------------------

*ID:* ``apply_energy_dynamics.fire_reset_to_zero``

Opt-in, default False (preserves existing behavior exactly). When
True, a fire event sets ``e <- 0.0`` instead of the default
``e <- e - 2*gamma`` refractory drain.

The default drain is tiny relative to ``drive`` whenever the opposing
continuous term (``gamma*|h|``) is small (e.g. a cold, near-zero-init
population) -- the population can climb back to threshold and re-fire
within a handful of steps, degenerating from "rare recovery event"
into near-continuous forced-firing that saturates the downstream
output to a near-constant value regardless of real input. Firing is a
discrete, metabolically distinct event from normal continuous
operation (biologically: an action potential), so a hard reset is a
more direct model than a small linear drain.

Does NOT weaken Theorem 6(a) (Omega = {|e|<=2} positively invariant,
``energy-proofs.md``) -- that proof only requires the post-fire update
to strictly reduce ``|e|``, which ``e<-0`` satisfies unconditionally.
Mutually exclusive with ``fire_cost``.

.. _apply_energy_dynamics.fire_wake_gradient:

``fire_wake_gradient``: a guaranteed-magnitude gradient, not a soft pressure
--------------------------------------------------------------------------------

*ID:* ``apply_energy_dynamics.fire_wake_gradient``

Adds ``fire_wake_gradient * wake_sign * h`` at KEPT-FIRED positions
only. Its gradient is exactly ``fire_wake_gradient * wake_sign`` there
regardless of ``h``'s value -- a hard, deterministic magnitude, unlike
``energy_loss``'s gradient (which depends on
``new_energy_t - setpoint``). Deliberately separate from
``energy_loss``: ``(h * wake_gate).sum()`` backprops exactly
``wake_gate`` into ``h`` regardless of ``h``'s actual value.

Caller must size this to clear whatever downstream quantization floor
applies -- this module has no visibility into layer-specific scale, so
tune empirically against a direct before/after weight measurement, not
by derivation.

``wake_sign`` is a required, precomputed +-1 array, same length as
``h.ravel()``, fixed per position so fired neurons don't all get
pushed the same direction every time. ``EnergyDynamics.forward``
derives this once from a fixed seed (``wake_seed``) and reuses it;
direct callers of ``_apply_energy_dynamics`` must supply their own.

.. _apply_energy_dynamics.rng_seeding_bug:

Real bug: the exploration noise draw had no seeding path at all
----------------------------------------------------------------------

*ID:* ``apply_energy_dynamics.rng_seeding_bug``

The ``rng`` parameter is opt-in (default None, preserves exact
existing behavior: draws from bare global numpy RNG state). Before it
existed, the exploration noise draw had no way to be seeded at all --
confirmed directly this makes energy-enabled runs genuinely
non-reproducible run-to-run even with an otherwise identical config
(task RNG, FP4 rounding RNG, and everything else seeded), since
whatever consumed global numpy state earlier in the process shifts
this draw. Pass a seeded ``np.random.Generator`` (matching
``DISLDOLayer``'s own ``rng=`` convention) for reproducible
comparisons.

.. code-block:: python

   # as of PR (energy.py, this docs pass):
   noise_src = rng if rng is not None else np.random
   noise = noise_src.normal(0.0, exploration, size=(n,)).astype(dtype)

.. _apply_energy_dynamics.wake_gate_steps:

``wake_gate_steps``: the whole mechanism must be gated, not just drive
-----------------------------------------------------------------------

*ID:* ``apply_energy_dynamics.wake_gate_steps``

Opt-in, default None (preserves exact existing behavior). Gates the
WHOLE per-neuron mechanism, not just ``drive``. Neurons whose
``steps_since_fired`` has reached or passed this many calls without
firing ("stale") get the full computation -- drive, noise, drain,
fire/shutoff eligibility, the top-p competition, and ``energy_loss``'s
gradient pressure -- exactly as if ``wake_gate_steps`` were None.
Neurons that HAVE fired within the last ``wake_gate_steps`` calls
("awake") are excluded from ALL of that entirely: their energy stays
frozen at its input value, they never enter fire/shutoff or the top-p
competition, they contribute nothing to ``energy_loss``, and
``h_out=h`` for them exactly -- as if they were never passed through
``EnergyDynamics`` this call.

Distinct from the population-level ``p``/``density`` knobs (which
shape the aggregate active fraction, not which specific neurons) and
from ``fire_wake_gradient`` (a one-off gradient at fire time, not an
ongoing state change). Proposed to address the classic dead-neuron
problem generally (not just zero-init specifically): a neuron that
hasn't fired in a long time keeps getting pushed toward threshold,
while a recently-active one is left alone entirely to settle.

**Two earlier, narrower versions were tried and rejected**: (1) a
multiplicative BOOST on drive for stale neurons (``drive * multiplier
> 1``), and (2) masking ONLY drive to 0 for awake neurons while
leaving noise, drain, thresholds, and ``energy_loss``'s gradient
pressure active for them. BOTH confirmed directly to cause the same
sustained-divergence failure (grad_norm climbing to NaN by roughly
step 1600) at every tested gate value -- masking drive alone isn't
enough; the ENTIRE mechanism must be gated. Do not reintroduce either
narrower version without re-verifying stability.

.. code-block:: python

   # as of PR (energy.py, this docs pass) -- the gate computation:
   stale = (ssf_flat >= wake_gate_steps) if wake_gate_steps is not None else np.ones(n, dtype=bool)
   new_energy = energy_flat + drive + noise - activation_cost * np.abs(h_dz)
   # ... fire/shutoff resolved only within `stale` ...
   new_energy = np.where(stale, new_energy, energy_flat)  # awake: energy frozen
   # WRONG (the two rejected versions): scaling/masking drive alone, e.g.
   #   awake_drive_scale = 0.0 if not stale else 1.0
   #   new_energy = energy_flat + drive * awake_drive_scale + noise - activation_cost * np.abs(h_dz)
   # -- still leaves noise/drain/thresholds/energy_loss active for "awake"
   # neurons, reproducing the same NaN-by-step-1600 divergence.

.. _apply_energy_dynamics.stagger_wake_init:

``stagger_wake_init``: avoiding simultaneous first-eligibility
---------------------------------------------------------------

*ID:* ``apply_energy_dynamics.stagger_wake_init``

Opt-in, default False. Only meaningful with ``wake_gate_steps`` set.
``steps_since_fired`` resets to all-zeros on every shape change (see
``EnergyDynamics.forward``) -- with nothing having fired yet (e.g.
all-zero-init), every neuron reaches "stale" on the exact same step
(``step == wake_gate_steps``), so the whole population becomes
fire-eligible simultaneously instead of the intended staggered
wake-up. When True, initializes ``steps_since_fired`` to a uniform
random draw over ``[0, wake_gate_steps)`` instead, spreading first
eligibility across the first ``wake_gate_steps`` calls. Uses ``rng``
if given, else ``RandomState(wake_seed)`` -- same convention as
``wake_sign``'s own seeded draw.

.. _apply_energy_dynamics.aux_loss_zero_init_escape:

``aux_loss``: the zero-init escape proof
-------------------------------------------

*ID:* ``apply_energy_dynamics.aux_loss_zero_init_escape``

Setting ``loss = aux_loss`` for every layer is sufficient for
training. Proof for the chain ``x ->[ED]-> x_out ->[W]-> y ->[ED]``:

.. code-block:: text

   c_j          = e_j + drive + noise_j       (independent of h)
   new_energy_t = c_j - activation_cost * y_j (linear surrogate)
   energy_loss  = (alpha/2) * sum((new_energy_t_j - tau)^2)

   d(loss)/d(W_ji) = alpha * (new_energy_t_j - tau) * (-gamma) * x_out_i

Zero-init escape has two independent mechanisms:

1. Each layer's own ``aux_loss``:
   ``d(loss)/d(h) = alpha*(c-gamma*h-tau)*(-gamma)``. ``c ~ drive`` at
   init, so ``(drive-tau) ~ -0.85 != 0`` from step 1 onward. This
   reaches upstream weights immediately, before any firing.
2. Fire-together-wire-together: once neurons fire (``~2.0/drive ~ 14``
   steps), ``x_out_i = 2.0`` makes ``d(aux_loss_y)/d(W_ji)`` non-zero
   for the NEXT layer.

These two mechanisms are complementary, not the same path.

Zero-loss target is ``new_energy_t = tau`` (the homeostatic setpoint).
Push direction and magnitude depend on running energy history -- not
just ``sign(h)`` -- producing genuine homeostatic drive rather than
simple weight decay. In RNNs/agents the network can learn to take
actions that bring future inputs into the comfort zone.

Learning rule: ``x_out_i`` fires AND ``y_j`` needs more drain ->
``W_ji`` grows (fire together, wire together).

The physical energy update still uses ``np.abs(h_dz)`` -- only the
gradient path changes. ``kl_loss`` has no ``h``-gradient (discrete
mask); added as a plain float, ``_coerce`` promotes it when summed
with ``energy_loss``.

The quadratic loss form targets exactly ``setpoint`` (tau) for zero
loss:

.. code-block:: text

   L_j = (alpha/2) * (new_energy_t_j - tau)^2
   d(L)/d(W_ji) = alpha * (new_energy_t_j - tau) * (-gamma) * x_out_i

At zero init, ``new_energy_t_j ~ drive``, so ``(drive - tau)`` is a
non-zero constant (e.g. ``0.15 - 1.0 = -0.85``), giving non-zero
gradients immediately. The zero-crossing is at
``new_energy_t = tau`` -- the homeostatic setpoint. Push direction
depends on history (running energy) and upstream activity, not just on
the sign of ``h``, which produces richer dynamics in RNNs/agents.

Human note: this actually pulls ``e`` to ``t``, not ``|e|`` to ``t``.
To pull ``|e|`` to ``t``, we would need
``energy_loss = (reactivity/2.0) * (new_energy_t.abs() - setpoint).pow(2).sum()``,
but that's complicated. Doing that would allow neurons at ``e=t`` to
fire rapidly and go to ``e=-t`` quickly, but there doesn't seem to be
much actual benefit to that.

.. _apply_energy_dynamics.gather_not_mask:

Real bug: a multiplicative stale-mask lets ``inf * 0.0 == nan`` poison the whole loss
------------------------------------------------------------------------------------------

*ID:* ``apply_energy_dynamics.gather_not_mask``

``energy_loss`` is restricted to STALE positions via ``gather()`` (true
topological exclusion), NOT ``expr * stale_mask``. Confirmed directly
this matters, not just style: L1-sparsity's own known unbounded-output
tendency can push ``abs(h)`` large enough at an AWAKE position (never
corrected by any energy mechanism while masked off) that
``(new_energy_t-setpoint)**2`` overflows to ``inf`` in float32 -- and
``inf * 0.0 == nan``, so a plain multiplicative mask does NOT protect
against contamination; the nan poisons the whole ``.sum()`` and thus
the ENTIRE ``aux_loss`` gradient, for every parameter, awake or stale.
``gather()`` never evaluates ``abs(h)`` at excluded positions in the
first place, so this can't happen no matter how large an awake
neuron's ``h`` grows.

.. code-block:: python

   # as of PR (energy.py, this docs pass):
   stale_idx = np.where(stale)[0]
   if len(stale_idx) > 0:
       c_stale_np = (energy_flat + drive + noise)[stale_idx]
       c_stale_t = Tensor(c_stale_np.astype(dtype), backend=h.backend)
       h_stale = gather(h, stale_idx)  # never touches excluded positions
       new_energy_t_stale = c_stale_t - activation_cost * abs(h_stale)
       energy_loss = (reactivity / 2.0) * ((new_energy_t_stale - setpoint) ** 2).sum()
   # WRONG: computing over the full population and multiplying by a mask,
   # e.g. `((new_energy_t - setpoint) ** 2 * stale_mask).sum()` -- an
   # overflowed awake-position term is still evaluated, and inf*0.0==nan
   # poisons the sum regardless of the mask.

.. _apply_energy_dynamics.kept_indices:

``kept_indices``: the gate decision IS the sparsity decision
----------------------------------------------------------------

*ID:* ``apply_energy_dynamics.kept_indices``

Flat (raveled) indices into ``h``'s original shape of every position
gating kept alive (normal + fired + shutoff-but-kept), sorted
ascending. This IS the gate decision energy already made -- the
intended source for building a sparse (CSR) representation of ``h``
for the next consumer, using ``h``'s PRE-gating values at these
indices (not ``h_out``'s post-gating fire/shutoff constants), rather
than re-deriving sparsity via an independent top-k pass that could
disagree with what energy actually decided.

``EnergyDynamics.forward`` caches this as ``self.kept_indices`` on
every call.

branching-ratio trackers
--------------------------

.. _branching_ratio_tracker.model:

The branching-with-immigration model and why recurrent-only activity matters
----------------------------------------------------------------------------------

*ID:* ``branching_ratio_tracker.model``

``BranchingRatioTracker`` and ``EMABranchingRatioTracker`` estimate the
branching ratio ``m`` of a self-propagating activity process from a
scalar activity count fed in once per step. Background:
``a[t+1] = m*a[t] + h`` models activity as a branching process with
immigration -- ``m`` is the self-propagation (recurrent) factor, ``h``
an external drive. ``m in [0.97, 0.99]`` ("near-critical") is the
intended healthy operating band for a genuinely self-sustaining
recurrent pathway; ``m`` near 0 means activity is being carried
entirely by fresh external drive each step, with no real recurrent
memory.

Identifiability note (why this exists as its own tracker, not a stat
computed on ``SparseRNNCell``'s combined ``h``): if fed the COMBINED
activity of ``input_proj(obs) + recurrent(state)``, this estimate
cannot distinguish "the recurrent pathway genuinely self-propagates"
from "fresh input alone keeps activity in-band while the recurrent
branching factor is silently 0" -- both produce statistically
identical activity sequences once mixed. Feed this tracker
``recurrent(state)``'s OWN activity, measured BEFORE it's summed with
``input_proj(obs)``'s contribution, to get a meaningful answer. See
``SparseRNNCell.forward`` for the split measurement this is meant to
be used with.

Estimator: single-lag OLS slope of ``a[t+1]`` on ``a[t]`` over a
sliding window (``BranchingRatioTracker``) or an EMA-streaming variant
(``EMABranchingRatioTracker``) -- a simplified, single-lag version of
the multistep-regression (MR) estimator (Wilting & Priesemann) that
method is otherwise named after; this implementation does not do the
multi-lag extrapolation MR uses to correct for subsampling bias, which
matters more at large scale than it does for the per-region window
sizes this is intended for. Treat ``branching_ratio()`` as a useful
trend/regime indicator, not a publication-grade MR estimate.

``avalanche_sizes()`` (``BranchingRatioTracker`` only) extracts
consecutive-nonzero-activity run lengths from the current history
window -- a power-law-tail in this distribution is the actual
falsifiable signature of self-organized criticality (Bak, Tang &
Wiesenfeld 1987; Beggs & Plenz 2003), independent of whether
``branching_ratio()`` alone looks healthy. ``EMABranchingRatioTracker``
does NOT implement this: that check inherently needs a retained
sequence of raw activity values, which an O(1) EMA state cannot
reconstruct by construction -- run a ``BranchingRatioTracker``
alongside if that check is wanted.

``EMABranchingRatioTracker``'s ``alpha`` decay rate is the direct
tradeoff ``BranchingRatioTracker``'s ``window`` only offers indirectly:
large alpha = fast response to a changing regime but noisy/short
effective memory; small alpha = smooth long-term trend but slow to
notice a real regime shift. Want both a fast and a long-term read at
once? Run two instances side by side at different alphas (e.g.
``alpha=0.2`` "fast" + ``alpha=0.02`` "slow") rather than looking for a
single-instance dual-timescale mode -- composition over a wider
single-class API, keeping each instance's semantics simple.
