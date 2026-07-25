# Citations

External work referenced by design decisions in `energy.py`,
`energy-params.md`, `energy-personality.md`, `energy-proofs.md`, and the
Phase E (embodiment/RL) planning in `sili_peridot/todolist.md`. Grouped by
topic; each entry is cited because it directly motivated or named a
specific mechanism in this codebase, not as general background reading.

## Homeostasis / cybernetics

- Ashby, W. R. — Homeostat (1948); *Design for a Brain*. The general
  homeostatic-regulation framing `EnergyDynamics` is built on.
- Ashby's Law of Requisite Variety — informs why per-region parameter
  diversity (not one global energy config) matters; see
  `energy-params.md`'s prefrontal meta-controller section.

## Active inference / free energy

- Friston, K. — Free Energy Principle; predictive coding; dark-room
  problem resolution via interoceptive priors. `energy-proofs.md` frames
  `EnergyDynamics` as an implicit predictive-coding mechanism (no
  explicit reconstruction loss needed) in this tradition.

## Self-organized criticality / avalanches

- Bak, P., Tang, C., Wiesenfeld, K. (1987) — sandpile model, SOC.
- Beggs, J. M. & Plenz, D. (2003) — neuronal avalanches. Both cited
  directly in `BranchingRatioTracker.avalanche_sizes()`'s docstring — a
  power-law tail in avalanche sizes is the actual falsifiable SOC
  signature, independent of whether `branching_ratio()` alone looks
  healthy.
- Wilting, J. & Priesemann, V. — multistep-regression (MR) estimator for
  branching ratio, separating external drive from internal propagation.
  `BranchingRatioTracker.branching_ratio()` is named after and modeled on
  this estimator, but implements only a simplified single-lag OLS version
  — not the full multi-lag subsampling-corrected MR estimator. See that
  method's docstring.
- Williams-García, R. et al. — quasicriticality / "Widom line". Relevant
  to interpreting `branching_ratio()` estimates near but not exactly at
  the intended `m` band.

## Metabolic constraints / efficient coding

- Attwell, D. & Laughlin, S. B. (2001) — energy budget for cortical
  signaling. Motivates `activation_cost` (gamma) as a real metabolic
  constraint, not an arbitrary regularizer.
- Levy, W. B. & Baxter, R. A. — efficient coding under energy constraint.
- Olshausen, B. A. & Field, D. J. (1996) — sparse coding. General
  motivation for the KL-sparsity term (`precision`/`density`).

## Reservoir computing

- Echo state property / spectral radius tuning near 1 (Jaeger — echo
  state networks). The `m` (branching ratio) target band `[0.97, 0.99]`
  is the same "near the edge of stability" idea as spectral-radius
  tuning in reservoir computing, applied to a spiking/energy-gated
  system instead of a fixed linear reservoir.

## Intrinsic motivation / curiosity

- Schmidhuber, J. — "Driven by Compression Progress" (2009); compression
  progress as reward. Motivates the NCD-based `ncd_view_hidden` curiosity
  metric in `experiment_mandelbrot.py`.
- Pathak, D. et al. (2017) — Intrinsic Curiosity Module (ICM).

## Homeostatic RL (formal)

- Keramati, M. & Gutkin, B. — homeostatic reinforcement learning.
- Hull, C. — drive-reduction theory (1943). Historical precedent for
  "drive" as an RL-relevant quantity, predating `EnergyDynamics`'s own
  `drive` parameter by 80+ years but naming a related idea.

## Sparse coding mechanisms (lifetime vs. population sparsity)

- Willmore, B. & Tolhurst, D. — lifetime vs. population sparsity
  distinction (following Foldiak). Relevant to the difference between
  `EnergyDynamics`'s per-neuron energy setpoint (lifetime-sparsity-like)
  and its population-level KL density term (population-sparsity-like) —
  see the `precision`/`reactivity` docstring split in `energy.py`.
- Makhzani, A. & Frey, B. (2013) — k-sparse autoencoders. The hard top-p
  gate (`p`) is a k-sparse-style hard constraint, though here explicitly
  scoped as a hardware ceiling rather than the mechanism shaping learned
  sparsity — see `p`'s docstring in `energy.py`.
- Rozell, C. et al. (2008) — locally competitive algorithms (LCA).

## Signal detection / stochastic resonance

- Signal detection theory (general) — relevant to the fire/shutoff
  threshold framing in `_apply_energy_dynamics`.
- Stochastic resonance literature (general) — relevant to why
  `exploration` (noise) is bounded below `drive/2` rather than being
  minimized to zero; see that parameter's docstring.

## Multi-agent credit assignment

(Relevant to the eventual per-neuron critic needed before Phase E's
action pathway — see `sili_peridot/todolist.md` Phase E — not yet
implemented, tracked as a blocker on that work.)

- Wolpert, D. & Tumer, K. (1999) — COllective INtelligence (COIN),
  factoredness/sensitivity, difference/aristocrat utility.
- Foerster, J. et al. (2018) — COMA (counterfactual multi-agent policy
  gradients).
- Sunehag, P. et al. (2017) — VDN (value decomposition networks).
- Rashid, T. et al. (2018) — QMIX.

## Indirect encoding / decoupled updates

- Stanley, K., D'Ambrosio, D., Gauci, J. (2009) — HyperNEAT, CPPN
  indirect encoding.
- Jaderberg, M. et al. (2016) — Synthetic gradients / Decoupled Neural
  Interfaces (DNI). Relevant to any future work decoupling
  synaptogenesis's update cadence from backprop's.
