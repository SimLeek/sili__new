# Behavioral Regimes of the Homeostatic Parameter Space

Analysis of what each parameter independently controls, and how combinations
map to mammal-like behavioral and affective states. Intended as a reference
for the prefrontal meta-controller that sets per-region hyperparameters.

---

## Parameter Axes: Distinct Roles

The full parameter set $\{\delta, \gamma, \lambda_{KL}, \beta, \varepsilon, \sigma, \alpha, \tau\}$
spans **five independent behavioral axes**. These are not interchangeable.

---

### Axis 1 — Metabolic Drive: $\delta$

**What it controls:** Baseline energy accumulation rate, independent of input.

$\delta$ is the *unconditional* pressure toward firing. It represents how much
the system wants to be active regardless of what it is receiving.

| $\delta$ | Behavioral Analog |
|---|---|
| $\delta \to 0$ | Coma / deep anesthesia — neurons only fire if driven by input |
| $\delta$ low | Sleep — slow spontaneous activity, consolidation-dominant |
| $\delta$ moderate | Resting wakefulness |
| $\delta$ high | Heightened arousal — system is chronically near threshold |
| $\delta \gg 2\gamma$ | Seizure-like — fire threshold reached faster than input can drain energy |

**Key distinction from $\sigma$:** $\delta$ is a deterministic drift. Every neuron
is pushed toward firing at the same rate, uniformly. It sets the *tempo* of
the system.

---

### Axis 2 — Input Sensitivity: $\gamma$

**What it controls:** How much the hidden state magnitude $|h|$ drains energy.

From Theorem 1, the fixed-point manifold is $|h^*| = \delta/\gamma$. So $\gamma$
sets how *costly* active representation is. High $\gamma$ means large activations
rapidly deplete energy — the system is penalized for sustained strong responses.

| $\gamma$ | Behavioral Analog |
|---|---|
| $\gamma$ low | Low metabolic cost of attention — can sustain high $|h|$ indefinitely |
| $\gamma$ moderate | Normal cognitive load |
| $\gamma$ high | Rapid mental fatigue — strong responses quickly exhaust energy |
| $\gamma \gg \delta$ | System is chronically energy-starved by its own activity |

**The $\delta/\gamma$ ratio specifically:** This is the *target activation magnitude*
— the prediction error the system is calibrated to tolerate at equilibrium
(Theorem 4). It is the single most important ratio for setting cognitive baseline.

$$\text{Target prediction error} = \frac{\delta}{c\gamma}$$

Changing $\delta$ and $\gamma$ by the same factor leaves the fixed point unchanged
but alters the *timescale* of energy dynamics.

---

### Axis 3 — Neural Noise: $\sigma$

**What it controls:** Stochasticity of the energy random walk — spontaneous,
input-independent variation in energy per step.

$$e_{t+1} = e_t + \delta + \xi_t - \gamma|h_t|, \qquad \xi_t \sim \mathcal{N}(0, \sigma^2 I)$$

$\sigma$ is *not* the same as $\delta$. $\delta$ pushes every neuron in the same
direction at the same rate. $\sigma$ introduces *independent* random fluctuations
per neuron per step — some neurons are randomly pushed toward threshold, others
away, with no correlation.

| $\sigma$ | Behavioral Analog |
|---|---|
| $\sigma \to 0$ | Purely deterministic — firing is entirely input-driven |
| $\sigma$ low | Normal neural noise — occasional spontaneous activity |
| $\sigma$ moderate | Hypnagogic state — spontaneous imagery, loosely associated thought |
| $\sigma$ high | Psychedelic-like — uncorrelated spontaneous firing across neurons |
| $\sigma \gg \delta$ | Noise-dominated — input signal is drowned out |

**Key distinction from $\delta$:** $\sigma$ introduces *decorrelated* variability.
At high $\sigma$, different neurons fire at different times for no input reason —
this breaks the synchrony that $\delta$ alone would produce.

From the first-passage analysis, $\sigma$ shortens the expected time to fire
and adds variance to *when* any given neuron fires:

$$\mathbb{E}[T_{fire}] \approx \frac{2}{\delta}, \qquad \text{Var}(T_{fire}) \propto \frac{\sigma^2}{\delta^3}$$

High $\sigma$ means fire times are unpredictable even if $\delta$ is moderate.

---

### Axis 4 — Homeostatic Setpoint: $\tau$

**What it controls:** The energy magnitude that the system is rewarded for,
i.e., the center of the comfort zone.

$$\mathcal{L}_{energy} = \alpha \sum_i (|e_i| - \tau)$$

The system is rewarded when $|e| < \tau$ and penalized when $|e| > \tau$.
This shifts the entire homeostatic manifold $\mathcal{H}^*$.

| $\tau$ | Behavioral Analog |
|---|---|
| $\tau \to 0$ | Maximally tight regulation — any deviation from zero is penalized |
| $\tau < 1$ | High-precision, low-tolerance state — anxiety-like |
| $\tau = 1$ | Default — balanced comfort zone |
| $\tau > 1$ | Wide comfort zone — system tolerates high-energy states |
| $\tau \to 2$ | Comfort zone encompasses almost the full range — near-indifferent |

**Affective interpretation:** $\tau$ is the system's *emotional set point* — how
much internal deviation it considers acceptable before triggering corrective
action. Low $\tau$ is anxiety: the system acts to restore calm at the slightest
perturbation. High $\tau$ is emotional blunting: large deviations are tolerated
passively.

---

### Axis 5 — Homeostatic Pressure: $\alpha$

**What it controls:** How forcefully the system is pushed back toward $|e| < \tau$.
This is the *gain* of the homeostatic feedback loop.

| $\alpha$ | Behavioral Analog |
|---|---|
| $\alpha \to 0$ | No homeostatic pressure — system drifts freely |
| $\alpha$ low | Slow emotional recovery — perturbations decay gradually |
| $\alpha$ moderate | Normal emotional regulation speed |
| $\alpha$ high | Rapid regulation — strong snap-back to comfort zone |
| $\alpha \gg 1$ | Overcorrection — oscillation around $\tau$ |

**Key distinction from $\tau$:** $\tau$ sets *where* the comfort zone is.
$\alpha$ sets *how hard* the system fights to stay there.

A high-$\tau$, high-$\alpha$ system tolerates wide deviations but snaps back
forcefully when it does react — impulsive, with high activation threshold but
strong response once triggered.

A low-$\tau$, low-$\alpha$ system reacts to tiny deviations but slowly —
chronically anxious but with poor ability to actually regulate.

---

### Axes 6–8 — Representational Parameters: $\beta$, $\lambda_{KL}$, $\varepsilon$

These three act jointly on the *structure* of representations rather than the
energy dynamics directly.

**$\beta$ — Target sparsity:**
The density of active neurons at equilibrium. From Theorem 3, lower $\beta$
drives stronger representational independence (more ICA-like). Higher $\beta$
permits denser, more overlapping codes.

| $\beta$ | Behavioral Analog |
|---|---|
| $\beta$ low | Highly localized, specific representations — expert knowledge |
| $\beta$ moderate | Mixed sparse-dense — general cognition |
| $\beta$ high | Dense codes — fast pattern matching, poor discrimination |

**$\lambda_{KL}$ — Sparsity enforcement strength:**
How aggressively the system is pushed toward $\beta$. Analogous to how strongly
the brain enforces metabolic efficiency constraints on neural coding.

High $\lambda_{KL}$ with low $\beta$: very few neurons active, representations
are highly separated — good for discrimination, poor for generalization.

Low $\lambda_{KL}$: sparsity is a soft suggestion — the system may settle at
$\rho \neq \beta$ if other pressures dominate.

**$\varepsilon$ — Activation dead zone:**
Sets the granularity of $\rho$. Small $\varepsilon$ means tiny activations count
as "on" — the system has fine-grained control over sparsity. Large $\varepsilon$
ignores small activations — coarser, more binary behavior.

---

## Behavioral State Mapping

Using all axes together. Each state specifies the *direction* of each parameter
relative to its default.

### Sleep / Memory Consolidation

$$\delta \downarrow,\quad \gamma \text{ nominal},\quad \sigma \downarrow,\quad \tau \downarrow,\quad \alpha \uparrow,\quad \lambda_{KL} \uparrow$$

- Low drift and noise: neurons fire only when residual energy demands it
- High $\alpha$ with low $\tau$: tight homeostatic enforcement — the system
  is actively compressing toward $|e| \approx 0$
- High $\lambda_{KL}$: sparsity enforcement runs without input competition —
  redundant co-activations are pruned
- This implements **synaptic homeostasis** (Tononi): weak attractors are
  starved out, strong ones are preserved

### Resting Wakefulness / Default Mode

$$\delta \text{ nominal},\quad \sigma \text{ low},\quad \tau = 1,\quad \alpha \text{ moderate}$$

- Theorem 7 conditions all approximately satisfied
- System sits near $\mathcal{H}^*$ — prediction error at equilibrium, sparse
  independent codes, bounded energy
- Occasional spontaneous firing from $\sigma > 0$ drives offline exploration

### Focused Attention / Flow

$$\delta/\gamma \text{ tightly calibrated to input},\quad \sigma \downarrow,\quad \tau = 1,\quad \alpha \uparrow,\quad \lambda_{KL} \uparrow$$

- Near-zero noise: firing is entirely input-driven
- High homeostatic pressure: rapid snap-back to comfort zone after perturbation
- High sparsity enforcement: each active neuron is as informative as possible
- From Theorem 5: world model and critic are receiving maximally clean signal

### Curiosity / Intrinsic Motivation

$$\sigma \uparrow,\quad \delta \text{ moderate},\quad \tau \text{ moderate},\quad \gamma \downarrow$$

- High noise: spontaneous exploration of state space not driven by input
- Low $\gamma$: activations are cheap — the system can afford to sustain
  exploratory representations without energy penalty
- World model receives surprise signal from spontaneous firing in
  under-predicted regions — equivalent to curiosity bonus

### Anxiety

$$\tau \downarrow,\quad \alpha \uparrow,\quad \delta \uparrow,\quad \sigma \text{ low}$$

- Tight comfort zone with high gain: any deviation triggers strong corrective action
- High $\delta$: system is chronically near threshold — alert to everything
- Low $\sigma$: not random noise, but deterministic over-responsiveness
- Actor preferentially takes actions that minimize $|e|$ — behavioral rigidity,
  preference for predictable outcomes

### Stress / Acute Threat Response

$$\delta \uparrow\uparrow,\quad \gamma \uparrow,\quad \tau \uparrow,\quad \alpha \downarrow,\quad \sigma \text{ low}$$

- High $\delta$ and $\gamma$: both accumulation and drain rates are elevated —
  fast dynamics, high throughput
- High $\tau$: comfort zone is widened — system accepts high-energy states as
  normal (allostatic shift)
- Low $\alpha$: homeostatic pressure is reduced — the system *tolerates*
  deviation rather than fighting it, freeing resources for action
- This is the **allostatic** model of stress: the setpoint shifts rather than
  the system trying to restore the old setpoint

### Mania

$$\delta \uparrow\uparrow,\quad \sigma \uparrow,\quad \tau \uparrow\uparrow,\quad \alpha \downarrow,\quad \lambda_{KL} \downarrow$$

- Very high metabolic drive with wide comfort zone: system is energy-rich and
  not penalized for it
- High noise: uncorrelated spontaneous firing — loosely associated ideation
- Low sparsity enforcement: representations become dense and overlapping —
  ideas bleed into each other, poor discrimination
- From Theorem 6(a), pathological attractors that are normally extinguished
  by the threshold can now be sustained because $\tau$ is high enough that
  the system never receives a strong corrective signal

### Depression

$$\delta \downarrow,\quad \sigma \downarrow,\quad \tau \downarrow,\quad \alpha \text{ high},\quad \gamma \uparrow$$

- Low metabolic drive: neurons only fire under strong input pressure
- High $\gamma$: representations are metabolically expensive — the system
  rapidly drains energy for any activity, reinforcing inactivity
- Low $\tau$ with high $\alpha$: the comfort zone is narrow and the system
  fights to stay there — any excitation is rapidly suppressed
- World model and critic (Theorem 5) receive sparse training signal —
  predictions become stale, value estimates regress to mean

### Psychedelic / Altered Perception

$$\sigma \uparrow\uparrow,\quad \lambda_{KL} \downarrow,\quad \varepsilon \downarrow,\quad \tau \uparrow$$

- Extreme noise: spontaneous firing is high and decorrelated — perceptual
  content is generated endogenously
- Low sparsity enforcement with small dead zone: many neurons are active,
  many representations are co-active — pattern boundaries dissolve
- High $\tau$: system does not attempt to restore order — deviation is accepted

---

## The Prefrontal Meta-Controller

The system described — reading energy maps as images via a transformer and
setting $\{\delta, \gamma, \sigma, \tau, \alpha, \beta, \lambda_{KL}\}$ per region — is doing
**hierarchical homeostatic regulation**, which is precisely the functional
definition of prefrontal cortical control in neuroscience.

The key design constraints that follow from the above analysis:

**1. Parameters must be set per-region, not globally.**
Different regions need different regimes simultaneously. A sensory region during
focused attention needs low $\sigma$, high $\lambda_{KL}$. A motor region during
the same task needs higher $\delta$ and $\gamma$ to sustain action readiness.
Global parameter changes would produce only global state shifts — no selectivity.

**2. Energy maps as images is well-motivated.**
The energy vector $e \in \mathbb{R}^d$ per region, viewed spatially, encodes:
- Which neurons are near firing (high $e$)
- Which are in refractory states (low $e$)
- The spatial correlational structure of activation

A convolutional or attention-based transformer can read these as local and
global patterns — analogous to how the PFC reads neuromodulatory signals from
subcortical structures.

**3. Output limits are necessary.**
From the regime map, some parameter combinations are destabilizing
($\delta \gg 2$ before threshold, $\alpha \to 0$ with high $\delta$). The
meta-controller must be constrained to the stable subspace. This is equivalent
to the biological finding that PFC modulation of subcortical arousal systems
operates within narrow gain ranges — runaway parameter settings correspond to
psychiatric pathology.

**4. The meta-controller is itself a homeostatic system.**
If the prefrontal network is also an RNN with energy dynamics, it is subject
to the same theorems. Its own $\mathcal{H}^*$ enforces that its parameter
outputs remain stable — a self-regulating regulator. Pathological PFC states
(extreme values of its own $e$) would map to failures of affect regulation,
consistent with models of PTSD, bipolar disorder, and addiction as dysregulated
prefrontal homeostasis.