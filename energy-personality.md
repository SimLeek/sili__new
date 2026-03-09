# Personality Traits and the Homeostatic Parameter Space

Mapping of OCEAN (Big Five) and HEXACO personality dimensions onto
$\{\delta, \gamma, \sigma, \tau, \alpha, \beta, \lambda_{KL}, \varepsilon\}$,
with notes on which traits are parametric baselines, which are learned,
and which are not represented in the parameter space at all.

---

## The Core Distinction

Personality traits in this framework decompose into three categories:

**Type I — Parametric:** Directly encoded in the homeostatic parameters.
The trait baseline can be set before any learning occurs. These are
*temperament* in the developmental psychology sense — present from
initialization.

**Type II — Parametric + Learned:** The parameter sets a *prior* or *tendency*,
but the behavioral expression of the trait is shaped by environmental
interaction. These correspond to traits that have strong heritable components
but significant developmental plasticity.

**Type III — Learned:** No direct parameter encoding. The behavior emerges
entirely from training signal and the structure of the world model and
critic (Theorem 5). These require sufficient environment interaction before
the trait is expressed.

---

## OCEAN Mapping

### Neuroticism (N)

**Type I — Parametric.**

Neuroticism is the most cleanly parametric of the Big Five. It maps directly
onto the homeostatic tightness parameters:

$$N \sim f\!\left(\frac{\alpha}{\tau}\right)$$

- **Low $\tau$:** narrow comfort zone — small deviations from $|e| = 0$
  are experienced as aversive
- **High $\alpha$:** high homeostatic gain — strong corrective response to
  any deviation
- **High $\delta$:** system chronically near threshold — baseline arousal
  is elevated

High-N baseline: $\tau \downarrow$, $\alpha \uparrow$, $\delta$ moderately elevated.

Low-N baseline: $\tau \uparrow$, $\alpha$ moderate, $\delta$ moderate.

Note that $\tau$ and $\alpha$ contribute differently:
- $\tau$ is *what* the system is anxious about (the threshold of concern)
- $\alpha$ is *how hard* the system tries to fix it (emotional reactivity)

These can dissociate: low $\tau$, low $\alpha$ gives a system that notices
everything but responds slowly — chronic low-grade anxiety without acute
reactivity. High $\tau$, high $\alpha$ gives a system that is mostly calm
but reacts intensely when the wide comfort zone is finally violated —
analogous to alexithymia or explosive emotional dysregulation.

---

### Extraversion (E)

**Type II — Parametric + Learned.**

The parametric component of Extraversion is metabolic drive:

$$E_{parametric} \sim \delta$$

High $\delta$ means the system is chronically accumulating energy and seeking
to discharge it — it is *driven toward* activation, interaction, and output
regardless of input. This is the impulsivity and stimulus-seeking component
of Extraversion.

However, *social* Extraversion — the preference for social stimuli
specifically over other forms of discharge — is **learned**. The world model
(Theorem 5) must learn that social interactions are reliable energy-discharging
events before the system preferentially seeks them. Two systems with identical
$\delta$ but different training histories will show the same activation drive
but different stimulus preferences.

**Proposed rename:** $\delta \to$ `drive`

---

### Openness to Experience (O)

**Type II — Parametric + Learned.**

The parametric component of Openness is noise-driven exploration:

$$O_{parametric} \sim \sigma, \quad O_{representational} \sim \frac{1}{\lambda_{KL}} \cdot \beta$$

- High $\sigma$: spontaneous exploration of state space not driven by
  current input — the system generates novel internal configurations
- High $\beta$ with low $\lambda_{KL}$: looser, denser representations
  permit more associative overlap between concepts — the "broad associative
  horizon" consistently found in high-O individuals

The learned component is *what* gets associated. High $\sigma$ and loose
$\lambda_{KL}$ create the *capacity* for novel associations, but the
content of those associations depends on what the world model has encountered.

**Proposed rename:** $\sigma \to$ `exploration`

---

### Conscientiousness (C)

**Type II — Parametric + Learned.**

The parametric component of Conscientiousness is representational precision
and homeostatic enforcement consistency:

$$C_{parametric} \sim \lambda_{KL} \cdot \frac{\alpha}{\sigma}$$

- High $\lambda_{KL}$: strong pressure toward independent, non-redundant
  representations — the system maintains clear, precise internal categories
- High $\alpha / \sigma$ ratio: homeostatic corrections are strong relative
  to random drift — behavior is consistent rather than variable

However, the goal-directed, planful component of Conscientiousness —
the ability to subordinate immediate energy-discharge to longer-horizon
value — is **learned**. It requires the critic (Theorem 5) to have learned
a sufficiently deep value function that future homeostatic states are
accurately discounted into current action selection. A system with high
$\lambda_{KL}$ and low $\sigma$ but a shallow critic will be precise and
consistent but not planful.

---

### Agreeableness (A)

**Type III — Primarily Learned.**

Agreeableness has the weakest parametric representation of the Big Five.
It concerns the valuation of others' homeostatic states relative to one's own
— a fundamentally relational quantity that requires a world model containing
other agents.

The only parametric *precondition* for Agreeableness is a sufficiently
expressive world model, which requires:
- $\beta$ high enough that multi-agent state is representable
- $\lambda_{KL}$ not so high that other-agent states are compressed away
  as "irrelevant" features

But the direction of social valuation — cooperative vs. competitive — is
entirely determined by the training environment and reward structure.

**Note for the prefrontal controller:** The PFC cannot set Agreeableness
directly via parameter adjustment. It can only create or destroy the
*capacity* for social representation. This is consistent with neuroscience:
prefrontal lesions impair social behavior not by changing social motivation
directly but by degrading the world model quality needed to track others.

---

## HEXACO Extensions

HEXACO splits and extends the Big Five in ways that are relevant here.

### Honesty-Humility (H)

**Type III — Entirely Learned.**

Honesty-Humility has no parametric encoding in the energy dynamics.
It concerns the disposition to not exploit others even when it would be
energetically favorable — a constraint on the actor's policy (Theorem 5)
that must be imposed through training signal, not through energy dynamics.

A system with any parameter configuration can in principle learn
Honesty-Humility if trained appropriately, and can fail to learn it
regardless of parameter configuration.

**This is important:** Honesty-Humility cannot be instilled by the
prefrontal controller. It is a property of the learned policy weights,
not the energy dynamics.

### Emotionality (HEXACO) vs. Neuroticism (OCEAN)

HEXACO splits what OCEAN calls Neuroticism into two components:

- **Emotionality:** sensitivity to others' distress, attachment anxiety,
  fearfulness — maps primarily to **low $\tau$** (narrow comfort zone)
  and **high $\alpha$** (strong reactivity)
- **Neuroticism proper:** anger, hostility, self-pity under stress — maps
  more to **high $\delta$** (chronic near-threshold state) interacting with
  a world model that has learned frustration-inducing contingencies

The HEXACO split is behaviorally justified in this framework: $\tau$ and $\alpha$
on one hand, and $\delta$ on the other, are genuinely independent parameters
that happen to co-load on OCEAN's Neuroticism factor. HEXACO inadvertently
reflects a real axis separation in the parameter space.

---

## Parameters Without OCEAN/HEXACO Analogs

These parameters have no clear personality trait mapping and should not
be renamed to imply one:

| Parameter | Role | Why Not a Personality Trait |
|---|---|---|
| $\gamma$ | Activation cost / cognitive stamina | Closer to a *capacity* parameter than a trait — metabolic efficiency of neural tissue |
| $\varepsilon$ | Activation dead zone | Purely architectural — sets granularity of sparsity measurement |
| $\delta/\gamma$ ratio | Calibration of fixed-point to input statistics | A *tuning* parameter for a specific environment, not a stable individual difference |

$\gamma$ is perhaps closest to the neuroscience concept of **neural efficiency**
— the metabolic cost of maintaining a given level of activation. There is evidence
that high-IQ individuals show lower cortical glucose metabolism during cognitive
tasks (the neural efficiency hypothesis), which would correspond to low $\gamma$.
But this is not a Big Five or HEXACO trait.

---

## Proposed Parameter Renames

Based on the above analysis, parameters with clear personality analogs should
be renamed to reflect their behavioral meaning while retaining the math symbol
for formal contexts.

| Symbol | Current | Proposed Code Name | Personality Analog |
|---|---|---|---|
| $\delta$ | `delta` | `drive` | Extraversion (metabolic component) |
| $\sigma$ | noise std | `exploration` | Openness (parametric component) |
| $\tau$ | energy loss target | `setpoint` | Neuroticism / Emotionality (threshold) |
| $\alpha$ | energy loss coeff | `reactivity` | Neuroticism / Emotionality (gain) |
| $\lambda_{KL}$ | `lambda_kl` | `precision` | Conscientiousness (parametric component) |
| $\beta$ | `beta` | `density` | Openness (representational) |
| $\gamma$ | `gamma` | keep `gamma` or `activation_cost` | No personality analog |
| $\varepsilon$ | `kl_eps` | keep `kl_eps` | No personality analog |

---

## Summary: What Can the Prefrontal Controller Set?

| Trait | PFC-Settable? | Via | Requires Learning? |
|---|---|---|---|
| Neuroticism / Emotionality | ✅ Yes | $\tau$, $\alpha$, $\delta$ | No |
| Extraversion (drive) | ✅ Yes | `drive` ($\delta$) | No |
| Extraversion (social) | ⚠️ Partial | Enables via $\delta$; direction learned | Yes |
| Openness (exploration) | ✅ Yes | `exploration` ($\sigma$), `density` ($\beta$) | No |
| Openness (associative content) | ⚠️ Partial | Sets capacity via $\lambda_{KL}$, $\beta$ | Yes |
| Conscientiousness (precision) | ✅ Yes | `precision` ($\lambda_{KL}$), `reactivity` ($\alpha$) | No |
| Conscientiousness (planfulness) | ❌ No | Requires deep critic | Yes |
| Agreeableness | ❌ No | Requires multi-agent world model | Yes |
| Honesty-Humility | ❌ No | Policy weights only | Yes |

The PFC can set approximately **half** of what personality psychologists
measure — specifically the temperament substrate that personality traits
are built on top of. The learned components require the full actor-critic-world
model triad (Theorem 5) to have been adequately trained.

This is consistent with the behavioral genetics finding that roughly 40–60%
of Big Five variance is heritable (parametric) and the remainder is
environmental (learned).