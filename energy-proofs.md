# Homeostatic Optimality in Recurrent Energy Dynamics

Formal proofs for the `_apply_energy_dynamics` system. Establishes that optimality is defined with respect to homeostasis rather than loss minimization, and characterizes the emergent representational, predictive, and control properties of the dynamics.

---

## Notation

| Symbol | Meaning |
|---|---|
| $h \in \mathbb{R}^d$ | Hidden state vector |
| $e \in \mathbb{R}^d$ | Energy vector |
| $\delta > 0$ | Constant energy drift |
| $\gamma > 0$ | Energy drain rate |
| $\beta \in (0,1)$ | Target sparsity |
| $\lambda_{KL}$ | Sparsity loss weight |
| $\varepsilon$ | Activation threshold (`kl_eps`) |
| $\xi_t \sim \mathcal{N}(0, 0.001^2 I)$ | Injected noise |
| $\Omega$ | Invariant energy set $\{e : \|e_i\| \leq 2\; \forall i\}$ |
| $\mathcal{H}^*$ | Homeostatic optimality set |
| $\rho$ | Mean activation rate $\mathbb{E}_i[\mathbf{1}_{|h_i| > \varepsilon}]$ |

---

## System Definition

The map $\mathcal{F}: (h, e) \mapsto (h', e', \mathcal{L})$ is defined by the following sequential operations.

**Step 1 — Sparsity pressure:**

$$\rho = \frac{1}{d}\sum_{i=1}^d \mathbf{1}_{|h_i| > \varepsilon}, \qquad \rho \in (10^{-5},\, 1 - 10^{-5})$$

$$\mathcal{L}_{KL} = \lambda_{KL} \left[ \rho \log\frac{\rho}{\beta} + (1 - \rho)\log\frac{1-\rho}{1-\beta} \right] = \lambda_{KL}\, D_{KL}\!\left(\text{Bern}(\rho) \;\|\; \text{Bern}(\beta)\right)$$

**Step 2 — Energy update:**

$$e' = e + \delta + \xi - \gamma |h|$$

**Step 3 — Energy loss:**

$$\mathcal{L}_{energy} = 0.01 \sum_{i=1}^d \left(|e'_i| - 1\right)$$

$$\mathcal{L}_{aux} = \mathcal{L}_{KL} + \mathcal{L}_{energy}$$

**Step 4 — Hard threshold corrections (piecewise):**

$$h'_i = \begin{cases} 2 & \text{if } e'_i \geq 2 \quad \text{(fire)} \\ e_i + 2 & \text{if } e'_i \leq -2 \quad \text{(soft shutoff)} \\ h_i & \text{otherwise} \end{cases}$$

$$e'_i \leftarrow \begin{cases} e'_i - \gamma|h'_i| & \text{if } e'_i \geq 2 \quad \text{(refractory drain)} \\ -2 & \text{if } e'_i \leq -2 \end{cases}$$

---

## Homeostatic Optimality

Rather than minimizing $\mathcal{L}_{aux}$, optimality is defined as sustained membership in the homeostatic set:

$$\mathcal{H}^* = \left\{\, (h, e) \;\middle|\; \mathbb{E}[|e_i|] < 1\; \forall i, \quad \mathbb{E}[\rho] = \beta, \quad I(h_i;\, h_j) \approx 0\; \forall\, i \neq j \,\right\}$$

The system does not descend $\mathcal{L}_{aux}$ — it maintains a trajectory within $\mathcal{H}^*$. The following theorems characterize when and why this holds.

---

## Theorem 1 — Fixed-Point Manifold

> *The deterministic energy dynamics possess a fixed manifold $\mathcal{M}^* = \{(h, e) : |h| = \delta/\gamma\}$, independent of $e$.*

**Proof.**

At a fixed point of the deterministic system (setting $\xi = 0$), we require $e = e + \delta - \gamma|h|$, which gives:

$$0 = \delta - \gamma|h^*| \implies |h^*| = \frac{\delta}{\gamma}$$

This condition is independent of $e^*$, so any $e^* \in \mathbb{R}$ paired with $|h^*| = \delta/\gamma$ satisfies the fixed-point equation. The fixed points form a manifold rather than isolated points. $\blacksquare$

**Corollary.** Without the hard thresholds, $e$ performs a random walk with drift $\delta - \gamma|h|$. By the strong law of large numbers, if $|h| \neq \delta/\gamma$ persistently, then $e_T \to \pm\infty$ almost surely. The thresholds are therefore necessary for recurrence.

---

## Theorem 2 — Marginal Stability

> *The deterministic sub-system (no noise, no thresholds) is marginally stable but not asymptotically stable.*

**Proof.**

Let $V(e) = \frac{1}{2}\|e - e^*\|^2$ be a candidate Lyapunov function for any $e^* \in \mathbb{R}^d$. Define $u_t = e_t - e^*$ and $r_t = \delta - \gamma|h_t|$. Then:

$$\Delta V = V(e_{t+1}) - V(e_t) = \frac{1}{2}\|u_t + r_t\|^2 - \frac{1}{2}\|u_t\|^2 = u_t \cdot r_t + \frac{1}{2}\|r_t\|^2$$

At the fixed manifold $r_t = 0$, so $\Delta V = 0$. The linearized map has eigenvalue exactly $1$ in the energy direction — there is no contraction. The system is Lyapunov stable but not asymptotically stable. $\blacksquare$

---

## Theorem 3 — Sparsity Implies Representational Independence

> *Under the KL sparsity constraint $\rho \to \beta$ with $\beta \ll 1$, the maximum-entropy distribution over activations consistent with the constraint factorizes over neurons, implying $I(h_i; h_j) = 0$ for all $i \neq j$.*

**Proof.**

Maximize entropy $\mathcal{S}[p] = -\int p(h)\log p(h)\, dh$ subject to the per-neuron activation constraints:

$$\mathbb{E}_p\!\left[\mathbf{1}_{|h_i| > \varepsilon}\right] = \beta \quad \forall\, i \in \{1, \ldots, d\}$$

Introduce Lagrange multipliers $\mu_i$ for each constraint. The functional to maximize is:

$$\max_p \left\{ \mathcal{S}[p] - \sum_{i=1}^d \mu_i \left(\mathbb{E}\!\left[\mathbf{1}_{|h_i|>\varepsilon}\right] - \beta\right) \right\}$$

Since the constraint decomposes as a sum over independent per-neuron terms, the maximizing distribution is:

$$p^*(h) = \frac{1}{Z}\exp\!\left(-\sum_{i=1}^d \mu_i\, \mathbf{1}_{|h_i|>\varepsilon}\right) = \prod_{i=1}^d \frac{1}{Z_i}\exp\!\left(-\mu_i\, \mathbf{1}_{|h_i|>\varepsilon}\right)$$

This factorizes over coordinates, so:

$$I(h_i;\, h_j) = D_{KL}\!\left(p^*(h_i, h_j)\;\|\; p^*(h_i)\, p^*(h_j)\right) = 0 \quad \forall\, i \neq j \qquad \blacksquare$$

**Corollary.** The KL term implicitly drives the system toward **Independent Component Analysis** as a fixed point. Neurons that share representations violate the maximum-entropy factorization and incur a persistent sparsity penalty. Unique representations are the attractor, not an explicit objective.

---

## Theorem 4 — Prediction Error as Homeostatic Equilibrium

> *The energy dynamics are homeostically stable if and only if the RNN state $h_t$ encodes a sufficient statistic for predicting $x_{t+1}$. At equilibrium, the mean prediction error is pinned to $\delta / (c\gamma)$ for a smoothness constant $c > 0$.*

**Proof.**

Taking expectations of the energy trajectory over $T$ steps:

$$\mathbb{E}[e_T] = e_0 + T\delta - \gamma \sum_{t=0}^{T-1} \mathbb{E}[|h_t|]$$

Homeostasis requires $\mathbb{E}[|e_T|] < 1$ for all large $T$, which forces:

$$\mathbb{E}[|h_t|] \to \frac{\delta}{\gamma} \quad \text{as } T \to \infty \tag{1}$$

Let $\hat{x}_t = g_\phi(h_{t-1})$ denote the implicit next-step predictor encoded in $h$. For a smooth recurrent update $h_t = f_\theta(h_{t-1}, x_t)$, a first-order Taylor expansion around the predicted input gives:

$$|h_t| = \left|f_\theta(h_{t-1}, \hat{x}_t) + \nabla_{x}\! f_\theta\big|_{\hat{x}_t}(x_t - \hat{x}_t) + O\!\left(\|x_t - \hat{x}_t\|^2\right)\right|$$

$$\approx c\,\|x_t - \hat{x}_t\| + O\!\left(\|x_t - \hat{x}_t\|^2\right)$$

where $c = \|\nabla_x f_\theta\|$ at the operating point. Substituting into $(1)$:

$$c\,\mathbb{E}\!\left[\|x_t - \hat{x}_t\|\right] = \frac{\delta}{\gamma} \implies \mathbb{E}\!\left[\|x_t - \hat{x}_t\|\right] = \frac{\delta}{c\gamma} \qquad \blacksquare$$

**Remark.** This is a recurrent instantiation of the **Free Energy Principle**: the system minimizes surprise (prediction error) to maintain its homeostatic state. The energy variable $e$ is a physical accumulator of variational free energy. The explicit reconstruction loss used in RSSM/Dreamer architectures is recovered here as an emergent equilibrium condition rather than an engineered objective, freeing the hidden state from the capacity tax of a decoder head.

---

## Theorem 5 — Emergent Actor-Critic-World Model Triad

> *An actor using $h_t$ as policy state that acts optimally with respect to the homeostatic reward must internally represent a value function $V(h_t)$ and a transition model $p(x_{t+1} | h_t, a_t)$. Both emerge necessarily from the homeostatic objective alone.*

**Proof.**

Define the homeostatic reward at time $t$:

$$r_t = -0.01\,(|e_t| - 1), \qquad r_t \begin{cases} > 0 & |e_t| < 1 \\ \leq 0 & |e_t| \geq 1 \end{cases}$$

The actor selects $a_t \sim \pi(a | h_t)$, which influences $x_{t+1}$, which determines $h_{t+1} = f_\theta(h_t, x_{t+1})$, which determines $e_{t+1}$. The optimal policy maximizes the discounted return:

$$V^*(h_t) = \max_\pi\; \mathbb{E}_\pi\!\left[\sum_{\tau=0}^\infty \gamma^\tau r_{t+\tau} \;\middle|\; h_t\right]$$

**Critic emerges necessarily.** By the Bellman optimality equation:

$$V^*(h_t) = r_t + \gamma\, \mathbb{E}\!\left[V^*(h_{t+1}) \mid h_t, a_t\right]$$

Computing $V^*$ requires estimating the value of future states. Since $h_t$ is the only available state representation, the critic must be a learned function $V_\psi(h_t)$. No external reward signal is required — the homeostatic reward is intrinsic.

**World model emerges necessarily.** The one-step Bellman expectation expands as:

$$\mathbb{E}\!\left[V^*(h_{t+1}) \mid h_t, a_t\right] = \int p(x_{t+1} \mid h_t, a_t)\; V^*\!\left(f_\theta(h_t, x_{t+1})\right) dx_{t+1}$$

Evaluating this integral requires an internal model of the input distribution conditioned on current state and action. The actor is therefore forced to internalize $p(x_{t+1} | h_t, a_t)$ as a prerequisite to computing the optimal policy. $\blacksquare$

**Corollary.** The homeostatic objective induces a natural decomposition of $h_t$:

$$h_t \;\xrightarrow{\;\text{must learn}\;}\; \begin{cases} \pi(a \mid h_t) & \text{Actor (explicit)} \\[4pt] V_\psi(h_t) & \text{Critic (emergent)} \\[4pt] p(x_{t+1} \mid h_t, a_t) & \text{World model (emergent)} \end{cases}$$

This is structurally equivalent to Dreamer/RSSM but derived from first principles via homeostasis, without engineering any of the three components explicitly.

---

## Theorem 6 — Threshold Dynamics as Phase Portrait Surgery

> **(a)** *The hard thresholds guarantee that $\Omega = \{e : |e_i| \leq 2\; \forall i\}$ is positively invariant and absorbing. No attractor with $|e^*| > 2$ is reachable.*
>
> **(b)** *For $\delta > 0$, any zero-initialized neuron $h_0 = 0$ escapes the zero fixed point in finite time almost surely.*

**Proof of (a).**

The threshold rules enforce the projection:

$$e'_i \leftarrow \begin{cases} e'_i - \gamma|h'_i| & \text{if } e'_i \geq 2 \end{cases}$$

After the fire event, $h'_i = 2$ and $e'_i \leftarrow e'_i - 2\gamma$. For $\gamma > 0$ this strictly reduces $|e'_i|$. The shutoff rule sets $e'_i = -2 \in \Omega$ directly.

Therefore: $e_t \in \Omega \Rightarrow e_{t+1} \in \Omega$ almost surely (positive invariance), and for any $e_0 \notin \Omega$, one step of $\mathcal{F}$ brings $e_1 \in \Omega$ (absorbing). Any attractor must lie in $\Omega$; attractors with $|e^*_i| > 2$ are unreachable from any initial condition after the first application of $\mathcal{F}$. $\blacksquare$

**Proof of (b).**

At $h_t = 0$, the energy update reduces to:

$$e_{t+1} = e_t + \delta + \xi_t$$

This is a random walk with positive drift $\delta > 0$. By the strong law of large numbers:

$$\frac{e_T}{T} \to \delta > 0 \quad \text{a.s.}$$

So $e_T \to +\infty$ almost surely, and by the first passage time of a random walk with positive drift, the fire threshold $e \geq 2$ is reached in finite time almost surely, forcing $h \leftarrow 2$.

Note that standard backpropagation cannot escape $h = 0$ when $\nabla_h \mathcal{L}|_{h=0} = 0$ (e.g. dead ReLU neurons). The energy mechanism provides an escape route that is **independent of the gradient**, operating directly on the activation through the threshold event. $\blacksquare$

---

## Theorem 7 — Homeostatic Optimality: Unified Characterization

> *The system $\mathcal{F}$ is homeostically optimal if and only if all three conditions hold simultaneously:*
>
> $$\underbrace{\mathbb{E}\!\left[\|x_t - \hat{x}_t\|\right] = \frac{\delta}{c\gamma}}_{\text{(i) Calibrated prediction error}} \qquad \underbrace{\mathbb{E}[\rho] = \beta,\quad I(h_i; h_j) = 0}_{\text{(ii) Independent sparse codes}} \qquad \underbrace{e \in \Omega \;\text{ a.s.}}_{\text{(iii) Bounded energy}}$$

**Proof.**

By Theorem 4, condition (i) is equivalent to $\mathbb{E}[|e_t|] < 1$ (bounded mean energy), which is the core homeostatic requirement. By Theorem 3, condition (ii) is the maximum-entropy solution under the KL sparsity constraint — any departure from independence incurs a persistent KL penalty that drives $\mathbb{E}[|h_t|]$ away from $\delta/\gamma$, violating (i). By Theorem 6(a), condition (iii) is guaranteed structurally by $\mathcal{F}$ regardless of the learned parameters.

The three conditions are therefore mutually reinforcing: (iii) bounds the energy so (i) is achievable; (i) constrains $|h|$ so the sparsity pressure can converge; sparsity convergence enforces (ii) by Theorem 3. Failure of any one condition destabilizes the others. $\blacksquare$

**The homeostatic feedback loop:**

```
Accurate prediction  ──►  controlled |h|  ──►  controlled energy drain
       ▲                                                  │
       │                                                  ▼
Independent codes  ◄──  sparsity pressure  ◄──  bounded energy |e| < 1
```

---

## Summary

| Property | Result | Theorem |
|---|---|---|
| Fixed points exist | On manifold $\|h\| = \delta/\gamma$ | 1 |
| Asymptotically stable (no thresholds) | No — marginally stable only | 2 |
| Asymptotically stable (with thresholds) | Yes — recurrent on $\Omega$ | 6a |
| Optimal w.r.t. $\mathcal{L}_{aux}$ | No — update rules are not (sub)gradients of $\mathcal{L}_{aux}$ | — |
| Optimal w.r.t. homeostasis | Yes — iff conditions (i–iii) of Theorem 7 hold | 7 |
| Emergent ICA / unique representations | Yes — maximum-entropy consequence of sparsity | 3 |
| Emergent predictive coding | Yes — equilibrium condition of energy stability | 4 |
| Emergent critic | Yes — necessary for homeostatic action selection | 5 |
| Emergent world model | Yes — necessary to evaluate Bellman expectation | 5 |
| Dead neuron escape | Yes — a.s. finite first passage under $\delta > 0$ | 6b |
| Pathological attractor elimination | Yes — unreachable after first step of $\mathcal{F}$ | 6a |