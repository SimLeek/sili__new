# What the three Titans gates can and cannot do

A machine-checked account of the **retention kernel** behind the Titans
neural long-term memory (arXiv:2501.00663 §3.1), written because two
measurement confounds in the empirical work each reversed a conclusion,
and both turn out to be visible in one line of algebra.

Run `bend PROOF.bend --check-only` (prints `ALL PROOFS CHECK`). The Nat
semiring lemmas live in `../lib/NAT.bend`, shared with the other proofs.

## The object

The Titans inner loop is

```
S_t = eta_t * S_{t-1} - theta_t * grad_M l(M_{t-1}; x_t)     (10)
M_t = (1 - alpha_t) * M_{t-1} + S_t                          (13)
```

with three gates: `alpha` (forgetting / weight decay), `eta` (momentum on
the surprise trace), and `theta` (the inner step size). Write `b = 1 -
alpha`, hold the gates constant, and unroll both recurrences:

```
S_t = -theta * SUM_{i<=t} eta^(t-i) g_i
M_T = SUM_{t<=T} b^(T-t) S_t
    = -theta * SUM_i w(T-i) g_i ,     w(n) = SUM_{u=0..n} b^(n-u) eta^u
```

So the memory is a weighted sum of past gradients, and **the entire
temporal behaviour of the gate pair is carried by the single kernel `w`**.
`w(n)` is the weight given to a gradient written `n` ticks ago.

`w` is a polynomial in `b` and `eta` with Nat coefficients, so it lives in
a commutative semiring and Bend can check facts about it without
real-number arithmetic — the same move that made
`../synaptic_hold_bptt_approximation` checkable.

This derivation was verified numerically against the actual update before
anything was proved about it: simulating eqs 10/13 directly and comparing
against `-theta * SUM_i w(T-i) g_i` agrees to a relative error of ~1e-16
across `b` in {0.8, 0.9, 0.95, 1.0}, `eta` in {0, 0.5, 0.9, 0.95} and
`T` in {4, 8, 16}. A proof about the wrong object would be worse than no
proof.

## The theorems

**Law 1, `kernel_recurrence_exchanged`.** `w` satisfies its own defining
recurrence with `b` and `eta` swapped:

```
eta * w(n) + b^(n+1)  ==  w(n+1)  ==  b * w(n) + eta^(n+1)
```

Stepping the kernel one tick can be charged either to forgetting or to
momentum, with an identical result. (Stated with the exchanged form on the
left because that is the orientation a direct induction closes — see the
rewrite-direction note in `../lib/NAT.bend`.)

**Law 2, `kernel_symmetry` — the main theorem.** `w(b, eta, n) == w(eta,
b, n)`. **The retention kernel is symmetric in `1 - alpha` and `eta`.**

Forgetting and momentum are the *same knob* as far as the memory's
temporal profile is concerned: any kernel shape reachable by tuning
`alpha` is reachable by tuning `eta`, and vice versa. The pair contributes
**one** effective degree of freedom to retention shape, not two.

The practical consequence is a constraint on gate adaptation: a controller
that treats `alpha` and `eta` as independent targets is over-parameterized
on this axis, and any genuine asymmetry between them must come from
something *outside* this kernel. Two candidates, both real: `alpha` also
scales the *current* write while `eta` does not, and `g_i` itself depends
on `M`, so the update is not actually linear.

**Law 3, `kernel_no_forget_is_geometric`.** With no forgetting (`alpha =
0`, so `b = 1`), `w(1, eta, n)` is exactly the geometric sum
`SUM_{u=0..n} eta^u`, whose total mass is `1/(1 - eta)`.

**This is confound 1.** `eta` does not only set the trace — it multiplies
the *effective learning rate* by `1/(1 - eta)`. Comparing `eta` values at
fixed `theta` compares two different learning rates, not two momenta. At
`eta = 0.95` that is a 20x amplification, and the measured consequence was
a memory diverging to `||M|| = 11093` where the step-matched run stayed at
`4.94`. Every "momentum hurts" number collected before this was caught was
an artifact; holding `theta * (1 - eta)` fixed reversed the sign.

**Law 4, `kernel_age_monotone_without_forgetting`.** With no forgetting
the kernel is *non-decreasing in age*: `w(1, eta, n+1) = w(1, eta, n) +
eta^(n+1)`.

**This is confound 2.** Momentum without forgetting is a low-pass filter
that weights a write *more* heavily the older it gets, so `alpha` is the
only source of recency bias in the pair. A task that wrote its probed
content FIRST therefore handed momentum free credit (0.747 at `eta =
0.95`), while the identical gates scored 0.119 once the probes were
written LAST. Neither number measured the mechanism; both measured write
order.

## What this says about adapting the gates

The empirical work found that the two gates trade off along a *temporal*
axis — momentum broadens retention, weight decay sharpens it toward
recent — and that each is clearly beneficial in the regime it targets
(momentum 0.665 -> 0.871 with probes spread uniformly; forgetting 0.387 ->
0.879 with probes recent). Laws 2-4 explain why that is the shape of the
result rather than a coincidence:

- Laws 3 and 4 say `theta` must absorb the kernel's total mass, and that
  `alpha` is what buys recency. So the quantity a controller should sense
  is **how far back the task's retrievals actually reach**, which is
  observable from the stream's own queries without an oracle.
- Law 2 says it should sense *one* such quantity, not two, and spend it on
  the `{b, eta}` pair jointly.

The open question this does **not** settle: Law 2 holds for the linearized
kernel, and the real update's `g_i` depends on `M`. Whether the measured
asymmetry between the two gates is fully explained by that nonlinearity
plus the current-write term, or whether something else is going on, is not
answered here.

## Scope and honesty

- Everything is proved for **constant** gates. The paper's gates are
  data-dependent and learned; this is the constant-gate skeleton those
  live on, and Law 2 is exactly why constant gates are not a weak version
  of learned ones but a different mechanism.
- Nat-valued, so these are semiring identities. They say nothing about
  convergence, and inequalities over reals (e.g. `w` decaying when both
  `b, eta < 1`) are outside what is checked here. Law 4's monotonicity is
  stated as an exact increment, not as an inequality, for that reason.
- `--verdict` is deliberately not used: it re-checks with a Lean-backed
  kernel and gives up Bend's speed, which is the reason for using Bend.
