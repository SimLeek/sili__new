# What the three Titans gates can and cannot do

A machine-checked account of the **retention kernel** behind the Titans
neural long-term memory (arXiv:2501.00663 §3.1), written because two
measurement confounds in the empirical work each reversed a conclusion,
and both turn out to be visible in one line of algebra.

Run `bend PROOF.bend --check-only` (prints `ALL PROOFS CHECK`; five laws). The Nat
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

`w` is the complete homogeneous symmetric polynomial `h_n(b, eta)`, so it
depends on the two gates **only through their symmetric functions** (sum
and product). The gates are therefore **interchangeable, but not
redundant**: the kernel cannot tell which gate produced a given decay, yet
both still matter. That is 2 degrees of freedom with the *labeling*
removed, not 1.

(An earlier version of this file claimed "one effective degree of
freedom". That was wrong. Check: `b=1, eta=0.95` is equivalent to
`b=0.95, eta=1` -- i.e. `alpha=0.05` **with** `eta=1`, not `alpha=0.05`
with `eta=0`. So the large measured gap between those two settings is real
and fully consistent with this law.)

The practical consequence for gate adaptation: a controller need not
decide whether a job belongs to `alpha` or to `eta`, only what the pair
is. Any genuine asymmetry between them must come from something *outside*
this kernel -- and `g_i` depending on `M` is the real candidate, since
within the linear model the kernel is the complete story.

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

**Law 5, `kernel_diagonal` -- the knob-free rule.** On the diagonal
`b = eta` the kernel collapses to a closed form:

```
w(b, b, n) == (n + 1) * b^n
```

This is the **only setting with an interior peak**. Either gate alone
gives a kernel that merely rises and saturates (`alpha = 0`, Law 4) or
merely decays, so neither can express "weight writes of age about L most
heavily". The diagonal can: `(n+1) b^n` peaks at `n* = -1/ln b - 1 ~=
1/(1-b)`, with total mass `1/(1-b)^2`.

Inverting the peak against a **measured** mean retrieval lag `L` therefore
fixes every gate with nothing left over:

```
eta = 1 - 1/L        (peak at age L)
theta *= (1 - eta)   (cancel Law 3's trace mass, the real step multiplier)
alpha = 1/L          (the diagonal) -- but see the caveat below
```

`L` comes from the input stream, so this is derived rather than tuned.
Verified numerically: the closed form matches `w` to 1e-15 and `argmax_n`
lands at `L` for `L` in {2, 4, 8, 16, 32, 64}.

## What this says about adapting the gates

The empirical work found that the two gates trade off along a *temporal*
axis — momentum broadens retention, weight decay sharpens it toward
recent — and that each is clearly beneficial in the regime it targets
(momentum 0.665 -> 0.871 with probes spread uniformly; forgetting 0.387 ->
0.879 with probes recent). Laws 2-4 explain why that is the shape of the
result rather than a coincidence:

- Law 3 says `theta` must absorb the **trace** mass `1/(1 - eta)`, and
  only that. `alpha` attenuates already-stored content, which is retention
  *shape*, not step magnitude; cancelling the full kernel mass
  `1/((1-b)(1-eta))` conflates the two, and doing so measured 30x worse
  until the error was found.
- Law 4 says `alpha` is what buys recency, so the quantity a controller
  should sense is **how far back the task's retrievals actually reach** —
  observable from the stream's own queries, no oracle needed.
- Law 2 says the controller never has to attribute a job to one gate
  rather than the other, only to fix the pair.

**Measured, with no knob.** `eta = 1 - 1/L`, `theta *= (1-eta)`,
`alpha = 0` matches or beats the per-regime *tuned* constant in all three
probe-order regimes (0.8639 vs 0.8706 uniform; 0.9803 vs 0.7475
early-probe, i.e. it beats the tuned reference; 1.0000 vs 1.0000
recent-probe). Across a 32x range of filler gap its worst-case regret is
0.06, against 0.158 for a fixed `eta = 0` and 0.105 for a fixed
`eta = 0.95` -- every fixed value is excellent at one end of the range and
poor at the other, while the derived rule tracks `eta*` from 0.82 to 0.99
on its own.

**Caveat, and an honest limit of Law 5.** `alpha = 1/L` -- the actual
diagonal -- measured *worse* than `alpha = 0` (0.8269 vs 0.8639 uniform;
0.8550 vs 1.0000 recent-probe). Nothing in those tasks ever becomes
obsolete, so sharpening toward recent can only discard something still
wanted. Law 5's role is to supply the *invertible* form -- the one whose
peak can be solved for -- not to assert the diagonal is the right
operating point. `alpha` has to be earned by measured obsolescence, which
this harness does not provide.

Also unsettled: Law 2 holds for the linearized kernel, and the real
update's `g_i` depends on `M`. Whether the measured asymmetry between the
gates is fully explained by that nonlinearity is not answered here.

## Why the old tuned horizon was finicky, and this is not

The predecessor controller used `eta = eta_max * clip(L / horizon)`, whose
`horizon` had to be tuned and was sharply sensitive: 16 -> 48 swung recall
0.86 -> 0.45. The laws say why. Law 3 makes `1/(1-eta)` a *time constant*,
so `eta` near 1 is where all the dynamic range lives; a **linear** map
into it is badly conditioned by construction. `eta = 1 - 1/L` is the
correctly conditioned inverse, which is why removing the knob also removed
the sensitivity. That is the general lesson: the proof supplies the units
the knob should have been measured in, and once it is in those units there
is no knob.

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
