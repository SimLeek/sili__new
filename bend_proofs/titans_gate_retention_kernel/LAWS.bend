import Base

# LAWS: what the three Titans gates (alpha = forgetting, eta = momentum,
# theta = inner step size) can and cannot do to the memory's RETENTION
# KERNEL -- the weight the memory assigns to a gradient written n ticks
# ago. See README.md for the derivation and for the two measurement
# confounds these laws formalize.
#
# Setup. The Titans inner loop (arXiv:2501.00663 eqs 10/13) is
#   S_t = eta * S_{t-1} - theta * g_t
#   M_t = (1 - alpha) * M_{t-1} + S_t
# Writing b = 1 - alpha and unrolling both recurrences gives
#   M_T = -theta * SUM_i w(T - i) * g_i,     w(n) = SUM_{u=0..n} b^(n-u) eta^u
# so the ENTIRE temporal behaviour of the gate pair is carried by w.
# Kernel.w below is that w, defined by the recurrence it satisfies
# (w(0) = 1, w(n+1) = b*w(n) + eta^(n+1)), which expands to exactly the
# sum above. Everything is Nat-valued: w is a polynomial in b and eta
# with Nat coefficients, so it lives in a commutative semiring and Bend
# can check it without real-number arithmetic -- the same trick that
# made synaptic_hold_bptt_approximation checkable.
#
# Kernel.geo(e, n) = SUM_{u=0..n} e^u is the plain geometric sum, used to
# state the alpha = 0 case.

def Kernel.pow(+b: Nat, n: Nat) -> Nat:
  match n:
    case 0n:
      1n
    case 1n+p:
      Nat.mul(b, Kernel.pow(b, p))

def Kernel.w(+b: Nat, +e: Nat, n: Nat) -> Nat:
  match n:
    case 0n:
      1n
    case 1n++p:
      Nat.add(Nat.mul(b, Kernel.w(b, e, p)), Kernel.pow(e, 1n+p))

def Kernel.geo(+e: Nat, n: Nat) -> Nat:
  match n:
    case 0n:
      1n
    case 1n++p:
      Nat.add(Kernel.geo(e, p), Kernel.pow(e, 1n+p))

# LAW 1 (the exchanged recurrence). w satisfies its own defining
# recurrence with the roles of b and eta SWAPPED: stepping the kernel by
# one tick can be charged either to forgetting or to momentum, and the
# result is identical. Stated with the exchanged form on the LEFT because
# that is the orientation a direct induction closes (Bend rewrites
# right-to-left, so the definitional form has to be the rewritable side).
law kernel_recurrence_exchanged:
  for +b: Nat
  for +e: Nat
  for +n: Nat
  {Nat.add(Nat.mul(e, Kernel.w(b, e, n)), Kernel.pow(b, 1n+n)) == Kernel.w(b, e, 1n+n) : Nat}

# LAW 2, THE MAIN THEOREM (gate symmetry). The retention kernel is
# SYMMETRIC in (1 - alpha) and eta. Forgetting and momentum are therefore
# the SAME knob as far as the memory's temporal profile is concerned: any
# kernel reachable by tuning alpha is reachable by tuning eta and vice
# versa, so the pair contributes ONE effective degree of freedom to
# retention shape, not two.
#
# Consequence for adapting the gates to a task: a controller that tunes
# alpha and eta as independent knobs is over-parameterized, and the
# practical asymmetry between them must come from something OUTSIDE this
# kernel -- namely that alpha also scales the CURRENT write while eta
# does not, and that g_i itself depends on M (the update is nonlinear).
law kernel_symmetry:
  for +b: Nat
  for +e: Nat
  for n: Nat
  {Kernel.w(b, e, n) == Kernel.w(e, b, n) : Nat}

# LAW 3 (why eta silently rescales the learning rate). With NO forgetting
# (alpha = 0, so b = 1) the kernel is exactly the geometric sum in eta.
# Its total mass is therefore the geometric series 1/(1 - eta), which is
# the hidden coupling: raising eta multiplies the effective step by
# 1/(1 - eta), so comparing eta values at fixed theta compares two
# different learning rates and not two momenta. Measured consequence of
# missing this: ||M|| reached 11093 at eta=0.95 where the normalized run
# stayed at 4.94.
law kernel_no_forget_is_geometric:
  for +e: Nat
  for n: Nat
  {Kernel.w(1n, e, n) == Kernel.geo(e, n) : Nat}

# LAW 4 (why momentum alone privileges OLD writes). With no forgetting the
# kernel is non-decreasing in age n: w(n+1) exceeds w(n) by exactly
# eta^(n+1). So alpha = 0 weights a write at least as heavily the older it
# gets -- momentum without forgetting is a low-pass filter favouring EARLY
# content, and forgetting is the only source of recency bias in the pair.
# Measured consequence of missing this: a task that wrote its probed pairs
# FIRST handed momentum free credit (0.747 at eta=0.95), while the same
# gates scored 0.119 once the probes were written LAST.
law kernel_age_monotone_without_forgetting:
  for +e: Nat
  for +n: Nat
  {Kernel.w(1n, e, 1n+n) == Nat.add(Kernel.w(1n, e, n), Kernel.pow(e, 1n+n)) : Nat}
