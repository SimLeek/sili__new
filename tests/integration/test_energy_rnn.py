"""
Integration test: EnergyDynamics + sili autograd RNN.

Two separate test categories:

  PASS-THROUGH (--task pass_through)
    Prove gradient flow works: predict current input from current input (MSE).
    Input can be anything -- even random numbers -- and the model should still
    converge. If it doesn't, something is broken in the gradient stack.
    This is NOT a temporal task; it is a wiring check.

  TEMPORAL PREDICTION (--task sine|rare|copy)
    Next-step prediction with MSE loss. The model must learn dynamics.
    Cross-entropy / classification is NOT used here -- these are regression
    tasks on binary sequences, not labeled classification problems.
    char is excluded: next digit's even/odd is independent of current
    (random digits), so the next-step ceiling is ~50% -- not useful.

    sine  -- predict [cos(t+1), sin(t+1)] from [cos(t), sin(t)]
             deterministic; optimal solution is a linear map
    rare  -- predict next [1,0] or [0,1] in geometrically-spaced event sequence
             sparse signal; energy excels here (intrinsic curiosity)
    copy  -- predict next bit in repeating 15-bit sub-sequence
             model needs positional memory of cycle

Architecture:
  - EnergyDynamics on HIDDEN neurons only (not output)
  - Task gradient: DenseHead.backward() -> h_out.grad -> h_out.backward()
    propagates through the energy gate to the sili Tensor hidden weights
  - aux_loss (energy homeostasis) provides an extra gradient path through
    fired neurons whose gate blocks the task gradient

Init modes:
  normal  -- random weights
  zero    -- all-zero weights; energy pre-set to 1.9 to escape dead zone

Run:
  python -m examples.sanity_checks --task pass_through --init zero
  python -m examples.sanity_checks --task rare --steps 3000
"""

import argparse, math, zlib
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import sili.cpu
from sili.tensor import Tensor
from sili.energy import EnergyDynamics


# ── Datasets ──────────────────────────────────────────────────────────────────

def gen_pass_through(T=150):
    """Bounded random 2D input; target is the same input (same timestep).
    Uses uniform[-1,1] to avoid RNN state divergence from large normal values.
    Even completely random sequences converge -- the model just learns identity.
    Proves gradient wiring works regardless of input structure."""
    return np.random.uniform(-1., 1., (T, 2)).astype(np.float32)

def gen_sine(T=150):
    phi = np.random.uniform(0, 2*math.pi)
    t   = np.arange(T) * 0.1
    return np.stack([np.cos(t+phi), np.sin(t+phi)], 1).astype(np.float32)

def gen_rare(T=150):
    """Geometrically-spaced rare events. Most steps are [1,0]; event steps [0,1].
    Next-step prediction forces the model to track the halving-interval pattern."""
    x = np.zeros((T,2), np.float32); x[:,0] = 1.
    k = np.random.randint(0, max(1, T//2+1))
    cur, gap = k, k//2; indices=[k]
    while cur+gap+1 < T and gap > 0: cur+=gap+1; indices.append(cur); gap//=2
    for i in indices: x[i] = [0., 1.]
    return x

def gen_copy(T=150):
    """Repeating 15-bit sub-sequence. Next-step prediction requires positional
    memory of where in the cycle the sequence currently is."""
    sub = np.random.randint(0, 2, 15)
    x   = np.zeros((T,2), np.float32)
    for t in range(T): x[t, sub[t%15]] = 1.
    return x

GENS = dict(pass_through=gen_pass_through, sine=gen_sine, rare=gen_rare, copy=gen_copy)

# Tasks using current-step target (pass-through) vs next-step (temporal)
NEXT_STEP_TASKS = {'sine', 'rare', 'copy'}


def compression_ratio(bufs):
    if not bufs: return 1.0
    flat = np.concatenate(bufs).tobytes()
    return len(zlib.compress(flat, 1)) / max(len(flat), 1)


class DenseHead:
    """Plain dense output head -- no energy applied here.
    backward() returns gradient to h so it propagates to sili Tensor weights."""
    def __init__(self, in_f, out_f, lr):
        s = np.sqrt(2./in_f)
        self.w = np.random.randn(in_f, out_f).astype(np.float32) * s
        self.b = np.zeros(out_f, np.float32)
        self.lr = lr; self._h = None

    def zero_init(self): self.w[:] = 0.; self.b[:] = 0.

    def forward(self, h_np):
        self._h = h_np.copy()
        return h_np @ self.w + self.b

    def backward(self, g_out):
        self.w -= self.lr * np.outer(self._h, g_out)
        self.b -= self.lr * g_out
        return g_out @ self.w.T   # gradient back to h for propagation to W


# ── Training ──────────────────────────────────────────────────────────────────

def run(task='rare', steps=2000, hidden=32, lr=0.01,
        aux_weight=0.05, clip=1.0, init='normal', report_every=200):
    T   = 150
    n_in  = 2
    n_out = 2   # all tasks output 2D (MSE regression, not classification)

    # Active fraction: target ~6 neurons minimum for gradient coverage
    frac    = max(0.1, min(0.4, 6./hidden))
    k_active = max(1, round(frac*hidden))
    energy  = EnergyDynamics(
        drive=1./hidden, activation_cost=0.05,
        precision=0.01, density=frac/2,
        exploration=0.001, p=frac)

    print(f"\n=== task={task}  init={init}  hidden={hidden}  k_active={k_active} ===")
    print(f"energy_start={energy._energy_start:.2f}  (threshold 2.0)")
    is_temporal = task in NEXT_STEP_TASKS
    print(f"{'next-step prediction (temporal)' if is_temporal else 'pass-through check (gradient wiring)'}")

    scale = np.sqrt(2./(n_in+hidden))
    if init == 'zero':
        # Zero the HIDDEN layer weights only.
        # Head stays random-initialized so the task gradient (g_h = g_out @ w.T)
        # is non-zero and can bootstrap W from the start.
        # Zeroing the head as well would make g_h=0 -- no task gradient ever
        # reaches W, so only aux_loss acts on W (homeostasis, not task signal).
        W  = Tensor(np.zeros((n_in+hidden, hidden), np.float32))
        bW = Tensor(np.zeros(hidden, np.float32))
        # Pre-charge energy near threshold: with W=0, neurons get no input
        # signal, but pre-charging to 1.9 means the FIRST nonzero activation
        # from aux_loss bootstrapping will immediately trigger firing.
        energy.energy = np.full(hidden, 1.9, np.float32)
    else:
        W  = Tensor(np.random.randn(n_in+hidden, hidden).astype(np.float32)*scale)
        bW = Tensor(np.zeros(hidden, np.float32))

    head = DenseHead(hidden, n_out, lr)  # always random-init for task gradient
    params = [W, bW]

    mse_sum = 0.; n_steps = 0
    hid_buf = []

    for step in range(steps):
        seq   = GENS[task](T)
        state = np.zeros(hidden, np.float32)

        T_range = T-1 if is_temporal else T

        for t in range(T_range):
            xh    = Tensor(np.concatenate([seq[t], state]))
            h_raw = xh @ W + bW              # sili autograd
            h_out, aux, _ = energy.forward(h_raw)

            pred = head.forward(h_out.data)

            # MSE loss (regression, not classification)
            # pass_through: predict current step
            # temporal tasks: predict next step
            tgt = seq[t+1] if is_temporal else seq[t]
            err  = pred - tgt
            g_out = (2./n_out) * err         # dL/dpred

            # Gradient: head update + propagate back through energy to W
            g_h = head.backward(g_out)
            h_out.grad = g_h
            h_out.backward()                 # propagates to W and bW

            # Energy aux_loss: homeostasis gradient (reaches fired neurons)
            if aux is not None:
                aux.grad = np.array([aux_weight], np.float32)
                aux.backward()

            for p in params:
                if p.grad is not None:
                    np.clip(p.grad, -clip, clip, out=p.grad)
                    p.data -= lr * p.grad; p.grad = None

            state = h_out.data.copy()

            mse_sum += float(np.mean(err**2)); n_steps += 1

            hid_buf.append(state.copy())
            if len(hid_buf) > 200: hid_buf.pop(0)

        if (step+1) % report_every == 0:
            avg_mse = mse_sum / max(1, n_steps)
            print(f"  step {step+1:5d}  mse={avg_mse:.6f}  "
                  f"compress={compression_ratio(hid_buf):.3f}")
            mse_sum = 0.; n_steps = 0

    print("Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",         default="rare", choices=list(GENS))
    ap.add_argument("--init",         default="normal", choices=["normal","zero"])
    ap.add_argument("--steps",        type=int,   default=2000)
    ap.add_argument("--hidden",       type=int,   default=32)
    ap.add_argument("--lr",           type=float, default=0.01)
    ap.add_argument("--aux-weight",   type=float, default=0.05)
    ap.add_argument("--report-every", type=int,   default=200)
    a = ap.parse_args()
    run(a.task, a.steps, a.hidden, a.lr, a.aux_weight,
        init=a.init, report_every=a.report_every)

if __name__ == "__main__":
    main()
