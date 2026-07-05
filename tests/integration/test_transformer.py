"""
Integration test: transformer attention with sili autograd vs PyTorch baseline.

Strategy: run identical architectures in both sili and PyTorch with shared
weight initialization. Compare MSE curves step-by-step.

If PyTorch and sili converge similarly -> architecture/task is the issue.
If PyTorch wins significantly -> sili backprop through attention needs fixing.

Architecture (both implementations):
  Input token [1,0] or [0,1]  (2D binary)
  Token K/V projection: K[t] = token[t] @ Wk, V[t] = token[t] @ Wv
    (K/V from input tokens, not energy-gated hidden -- ensures informative keys)
  RNN hidden: h_raw = [token, state] @ W + bW
  Energy gating (sili only; PyTorch uses tanh as equivalent nonlinearity)
  Q from hidden: q[t] = h_out[t] @ Wq
  Attention: softmax(Q @ K_hist^T / sqrt(d_k)) @ V_hist
  Output head: pred = h_att @ V_head + b_head (MSE regression)

Tasks:
  copy  -- 15-token repeating cycle; induction head attends back period steps
  rare  -- sparse events; attention provides wider context than RNN alone

Run:
  python -m tests.integration.test_transformer [--steps 2000] [--task copy]
"""

import argparse, math, zlib, warnings
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F

import sili.cpu
from sili.tensor import Tensor, tanh as sili_tanh
from sili.energy import EnergyDynamics


# ── Datasets ──────────────────────────────────────────────────────────────────

def gen_copy(T=150, period=15):
    sub = np.random.randint(0, 2, period)
    x   = np.zeros((T, 2), np.float32)
    for t in range(T): x[t, sub[t % period]] = 1.
    return x

def gen_rare(T=150):
    x = np.zeros((T, 2), np.float32); x[:, 0] = 1.
    k = np.random.randint(0, max(1, T // 2 + 1))
    cur, gap = k, k // 2; indices = [k]
    while cur + gap + 1 < T and gap > 0: cur += gap + 1; indices.append(cur); gap //= 2
    for i in indices: x[i] = [0., 1.]
    return x

GENS = dict(copy=gen_copy, rare=gen_rare)


def compression_ratio(bufs):
    if not bufs: return 1.0
    flat = np.concatenate(bufs).tobytes()
    return len(zlib.compress(flat, 1)) / max(len(flat), 1)


# ── Shared weight initialisation ───────────────────────────────────────────────

def make_weights(shapes: dict, scale=0.05, seed=0) -> dict:
    """Return dict of named numpy arrays initialised identically for both runs."""
    rng = np.random.default_rng(seed)
    return {k: (rng.standard_normal(s) * scale).astype(np.float32)
            for k, s in shapes.items()}


# ── sili implementation ────────────────────────────────────────────────────────

def run_sili(task, steps, hidden, d_k, window, lr, aux_weight,
             clip, report_every, init_weights,
             nonlinearity="energy", peaked_init=False):
    n_in = 2; n_out = 2

    W  = Tensor(init_weights['W'].copy())
    bW = Tensor(np.zeros(hidden, np.float32))
    Wq = Tensor(init_weights['Wq'].copy())
    Wo = Tensor(init_weights['Wo'].copy())
    Wk = init_weights['Wk'].copy()   # numpy: updated outside autograd
    Wv = init_weights['Wv'].copy()
    Vh = init_weights['Vh'].copy()   # output head weights
    bh = np.zeros(n_out, np.float32)
    sili_params = [W, bW, Wq, Wo]

    scale_att = math.sqrt(d_k)
    frac = max(0.1, min(0.4, 6. / hidden))
    energy = EnergyDynamics(drive=1./hidden, activation_cost=0.05,
                            precision=0.01, density=frac/2,
                            exploration=0.001, p=frac)

    if peaked_init:
        # Init Wq = Wk at larger scale so softmax peaks on same-token
        # positions from step 1 rather than waiting for gradient to sharpen.
        # Same-token matching: for bit=[1,0], K[i] = Wk[0]; for [0,1], K[i] = Wk[1].
        # Q = h_out @ Wq; with Wq initialized to project h_out onto token-like
        # directions, same-token positions get high dot product immediately.
        Wk[:] = init_weights['Wk'].copy() * 5.0   # larger scale -> sharper softmax
        Wv[:] = init_weights['Wv'].copy() * 5.0
        # Wq not changed: gradient from task loss shapes Q toward K

    mse_log = []

    for step in range(steps):
        np.random.seed(step)  # reproducible per-epoch sequence
        seq   = GENS[task]()
        state = np.zeros(hidden, np.float32)
        K_hist, V_hist = [], []
        mse_step = 0.; n_step = 0

        for t in range(len(seq) - 1):
            tok = seq[t]

            # K/V from input tokens
            K_hist.append(tok @ Wk)
            V_hist.append(tok @ Wv)
            if len(K_hist) > window: K_hist.pop(0); V_hist.pop(0)

            # RNN step
            xh    = Tensor(np.concatenate([tok, state]))
            h_raw = xh @ W + bW
            if nonlinearity == "tanh":
                h_out = sili_tanh(h_raw)
                aux   = None
            else:
                h_out, aux, _ = energy.forward(h_raw)

            # Attention: Q from hidden (in autograd), K/V from token history
            if len(K_hist) > 0:
                L    = len(K_hist)
                K_np = np.stack(K_hist, 0)
                V_np = np.stack(V_hist, 0)
                q    = h_out @ Wq                      # Tensor in graph

                scores = q.data @ K_np.T / scale_att
                scores -= scores.max()
                w = np.exp(scores); w /= w.sum()
                attended_np = w @ V_np

                attended = Tensor(attended_np, (q,), 'att', q.backend)
                def _bwd(_q=q, _K=K_np, _V=V_np, _w=w, _att=attended):
                    g  = np.asarray(_att.grad, np.float32)
                    dw = _V @ g
                    ds = _w * (dw - (_w * dw).sum())
                    dq = _K.T @ ds / scale_att
                    np.clip(dq, -clip, clip, out=dq)
                    if _q.grad is None: _q.grad = dq.copy()
                    else:               _q.grad += dq
                attended._backward = _bwd

                att_out = attended @ Wo
                h_att   = h_out + att_out
            else:
                h_att = h_out

            # Head (numpy SGD)
            pred  = h_att.data @ Vh + bh
            tgt   = seq[t + 1]
            err   = pred - tgt
            g_out = (2. / n_out) * err

            # Gradients through head -> attention -> energy -> W
            g_h   = g_out @ Vh.T
            Vh   -= lr * np.outer(h_att.data, g_out)
            bh   -= lr * g_out

            h_att.grad = g_h
            h_att.backward()

            if aux is not None:
                aux.grad = np.array([aux_weight], np.float32)
                aux.backward()

            for p in sili_params:
                if p.grad is not None:
                    np.clip(p.grad, -clip, clip, out=p.grad)
                    p.data -= lr * p.grad; p.grad = None

            state = h_att.data.copy()
            mse_step += float(np.mean(err**2)); n_step += 1

        mse_log.append(mse_step / max(1, n_step))

        if (step + 1) % report_every == 0:
            avg = float(np.mean(mse_log[-report_every:]))
            print(f"  [sili]   step {step+1:5d}  mse={avg:.6f}")

    return mse_log


# ── PyTorch implementation (identical architecture, tanh instead of energy) ──

def run_pytorch(task, steps, hidden, d_k, window, lr, clip,
                report_every, init_weights, peaked_init=False):
    n_in = 2; n_out = 2
    scale_att = math.sqrt(d_k)

    # Copy shared init weights into torch tensors (require_grad where needed)
    W  = torch.tensor(init_weights['W'].copy(), requires_grad=True)
    bW = torch.zeros(hidden, requires_grad=True)
    Wq = torch.tensor(init_weights['Wq'].copy(), requires_grad=True)
    Wo = torch.tensor(init_weights['Wo'].copy(), requires_grad=True)
    Wk = torch.tensor(init_weights['Wk'].copy())   # no grad: token-level
    Wv = torch.tensor(init_weights['Wv'].copy())
    Vh = torch.tensor(init_weights['Vh'].copy(), requires_grad=True)
    bh = torch.zeros(n_out, requires_grad=True)
    pt_params = [W, bW, Wq, Wo, Vh, bh]

    if peaked_init:
        Wk = torch.tensor(init_weights['Wk'].copy() * 5.0)
        Wv = torch.tensor(init_weights['Wv'].copy() * 5.0)

    mse_log = []

    for step in range(steps):
        np.random.seed(step)  # same seed as sili run for fair comparison
        seq   = GENS[task]()
        state = torch.zeros(hidden)
        K_hist, V_hist = [], []
        mse_step = 0.; n_step = 0

        for t in range(len(seq) - 1):
            tok = torch.tensor(seq[t])

            K_hist.append(tok @ Wk)
            V_hist.append(tok @ Wv)
            if len(K_hist) > window: K_hist.pop(0); V_hist.pop(0)

            xh    = torch.cat([tok, state])
            h_raw = xh @ W + bW
            h_out = torch.tanh(h_raw)         # tanh matches energy gating role

            if len(K_hist) > 0:
                K_t  = torch.stack(K_hist, 0)    # (L, d_k)
                V_t  = torch.stack(V_hist, 0)    # (L, d_k)
                q    = h_out @ Wq                # (d_k,)
                sc   = q @ K_t.T / scale_att
                w    = F.softmax(sc, dim=0)
                att  = w @ V_t                   # (d_k,)
                h_att = h_out + att @ Wo
            else:
                h_att = h_out

            pred  = h_att @ Vh + bh
            tgt   = torch.tensor(seq[t + 1])
            loss  = F.mse_loss(pred, tgt)

            for p in pt_params:
                if p.grad is not None: p.grad.zero_()
            loss.backward()

            for p in pt_params:
                if p.grad is not None:
                    p.grad.clamp_(-clip, clip)
                    with torch.no_grad(): p -= lr * p.grad

            state = h_att.detach()
            mse_step += float(loss.item()); n_step += 1

        mse_log.append(mse_step / max(1, n_step))

        if (step + 1) % report_every == 0:
            avg = float(np.mean(mse_log[-report_every:]))
            print(f"  [torch]  step {step+1:5d}  mse={avg:.6f}")

    return mse_log


# ── Comparison runner ─────────────────────────────────────────────────────────

def run(task='copy', steps=2000, hidden=32, d_k=8, window=20,
        lr=0.005, aux_weight=0.05, clip=1.0, report_every=400, seed=42,
        nonlinearity='energy', peaked_init=False):
    n_in = 2
    np.random.seed(seed); torch.manual_seed(seed)

    # Shared initialisation (identical starting point for both frameworks)
    init_scale = np.sqrt(2. / (n_in + hidden))
    shapes = {
        'W':  (n_in + hidden, hidden),
        'Wq': (hidden, d_k),
        'Wo': (d_k, hidden),
        'Wk': (n_in, d_k),
        'Wv': (n_in, d_k),
        'Vh': (hidden, 2),
    }
    init_weights = {}
    rng = np.random.default_rng(seed)
    for k, s in shapes.items():
        sc = init_scale if k == 'W' else 0.05
        init_weights[k] = (rng.standard_normal(s) * sc).astype(np.float32)

    period = 15
    print(f"\n{'='*60}")
    print(f"TRANSFORMER COMPARISON  task={task}  steps={steps}")
    print(f"hidden={hidden}  d_k={d_k}  window={window}  lr={lr}")
    print(f"sili: {nonlinearity} on hidden; torch: tanh on hidden")
    print(f"Both: K/V from input tokens; Q from hidden; shared init")
    if peaked_init: print("peaked_init: Wk/Wv scaled 5x for sharper softmax from step 1")
    if task == 'copy':
        print(f"Copy: induction head should learn to attend {period-1} steps back")
    print(f"{'='*60}")
    print(f"\n--- sili run ---")
    sili_mse = run_sili(task, steps, hidden, d_k, window, lr, aux_weight,
                        clip, report_every, init_weights,
                        nonlinearity=nonlinearity, peaked_init=peaked_init)

    print(f"\n--- PyTorch run ---")
    pt_mse = run_pytorch(task, steps, hidden, d_k, window, lr, clip,
                         report_every, init_weights, peaked_init=peaked_init)

    # Summary comparison
    sili_final = float(np.mean(sili_mse[-report_every:]))
    pt_final   = float(np.mean(pt_mse[-report_every:]))
    ratio      = sili_final / max(pt_final, 1e-9)
    print(f"\n{'='*60}")
    print(f"FINAL MSE (last {report_every} steps):")
    print(f"  sili:  {sili_final:.6f}")
    print(f"  torch: {pt_final:.6f}")
    print(f"  ratio: {ratio:.2f}x  ", end='')
    if ratio < 1.5:
        print("(sili matches torch -- architecture/task is the bottleneck)")
    elif ratio < 3.0:
        print("(sili slightly behind -- minor gradient differences)")
    else:
        print("(sili significantly behind -- backprop bug likely)")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",         default="copy", choices=list(GENS))
    ap.add_argument("--steps",        type=int,   default=2000)
    ap.add_argument("--hidden",       type=int,   default=32)
    ap.add_argument("--d-k",          type=int,   default=8)
    ap.add_argument("--window",       type=int,   default=20)
    ap.add_argument("--lr",           type=float, default=0.005)
    ap.add_argument("--aux-weight",   type=float, default=0.05)
    ap.add_argument("--report-every", type=int,   default=400)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--nonlinearity",  default="energy", choices=["energy","tanh"])
    ap.add_argument("--peaked-init",   action="store_true",
                        help="Scale Wk/Wv 5x for sharper softmax from step 1")
    a = ap.parse_args()
    run(a.task, a.steps, a.hidden, a.d_k, a.window, a.lr, a.aux_weight,
        1.0, a.report_every, a.seed,
        nonlinearity=a.nonlinearity, peaked_init=a.peaked_init)

if __name__ == "__main__":
    main()
