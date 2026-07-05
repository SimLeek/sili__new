"""
Integration test: transformer attention with sili autograd vs PyTorch baseline.

Two separate comparisons:

  A) sili-tanh vs PyTorch-tanh  [FAIR -- exactly the same architecture]
     Only this comparison tests whether sili's backprop through attention is
     correct. Both frameworks use tanh as the hidden nonlinearity.
     PyTorch cannot replicate the sili energy function (discrete gating + abs
     backward + KL aux loss), so energy variants are NOT compared to PyTorch.
     Ratio > 3x on this comparison indicates a sili backprop bug.

  B) sili nonlinearity comparison  [sili-only, no PyTorch needed]
     Compares tanh vs energy vs energy+tanh within sili.
     energy+tanh: tanh is the smooth RNN activation; energy gates on top for
     homeostatic sparsity and curiosity signal via aux_loss.
     PyTorch equivalent does not exist; compare against sili-tanh as baseline.

Tasks:
  copy  -- 15-token repeating cycle; induction head
  rare  -- sparse events; attention provides wider context

Run:
  python -m tests.integration.test_transformer --mode compare --task copy
  python -m tests.integration.test_transformer --mode sili_only --task copy
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

def make_init_weights(n_in, hidden, d_k, seed=42):
    """Shared numpy arrays used to init BOTH sili and PyTorch identically."""
    rng   = np.random.default_rng(seed)
    sc_W  = np.sqrt(2. / (n_in + hidden))
    sc_att = 0.05
    return {
        'W':  (rng.standard_normal((n_in + hidden, hidden)) * sc_W).astype(np.float32),
        'Wq': (rng.standard_normal((hidden, d_k))            * sc_att).astype(np.float32),
        'Wo': (rng.standard_normal((d_k, hidden))            * sc_att).astype(np.float32),
        'Wk': (rng.standard_normal((n_in, d_k))              * sc_att).astype(np.float32),
        'Wv': (rng.standard_normal((n_in, d_k))              * sc_att).astype(np.float32),
        'Vh': (rng.standard_normal((hidden, 2))              * sc_att).astype(np.float32),
    }


# ── Core attention forward/backward (shared logic) ────────────────────────────

def _sili_attention_step(h_out, K_hist, V_hist, Wq, Wo, scale_att, clip):
    """Run one attention step in sili autograd. Returns h_att Tensor."""
    if not K_hist:
        return h_out
    L    = len(K_hist)
    K_np = np.stack(K_hist, 0)
    V_np = np.stack(V_hist, 0)
    q    = h_out @ Wq
    scores = q.data @ K_np.T / scale_att
    scores -= scores.max()
    w    = np.exp(scores); w /= w.sum()
    att_np = w @ V_np
    att  = Tensor(att_np, (q,), 'att', q.backend)
    def _bwd(_q=q, _K=K_np, _V=V_np, _w=w, _a=att):
        g  = np.asarray(_a.grad, np.float32)
        dw = _V @ g
        ds = _w * (dw - (_w * dw).sum())
        dq = _K.T @ ds / scale_att
        np.clip(dq, -clip, clip, out=dq)
        if _q.grad is None: _q.grad = dq.copy()
        else:                _q.grad += dq
    att._backward = _bwd
    att_out = att @ Wo
    return h_out + att_out


# ── sili run ──────────────────────────────────────────────────────────────────

def run_sili(task, steps, hidden, d_k, window, lr, aux_weight, clip,
             report_every, init_weights, nonlinearity='tanh', label='sili'):
    """
    nonlinearity: 'tanh'        -- tanh on h_raw (matches PyTorch exactly)
                  'energy'      -- EnergyDynamics gate on h_raw
                  'energy_tanh' -- tanh first, then EnergyDynamics gate on top
    """
    n_in = 2; n_out = 2
    scale_att = math.sqrt(d_k)

    W  = Tensor(init_weights['W'].copy())
    bW = Tensor(np.zeros(hidden, np.float32))
    Wq = Tensor(init_weights['Wq'].copy())
    Wo = Tensor(init_weights['Wo'].copy())
    Wk = init_weights['Wk'].copy()
    Wv = init_weights['Wv'].copy()
    Vh = init_weights['Vh'].copy()
    bh = np.zeros(n_out, np.float32)
    sili_params = [W, bW, Wq, Wo]

    use_energy = nonlinearity in ('energy', 'energy_tanh')
    frac   = max(0.1, min(0.4, 6. / hidden))
    energy = EnergyDynamics(drive=1./hidden, activation_cost=0.05,
                            precision=0.01, density=frac/2,
                            exploration=0.001, p=frac) if use_energy else None

    mse_log = []
    for step in range(steps):
        np.random.seed(step)
        seq   = GENS[task]()
        state = np.zeros(hidden, np.float32)
        K_hist, V_hist = [], []
        mse_step = 0.; n_step = 0

        for t in range(len(seq) - 1):
            tok   = seq[t]
            K_hist.append(tok @ Wk); V_hist.append(tok @ Wv)
            if len(K_hist) > window: K_hist.pop(0); V_hist.pop(0)

            xh    = Tensor(np.concatenate([tok, state]))
            h_raw = xh @ W + bW

            if nonlinearity == 'tanh':
                h_out = sili_tanh(h_raw)
                aux   = None
            elif nonlinearity == 'energy':
                h_out, aux, _ = energy.forward(h_raw)
            else:  # energy_tanh
                h_tanh        = sili_tanh(h_raw)
                h_out, aux, _ = energy.forward(h_tanh)

            h_att = _sili_attention_step(h_out, K_hist, V_hist, Wq, Wo, scale_att, clip)
            pred  = h_att.data @ Vh + bh
            tgt   = seq[t + 1]
            err   = pred - tgt
            g_out = (2. / n_out) * err

            g_h  = g_out @ Vh.T
            Vh  -= lr * np.outer(h_att.data, g_out)
            bh  -= lr * g_out
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
            print(f"  [{label:14s}]  step {step+1:5d}  mse={avg:.6f}")

    return mse_log


# ── PyTorch run (tanh only -- energy not replicable in PyTorch) ───────────────

def run_pytorch(task, steps, hidden, d_k, window, lr, clip,
                report_every, init_weights):
    """
    Uses tanh as the hidden nonlinearity (same as sili 'tanh' mode).
    PyTorch cannot replicate the sili EnergyDynamics (discrete gating, abs
    backward, KL aux loss), so this is only run for the fair tanh comparison.
    """
    n_in = 2; n_out = 2
    scale_att = math.sqrt(d_k)

    W  = torch.tensor(init_weights['W'].copy(),  requires_grad=True)
    bW = torch.zeros(hidden,  requires_grad=True)
    Wq = torch.tensor(init_weights['Wq'].copy(), requires_grad=True)
    Wo = torch.tensor(init_weights['Wo'].copy(), requires_grad=True)
    Wk = torch.tensor(init_weights['Wk'].copy())
    Wv = torch.tensor(init_weights['Wv'].copy())
    Vh = torch.tensor(init_weights['Vh'].copy(), requires_grad=True)
    bh = torch.zeros(n_out, requires_grad=True)
    pt_params = [W, bW, Wq, Wo, Vh, bh]

    mse_log = []
    for step in range(steps):
        np.random.seed(step)
        seq   = GENS[task]()
        state = torch.zeros(hidden)
        K_hist, V_hist = [], []
        mse_step = 0.; n_step = 0

        for t in range(len(seq) - 1):
            tok = torch.tensor(seq[t])
            K_hist.append(tok @ Wk); V_hist.append(tok @ Wv)
            if len(K_hist) > window: K_hist.pop(0); V_hist.pop(0)

            xh    = torch.cat([tok, state])
            h_raw = xh @ W + bW
            h_out = torch.tanh(h_raw)   # identical to sili 'tanh' mode

            if K_hist:
                K_t = torch.stack(K_hist, 0)
                V_t = torch.stack(V_hist, 0)
                q   = h_out @ Wq
                w   = F.softmax(q @ K_t.T / scale_att, dim=0)
                h_att = h_out + (w @ V_t) @ Wo
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
            print(f"  [torch (tanh)  ]  step {step+1:5d}  mse={avg:.6f}")

    return mse_log


# ── Runners ───────────────────────────────────────────────────────────────────

def run_compare(task, steps, hidden, d_k, window, lr, aux_weight,
                clip, report_every, seed):
    """
    Mode A: sili-tanh vs PyTorch-tanh (exactly identical architectures).
    Tests sili backprop correctness. Ratio > 3x = likely sili bug.
    """
    init = make_init_weights(n_in=2, hidden=hidden, d_k=d_k, seed=seed)
    print(f"\n{'='*62}")
    print(f"MODE: sili-tanh vs PyTorch-tanh  [IDENTICAL ARCHITECTURE]")
    print(f"task={task}  steps={steps}  hidden={hidden}  d_k={d_k}  window={window}")
    print(f"{'='*62}")

    print("\n--- sili (tanh) ---")
    sili_mse = run_sili(task, steps, hidden, d_k, window, lr, aux_weight,
                        clip, report_every, init, nonlinearity='tanh',
                        label='sili (tanh)')

    print("\n--- PyTorch (tanh) ---")
    pt_mse = run_pytorch(task, steps, hidden, d_k, window, lr, clip,
                         report_every, init)

    sili_f = float(np.mean(sili_mse[-report_every:]))
    pt_f   = float(np.mean(pt_mse[-report_every:]))
    ratio  = sili_f / max(pt_f, 1e-9)
    print(f"\n{'='*62}")
    print(f"sili (tanh):   {sili_f:.6f}")
    print(f"torch (tanh):  {pt_f:.6f}")
    print(f"ratio: {ratio:.2f}x  ", end='')
    if   ratio < 1.5: print("-> architecture/task is the bottleneck (backprop ok)")
    elif ratio < 3.0: print("-> minor gradient difference, worth investigating")
    else:             print("-> LIKELY SILI BACKPROP BUG, investigate")
    print(f"{'='*62}\n")


def run_sili_only(task, steps, hidden, d_k, window, lr, aux_weight,
                  clip, report_every, seed):
    """
    Mode B: tanh vs energy vs energy+tanh within sili.
    No PyTorch equivalent for energy variants.
    """
    init = make_init_weights(n_in=2, hidden=hidden, d_k=d_k, seed=seed)
    print(f"\n{'='*62}")
    print(f"MODE: sili nonlinearity comparison  [sili-only]")
    print(f"task={task}  steps={steps}  hidden={hidden}  d_k={d_k}  window={window}")
    print(f"PyTorch cannot replicate EnergyDynamics (discrete gating,")
    print(f"abs backward, KL aux loss) -- compare against sili-tanh baseline")
    print(f"{'='*62}")

    results = {}
    for nl in ('tanh', 'energy', 'energy_tanh'):
        print(f"\n--- sili ({nl}) ---")
        mse = run_sili(task, steps, hidden, d_k, window, lr, aux_weight,
                       clip, report_every, init, nonlinearity=nl,
                       label=f'sili ({nl})')
        results[nl] = float(np.mean(mse[-report_every:]))

    print(f"\n{'='*62}")
    baseline = results['tanh']
    for nl, val in results.items():
        diff = f"{val/baseline:.2f}x baseline"
        print(f"  sili ({nl:12s}): {val:.6f}  ({diff})")
    print(f"{'='*62}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",         default="compare",
                    choices=["compare", "sili_only"],
                    help="compare=sili-tanh vs torch-tanh; sili_only=tanh/energy/energy_tanh")
    ap.add_argument("--task",         default="copy", choices=list(GENS))
    ap.add_argument("--steps",        type=int,   default=800)
    ap.add_argument("--hidden",       type=int,   default=32)
    ap.add_argument("--d-k",          type=int,   default=8)
    ap.add_argument("--window",       type=int,   default=20)
    ap.add_argument("--lr",           type=float, default=0.005)
    ap.add_argument("--aux-weight",   type=float, default=0.05)
    ap.add_argument("--report-every", type=int,   default=400)
    ap.add_argument("--seed",         type=int,   default=42)
    a = ap.parse_args()

    kwargs = dict(task=a.task, steps=a.steps, hidden=a.hidden, d_k=a.d_k,
                  window=a.window, lr=a.lr, aux_weight=a.aux_weight,
                  clip=1.0, report_every=a.report_every, seed=a.seed)
    if a.mode == "compare":
        run_compare(**kwargs)
    else:
        run_sili_only(**kwargs)

if __name__ == "__main__":
    main()
