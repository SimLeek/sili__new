"""
Curiosity-driven Mandelbrot exploration via energy dynamics.

No external reward: homeostatic energy provides intrinsic curiosity.
Actions: pan_left, pan_right, pan_up, pan_down, zoom_in, zoom_out, reset.
The reset action lets the agent escape uninteresting (flat) regions.

Learning metric: zlib compression of recent views AND hidden states.
Lower compression ratio = more complex/varied content explored.

Run: python -m examples.mandelbrot_rl [--steps 3000] [--view 32]
"""

import argparse, zlib
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import sili.cpu
from sili.tensor import Tensor
from sili.energy import EnergyDynamics


# ── Mandelbrot ────────────────────────────────────────────────────────────────

def render_mandelbrot(cx, cy, zoom, size=32, max_iter=64):
    x = np.linspace(cx - size/(2*zoom), cx + size/(2*zoom), size)
    y = np.linspace(cy - size/(2*zoom), cy + size/(2*zoom), size)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    Z = np.zeros_like(C)
    out = np.zeros((size, size), dtype=np.float32)
    for _ in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        out += mask.astype(np.float32)
    return out / max_iter


def compression_ratio(bufs):
    if not bufs: return 1.0
    flat = np.concatenate([b.ravel() for b in bufs]).tobytes()
    return len(zlib.compress(flat, 1)) / max(len(flat), 1)


# ── Agent ─────────────────────────────────────────────────────────────────────

class DenseHead:
    def __init__(self, in_f, out_f, lr=0.005):
        s = np.sqrt(2.0 / in_f)
        self.w = np.random.randn(in_f, out_f).astype(np.float32) * s
        self.b = np.zeros(out_f, np.float32)
        self.lr = lr; self._h = None

    def forward(self, h):
        self._h = h.copy()
        return h @ self.w + self.b

    def backward(self, g):
        self.w -= self.lr * np.outer(self._h, g)
        self.b -= self.lr * g
        return g @ self.w.T


def run(steps=3000, hidden=64, view=32, lr=0.003, report_every=300):
    N_ACT = 7  # pan x4, zoom x2, reset
    ACT_NAMES = ["pan<","pan>","pan^","panv","zoom+","zoom-","reset"]
    vf = view * view

    # Linear layers as sili Tensors (in autograd graph for gradient flow)
    scale = np.sqrt(2.0 / (vf + hidden))
    W  = Tensor(np.random.randn(vf + hidden, hidden).astype(np.float32) * scale)
    bW = Tensor(np.zeros(hidden, np.float32))
    params = [W, bW]

    # Active fraction: ~4 neurons min, up to 25% for better gradient coverage
    frac = max(0.05, min(0.25, 4.0 / hidden))
    energy = EnergyDynamics(drive=1./hidden, activation_cost=0.05,
        precision=0.01, density=frac/2, exploration=0.002, p=frac)
    print(f"energy_start={energy._energy_start:.2f} k_active~{max(1,round(frac*hidden))}")

    action_head = DenseHead(hidden, N_ACT, lr=lr)
    pred_head   = DenseHead(hidden, vf,    lr=lr)

    cx, cy = -0.75, 0.0   # boundary area with rich structure
    zoom = 50.0
    HOME = (cx, cy, zoom)

    state = np.zeros(hidden, np.float32)
    view_buf = []; hidden_buf = []
    act_counts = np.zeros(N_ACT, int)
    total_recon = 0.0

    print(f"\n=== Mandelbrot RL  hidden={hidden}  view={view}x{view} ===")
    print(f"Starting at ({cx:.2f},{cy:.2f}) zoom={zoom:.0f}")
    print(f"7 actions: {ACT_NAMES}")

    for step in range(steps):
        view_arr = render_mandelbrot(cx, cy, zoom, view)
        vflat    = view_arr.ravel().astype(np.float32)

        # -- sili autograd forward --
        inp   = Tensor(np.concatenate([vflat, state]))
        h_raw = inp @ W + bW
        h_out, aux, _ = energy.forward(h_raw)

        # -- action (dense head, softmax sample) --
        logits = action_head.forward(h_out.data)
        lg = logits - logits.max()
        probs = np.exp(lg) / np.exp(lg).sum()
        action = int(np.random.choice(N_ACT, p=probs))
        act_counts[action] += 1

        # -- apply action --
        pan = 0.3 / zoom
        if   action == 0: cx -= pan
        elif action == 1: cx += pan
        elif action == 2: cy -= pan
        elif action == 3: cy += pan
        elif action == 4: zoom *= 1.4
        elif action == 5: zoom /= 1.4
        elif action == 6: cx, cy, zoom = HOME   # reset
        zoom = np.clip(zoom, 1.0, 1e8)

        # -- next view (reconstruction target) --
        nv   = render_mandelbrot(cx, cy, zoom, view)
        nflat = nv.ravel().astype(np.float32)

        # -- reconstruction loss (drives learning what's being viewed) --
        pred  = pred_head.forward(h_out.data)
        recon_err = pred - nflat
        recon_mse = float(np.mean(recon_err**2))
        total_recon += recon_mse

        # -- backward through dense head --
        g_pred = (2.0 / vf) * recon_err
        pred_head.backward(g_pred)

        # -- aux_loss for energy homeostasis (gradient through energy gating) --
        if aux is not None:
            aux.grad = np.array([0.05])
            aux.backward()

        # clip + SGD for sili params
        for p_ in params:
            if p_.grad is not None:
                np.clip(p_.grad, -1., 1., out=p_.grad)
                p_.data -= lr * p_.grad; p_.grad = None

        state = h_out.data.copy()

        view_buf.append(view_arr.copy())
        hidden_buf.append(state.copy())
        if len(view_buf)   > 60: view_buf.pop(0)
        if len(hidden_buf) > 60: hidden_buf.pop(0)

        if (step+1) % report_every == 0:
            vr = compression_ratio(view_buf)
            hr = compression_ratio(hidden_buf)
            avg = total_recon / report_every
            top = ACT_NAMES[act_counts.argmax()]
            print(f"  step {step+1:5d}  pos=({cx:+.3f},{cy:+.3f}) zoom={zoom:.0f}"
                  f"  view_cr={vr:.3f} hid_cr={hr:.3f}"
                  f"  recon={avg:.4f}  top={top}")
            total_recon = 0.0; act_counts[:] = 0

    print(f"\nFinal: ({cx:.4f},{cy:.4f}) zoom={zoom:.1f}")
    print("(lower compress_ratio = more complex/varied content explored)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",        type=int,   default=3000)
    ap.add_argument("--hidden",       type=int,   default=64)
    ap.add_argument("--view",         type=int,   default=32)
    ap.add_argument("--lr",           type=float, default=0.003)
    ap.add_argument("--report-every", type=int,   default=300)
    a = ap.parse_args()
    run(a.steps, a.hidden, a.view, a.lr, a.report_every)

if __name__ == "__main__":
    main()
