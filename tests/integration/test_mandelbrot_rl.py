"""
Integration test: curiosity-driven Mandelbrot exploration with sparse FoldedLayer.

Runs until EITHER --steps is reached OR --timeout seconds elapses, whichever
comes first. Always prints the last completed step so steps-per-second can
be computed externally (or is reported directly).

Architecture:
  - FoldedLayer (FP4 delta-CSR sparse) as the hidden RNN layer
  - EnergyDynamics on hidden neurons: intrinsic curiosity via homeostasis
  - DenseHead for action selection (7 actions: pan x4, zoom x2, reset)
  - DenseHead for next-view reconstruction (drives learning what model sees)
  - Compression ratio of view+hidden buffers as exploration metric

Actions:
  0=pan_left  1=pan_right  2=pan_up  3=pan_down
  4=zoom_in   5=zoom_out   6=reset (escape flat/uninteresting regions)

Exploration metric: zlib compression ratio of the last 60 views.
  - Lower ratio = visited more complex/varied Mandelbrot regions
  - Rising ratio = agent stuck in flat/simple region (reset should fire)

Run:
  python -m tests.integration.test_mandelbrot_rl [--steps 5000] [--timeout 120]
"""

import argparse, math, time, zlib, warnings, signal
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import sili.cpu
from sili.tensor import Tensor
from sili.sparse_rnn import FoldedLayer
from sili.conversion.rnn_fold import FoldedBlockDescriptor, stack_csr_vertical
from sili.energy import EnergyDynamics

import torch  # only for CSR construction -- not in compute path


# ── Mandelbrot environment ────────────────────────────────────────────────────

def render_mandelbrot(cx: float, cy: float, zoom: float,
                      size: int = 32, max_iter: int = 64) -> np.ndarray:
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


def compression_ratio(bufs: list) -> float:
    if not bufs: return 1.0
    flat = np.concatenate([b.ravel() for b in bufs]).tobytes()
    return len(zlib.compress(flat, 1)) / max(len(flat), 1)


# ── Sparse hidden layer from random FP4 weights ───────────────────────────────

def make_sparse_hidden(n_in: int, n_out: int, density: float = 0.25,
                       lr: float = 0.003, seed: int = 0) -> FoldedLayer:
    """
    Create a FoldedLayer (sparse FP4 delta-CSR) for the hidden RNN layer.
    n_folds=1: single block, behaves as one large sparse linear layer.
    """
    torch.manual_seed(seed)
    W = torch.randn(n_out, n_in) * np.sqrt(2. / n_in)
    mask = torch.rand_like(W) < density
    W = (W * mask).to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
    desc = FoldedBlockDescriptor(
        n_folds=1, block_indices=[0],
        stacked_weights={'.h.weight': W},
        out_dims={'.h.weight': n_out},
        band_half_widths={'.h.weight': None},
        prefix='rnn.',
    )
    return FoldedLayer.from_descriptor(desc, learning_rate=lr, num_cpus=1)


# ── Dense heads (no energy on output) ─────────────────────────────────────────

class DenseHead:
    def __init__(self, in_f: int, out_f: int, lr: float):
        s = np.sqrt(2. / in_f)
        self.w = np.random.randn(in_f, out_f).astype(np.float32) * s
        self.b = np.zeros(out_f, np.float32)
        self.lr = lr; self._h = None

    def forward(self, h: np.ndarray) -> np.ndarray:
        self._h = h.copy(); return h @ self.w + self.b

    def backward(self, g: np.ndarray) -> np.ndarray:
        self.w -= self.lr * np.outer(self._h, g)
        self.b -= self.lr * g
        return g @ self.w.T


# ── Main ──────────────────────────────────────────────────────────────────────

def run(max_steps: int = 5000, timeout: float = 120.0, hidden: int = 64,
        view: int = 32, lr: float = 0.003, density: float = 0.25,
        aux_weight: float = 0.05, report_every: int = 200,
        verbose: bool = True) -> dict:
    """
    Returns dict with:
      steps_completed, elapsed_sec, steps_per_sec,
      final_view_cr, final_hid_cr, final_recon_mse,
      stop_reason ('steps'|'timeout')
    """
    N_ACT  = 7
    ACT_NAMES = ['pan<','pan>','pan^','panv','zoom+','zoom-','reset']
    vf     = view * view
    n_in   = 2  # just current action context -- full view flattened below

    # Sparse hidden layer (FP4 delta-CSR)
    layer  = make_sparse_hidden(vf + hidden, hidden, density, lr)

    # Energy: intrinsic curiosity on hidden neurons only
    frac   = max(0.1, min(0.4, 8. / hidden))
    energy = EnergyDynamics(drive=1./hidden, activation_cost=0.05,
                            precision=0.01, density=frac/2,
                            exploration=0.002, p=frac)

    # Action and reconstruction heads
    act_head   = DenseHead(hidden, N_ACT, lr)
    recon_head = DenseHead(hidden, vf, lr)

    # Navigation state
    cx, cy = -0.75, 0.0   # Mandelbrot boundary: rich structure
    zoom   = 50.0
    HOME   = (cx, cy, zoom)

    state     = np.zeros(hidden, np.float32)
    view_buf  = []; hid_buf = []
    act_counts = np.zeros(N_ACT, int)
    recon_sum  = 0.; n_recon = 0

    start_time = time.perf_counter()
    stop_reason = 'steps'
    step = 0

    if verbose:
        print(f"\n=== Mandelbrot RL  hidden={hidden}  view={view}x{view}"
              f"  density={density:.0%} ===")
        print(f"    FoldedLayer nnz={layer.nnz_total()}"
              f"  energy k_active~{max(1,round(frac*hidden))}")
        print(f"    max_steps={max_steps}  timeout={timeout:.0f}s")
        print(f"    7 actions: {ACT_NAMES}")
        print(f"    Starting at ({cx},{cy}) zoom={zoom}")

    for step in range(max_steps):
        # -- timeout check --
        elapsed = time.perf_counter() - start_time
        if elapsed >= timeout:
            stop_reason = 'timeout'
            break

        # -- render current view --
        view_arr   = render_mandelbrot(cx, cy, zoom, view)
        vflat      = view_arr.ravel()

        # -- sparse RNN forward (sili autograd) --
        inp    = Tensor(np.concatenate([vflat, state]).astype(np.float32))
        h_raw  = layer(inp)
        h_out, aux, _ = energy.forward(h_raw)

        # -- action selection (softmax sampling, no energy on output) --
        logits = act_head.forward(h_out.data)
        lg     = logits - logits.max()
        probs  = np.exp(lg) / np.exp(lg).sum()
        action = int(np.random.choice(N_ACT, p=probs))
        act_counts[action] += 1

        # -- apply action --
        pan = 0.3 / zoom
        if   action == 0: cx -= pan
        elif action == 1: cx += pan
        elif action == 2: cy -= pan
        elif action == 3: cy += pan
        elif action == 4: zoom = min(zoom * 1.4, 1e8)
        elif action == 5: zoom = max(zoom / 1.4, 1.0)
        elif action == 6: cx, cy, zoom = HOME   # reset

        # -- reconstruction target (next view) --
        nv    = render_mandelbrot(cx, cy, zoom, view)
        nflat = nv.ravel()

        # -- reconstruction loss -> drives hidden layer learning --
        pred      = recon_head.forward(h_out.data)
        recon_err = pred - nflat
        recon_mse = float(np.mean(recon_err**2))
        recon_sum += recon_mse; n_recon += 1

        g_pred = (2. / vf) * recon_err
        g_h    = recon_head.backward(g_pred)

        # -- gradient chain: recon head -> energy -> sparse hidden W --
        h_out.grad = g_h
        h_out.backward()

        if aux is not None:
            aux.grad = np.array([aux_weight], np.float32)
            aux.backward()

        # Sparse layer weights updated via backward (no explicit SGD needed --
        # backward_dense inside FoldedLayer handles weight updates)

        state = h_out.data.copy()

        # -- track exploration --
        view_buf.append(view_arr.copy())
        hid_buf.append(state.copy())
        if len(view_buf)  > 60: view_buf.pop(0)
        if len(hid_buf)   > 60: hid_buf.pop(0)

        if verbose and (step + 1) % report_every == 0:
            elapsed = time.perf_counter() - start_time
            vcr   = compression_ratio(view_buf)
            hcr   = compression_ratio(hid_buf)
            rmse  = recon_sum / max(1, n_recon)
            sps   = (step + 1) / elapsed
            top   = ACT_NAMES[act_counts.argmax()]
            print(f"  step {step+1:6d}  "
                  f"pos=({cx:+.3f},{cy:+.3f}) zoom={zoom:.0f}  "
                  f"view_cr={vcr:.3f} hid_cr={hcr:.3f}  "
                  f"recon={rmse:.4f}  "
                  f"top={top}  {sps:.1f} steps/s")
            recon_sum = 0.; n_recon = 0; act_counts[:] = 0

    elapsed   = time.perf_counter() - start_time
    steps_done = step + 1 if stop_reason == 'steps' else step

    result = dict(
        steps_completed  = steps_done,
        elapsed_sec      = elapsed,
        steps_per_sec    = steps_done / max(elapsed, 1e-6),
        final_view_cr    = compression_ratio(view_buf),
        final_hid_cr     = compression_ratio(hid_buf),
        final_recon_mse  = recon_sum / max(1, n_recon),
        stop_reason      = stop_reason,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Stop reason : {stop_reason}")
        print(f"Steps done  : {result['steps_completed']}")
        print(f"Elapsed     : {elapsed:.1f}s")
        print(f"Steps/sec   : {result['steps_per_sec']:.1f}")
        print(f"view_cr     : {result['final_view_cr']:.3f}  (lower = more varied exploration)")
        print(f"hid_cr      : {result['final_hid_cr']:.3f}")
        print(f"recon MSE   : {result['final_recon_mse']:.4f}")
        print(f"Final pos   : ({cx:.4f},{cy:.4f}) zoom={zoom:.1f}")
        print(f"{'='*60}")

    return result


def main():
    ap = argparse.ArgumentParser(description='Mandelbrot curiosity RL integration test')
    ap.add_argument('--steps',        type=int,   default=5000,
                    help='Maximum number of steps (whichever limit hits first stops)')
    ap.add_argument('--timeout',      type=float, default=120.0,
                    help='Maximum wall-clock seconds (whichever limit hits first stops)')
    ap.add_argument('--hidden',       type=int,   default=64)
    ap.add_argument('--view',         type=int,   default=32)
    ap.add_argument('--lr',           type=float, default=0.003)
    ap.add_argument('--density',      type=float, default=0.25)
    ap.add_argument('--report-every', type=int,   default=200)
    a = ap.parse_args()
    run(max_steps=a.steps, timeout=a.timeout, hidden=a.hidden,
        view=a.view, lr=a.lr, density=a.density,
        report_every=a.report_every)


if __name__ == '__main__':
    main()
