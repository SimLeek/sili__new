"""
Integration test: curiosity RL on Mandelbrot exploration, with disambiguation.

WHAT WAS WRONG BEFORE (v1):
  - The action head NEVER received a gradient. No policy learning existed;
    actions came from a frozen random-init network. Results were ambiguous
    because there was nothing to attribute behavior to.
  - Online reconstruction MSE rising is EXPECTED under working curiosity
    (agent seeks views it can't predict), so it can't distinguish
    learning-and-exploring from diverging. A held-out probe set is needed.
  - final recon stat read an accumulator that was reset at the last report.

WHAT THIS VERSION TESTS:
  1. Policy learning: REINFORCE on the action head with novelty reward
     (reward = reconstruction error of the view the action produced).
  2. Learning vs novelty disambiguation: a FIXED probe set of 8 views is
     evaluated every report interval WITHOUT training on them.
       probe MSE falling + online MSE rising  -> curiosity working
       probe MSE rising                        -> model diverging (broken)
  3. Zero-init attribution (--core zero): ALL weights start at exactly 0 and
     energy is applied to hidden AND output neurons (pre-charged to 1.9).
     Any exploration that emerges is attributable to energy dynamics, not to
     random initialisation. Output uses GROUPED neurons (4 per action,
     averaged) to smooth the 2.0-fire twitching, per design discussion.
  4. Pipeline validation (--core mistral): the folded toy-Mistral weights
     (gen_toy_mistral -> prune -> per-suffix FoldedLayer) form the recurrent
     core, driven by image input. NOTE the "V part": toy Mistral has NO
     vision weights, so the patch projection is a new component (Mistral is
     text-only; VLM variants add a separate vision encoder). Single-token
     attention collapses exactly to o_proj(v_proj(x)) since softmax over one
     position is 1, with GQA handled by tiling v from 16 to 32 dims.
  5. Policy baseline (--policy random, or --compare): identical run with
     uniform random actions and the same reconstruction learning. Coverage
     difference between curiosity and random isolates the policy effect.

METRICS PER REPORT:
  online recon MSE, probe MSE, action entropy (nats; uniform = ln 7 = 1.95),
  coverage (distinct (x, y, log-zoom) grid cells visited), compression ratios,
  steps/sec.

LIMITS: --steps and --timeout; run stops at whichever hits first and always
reports the last completed step and steps-per-second.

Run:
  python -m tests.integration.test_mandelbrot_rl --core zero --steps 2000
  python -m tests.integration.test_mandelbrot_rl --core mistral --steps 500
  python -m tests.integration.test_mandelbrot_rl --compare --timeout 60
"""

import argparse, math, time, zlib, warnings
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'test', 'python'))
warnings.filterwarnings('ignore')

import torch  # CSR construction only -- not in the compute path
import sili.cpu
from sili.tensor import Tensor, tanh as sili_tanh
from sili.sparse_rnn import FoldedLayer
from sili.conversion.rnn_fold import FoldedBlockDescriptor, stack_csr_vertical
from sili.energy import EnergyDynamics


# ── Mandelbrot environment ────────────────────────────────────────────────────

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


# Fixed probe set: 8 boundary views the agent is never trained on directly.
PROBES = [(-0.75, 0.0, 50.), (-0.75, 0.1, 80.), (-1.25, 0.0, 60.),
          (0.275, 0.006, 200.), (-0.5, 0.56, 100.), (-0.16, 1.035, 150.),
          (-1.77, 0.0, 120.), (0.36, 0.32, 90.)]


def compression_ratio(bufs):
    if not bufs: return 1.0
    flat = np.concatenate([b.ravel() for b in bufs]).tobytes()
    return len(zlib.compress(flat, 1)) / max(len(flat), 1)


def entropy_nats(counts):
    p = counts / max(1, counts.sum())
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# ── Cores ─────────────────────────────────────────────────────────────────────

class ZeroCore:
    """
    Single dense hidden layer, ALL ZEROS. Escape path: pre-charged energy
    fires neurons -> state becomes nonzero -> aux gradient shapes W -> input
    gradient reaches Wp. Any structure that emerges is energy-attributable.
    """
    def __init__(self, d_img, hidden):
        self.W = Tensor(np.zeros((d_img + hidden, hidden), np.float32))
        self.params = [self.W]

    def forward(self, x_img: Tensor, state: np.ndarray) -> Tensor:
        xh = Tensor(np.concatenate([x_img.data, state]))
        # Keep x_img in the graph via a parallel path so Wp gets gradient:
        # graph-x = concat is not a sili op, so route x_img through the top
        # rows of W explicitly: xh_graph = [x_img ; state_const]
        # Implemented as: h = x_img @ W_top + state_const @ W_bot
        d_img = x_img.data.shape[0]
        W_top = self.W  # single matrix; split via constant selectors
        # selector matmuls keep everything in one W for simple SGD:
        # x_full = x_img @ S_top + state @ S_bot ; h = x_full @ W
        # Cheaper: two matmul halves against row-slices is not expressible
        # without slicing ops, so do the standard trick: lift both into the
        # full input vector with constant embedding matrices.
        E_top = np.zeros((d_img, d_img + len(state)), np.float32)
        E_top[:, :d_img] = np.eye(d_img, dtype=np.float32)
        x_lift = x_img @ E_top                      # (d_in,) graph
        x_full = x_lift + np.concatenate([np.zeros(d_img, np.float32), state])
        return x_full @ self.W


class MistralCore:
    """
    Folded toy-Mistral recurrent cell using real pruned weights.

    Single-token attention collapse: softmax over one position = 1, so
      attn_out = o_proj(v_proj(x))          (q, k cancel exactly)
    GQA: v_proj outputs N_KV_HEADS*HEAD_DIM=16; o_proj expects 32; tile x2.
    MLP: down(silu(gate(x)) * up(x)), silu via tanh identity:
      sigmoid(x) = 0.5*(1 + tanh(x/2))  ->  silu(x) = x * sigmoid(x)
    RMSNorm folded weights are 1-D and skipped (values ~1.0; noted).
    """
    def __init__(self, sd_sparse, prefix, n_layers, hidden, lr):
        def fold(suffix):
            per_block = []
            for i in range(n_layers):
                t = sd_sparse[f"{prefix}{i}{suffix}"]
                if not t.is_sparse_csr:
                    t = t.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
                per_block.append(t)
            stacked = stack_csr_vertical(per_block)
            desc = FoldedBlockDescriptor(
                n_folds=n_layers, block_indices=list(range(n_layers)),
                stacked_weights={suffix: stacked},
                out_dims={suffix: int(per_block[0].shape[0])},
                band_half_widths={suffix: None}, prefix=prefix)
            return FoldedLayer.from_descriptor(desc, learning_rate=lr, num_cpus=1)

        self.v    = fold('.self_attn.v_proj.weight')   # 32 -> 16
        self.o    = fold('.self_attn.o_proj.weight')   # 32 -> 32
        self.gate = fold('.mlp.gate_proj.weight')      # 32 -> 64
        self.up   = fold('.mlp.up_proj.weight')        # 32 -> 64
        self.down = fold('.mlp.down_proj.weight')      # 64 -> 32
        self.hidden = hidden
        # GQA tile 16 -> 32 as a constant matmul (stays in autograd)
        self._tile = np.hstack([np.eye(16, dtype=np.float32),
                                np.eye(16, dtype=np.float32)])  # (16, 32)
        self.params = []  # folded layers self-update in backward

    def forward(self, x_img: Tensor, state: np.ndarray) -> Tensor:
        x_in  = x_img + state                       # (32,) graph
        attn  = self.o(self.v(x_in) @ self._tile)   # o(tile(v(x)))
        h1    = x_in + attn
        g     = self.gate(h1)
        u     = self.up(h1)
        silu_g = g * ((sili_tanh(g * 0.5) + 1.0) * 0.5)   # g * sigmoid(g)
        return h1 + self.down(silu_g * u)


# ── Run ───────────────────────────────────────────────────────────────────────

def run(core='zero', policy='curiosity', max_steps=2000, timeout=60.0,
        view=32, lr=0.01, aux_weight=0.05, pol_weight=0.3,
        group=4, report_every=200, verbose=True, seed=0):
    N_ACT = 7
    ACT_NAMES = ['pan<','pan>','pan^','panv','zoom+','zoom-','reset']
    np.random.seed(seed); torch.manual_seed(seed)

    hidden = 32                     # matches toy-Mistral HIDDEN
    patch  = 8
    n_patches = (view // patch) ** 2
    d_patch   = patch * patch

    zero_mode = (core == 'zero')

    # -- patch projection (the "V part" -- new weights, toy Mistral has none) --
    # mean-pool patches then project (identical to project-then-mean: linear).
    Wp = Tensor(np.zeros((d_patch, hidden), np.float32) if zero_mode else
                (np.random.randn(d_patch, hidden) * 0.05).astype(np.float32))

    # -- core --
    if zero_mode:
        core_net = ZeroCore(hidden, hidden)
    else:
        from gen_toy_mistral import build_toy_mistral_state_dict, N_LAYERS
        from sili.conversion.sparse_prune import default_min_abs_param
        sd, _ = build_toy_mistral_state_dict(seed=1234)
        thr = default_min_abs_param()
        sd_sparse = {}
        for k, v in sd.items():
            if v.ndim >= 2:
                m = v.abs() >= thr
                sd_sparse[k] = (v*m).to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
            else:
                sd_sparse[k] = v
        core_net = MistralCore(sd_sparse, 'model.layers.', N_LAYERS, hidden, lr)

    # -- energy: hidden (always) + output (grouped) --
    frac_h = max(0.1, min(0.4, 8. / hidden))
    energy_h = EnergyDynamics(drive=1./hidden, activation_cost=0.05,
                              precision=0.01, density=frac_h/2,
                              exploration=0.002, p=frac_h)
    n_out_neurons = N_ACT * group
    frac_o = max(0.15, min(0.5, 2.*N_ACT / n_out_neurons))
    energy_o = EnergyDynamics(drive=1./n_out_neurons, activation_cost=0.05,
                              precision=0.01, density=frac_o/2,
                              exploration=0.002, p=frac_o)
    if zero_mode:
        energy_h.energy = np.full(hidden,        1.9, np.float32)
        energy_o.energy = np.full(n_out_neurons, 1.9, np.float32)

    # -- action head: hidden -> N_ACT*group, grouped mean -> N_ACT logits --
    Wa = Tensor(np.zeros((hidden, n_out_neurons), np.float32) if zero_mode else
                (np.random.randn(hidden, n_out_neurons) * 0.05).astype(np.float32))
    # constant grouping matrix: (n_out_neurons, N_ACT), entries 1/group
    Gm = np.zeros((n_out_neurons, N_ACT), np.float32)
    for a in range(N_ACT):
        Gm[a*group:(a+1)*group, a] = 1.0 / group

    # -- reconstruction head (numpy SGD; zero-init learns once hidden fires) --
    Vr = (np.zeros((hidden, view*view), np.float32) if zero_mode else
          (np.random.randn(hidden, view*view) * 0.02).astype(np.float32))
    br = np.zeros(view*view, np.float32)

    dense_params = [Wp, Wa] + core_net.params

    # -- navigation --
    cx, cy, zoom = -0.75, 0.0, 50.0
    HOME = (cx, cy, zoom)
    state = np.zeros(hidden, np.float32)

    # -- trackers --
    view_buf, hid_buf = [], []
    act_counts_win = np.zeros(N_ACT, int); act_counts_all = np.zeros(N_ACT, int)
    coverage = set()
    recon_win = 0.; n_win = 0
    recon_all = 0.; n_all = 0
    reward_baseline = 0.0
    probe_history = []

    def probe_eval():
        """Recon MSE on the fixed probe set. Saves/restores dynamic state."""
        sv_state = state.copy()
        sv_eh = energy_h.energy.copy() if energy_h.energy is not None else None
        sv_eo = energy_o.energy.copy() if energy_o.energy is not None else None
        total = 0.
        for (px, py, pz) in PROBES:
            pv = render_mandelbrot(px, py, pz, view)
            pm = pv.reshape(view//patch, patch, view//patch, patch) \
                   .transpose(0,2,1,3).reshape(n_patches, d_patch).mean(axis=0)
            x_img = Tensor(pm) @ Wp
            h_pre = core_net.forward(x_img, np.zeros(hidden, np.float32))
            h_o, _, _ = energy_h.forward(h_pre)
            pred = h_o.data @ Vr + br
            total += float(np.mean((pred - pv.ravel())**2))
        if sv_eh is not None: energy_h.energy = sv_eh
        if sv_eo is not None: energy_o.energy = sv_eo
        state[:] = sv_state
        return total / len(PROBES)

    start = time.perf_counter()
    stop_reason = 'steps'
    step = 0

    if verbose:
        print(f"\n=== Mandelbrot RL v2  core={core}  policy={policy} ===")
        print(f"    hidden={hidden}  view={view}x{view}  out_neurons={n_out_neurons} "
              f"({N_ACT} actions x {group} grouped)")
        print(f"    limits: steps={max_steps}  timeout={timeout:.0f}s")
        if zero_mode:
            print(f"    ZERO INIT everywhere; energy pre-charged 1.9 on hidden+output")
        else:
            print(f"    toy-Mistral folded core (v/o attn-collapse + gated MLP); "
                  f"patch projection is NEW (no vision weights in Mistral)")

    for step in range(max_steps):
        if time.perf_counter() - start >= timeout:
            stop_reason = 'timeout'
            break

        # -- observe --
        varr = render_mandelbrot(cx, cy, zoom, view)
        pmean = varr.reshape(view//patch, patch, view//patch, patch) \
                    .transpose(0,2,1,3).reshape(n_patches, d_patch).mean(axis=0)
        x_img = Tensor(pmean) @ Wp                       # (hidden,) graph

        # -- core + hidden energy --
        h_pre = core_net.forward(x_img, state)
        h_out, aux_h, _ = energy_h.forward(h_pre)

        # -- action path: energy-gated grouped output --
        la28 = h_out @ Wa                                # (28,) graph
        e28, aux_o, _ = energy_o.forward(la28)
        l7  = e28 @ Gm                                   # grouped mean, in graph

        if policy == 'random':
            probs = np.full(N_ACT, 1./N_ACT, np.float32)
        else:
            lg = l7.data - l7.data.max()
            probs = np.exp(lg); probs /= probs.sum()
        action = int(np.random.choice(N_ACT, p=probs))
        act_counts_win[action] += 1; act_counts_all[action] += 1

        # -- act --
        pan = 0.3 / zoom
        if   action == 0: cx -= pan
        elif action == 1: cx += pan
        elif action == 2: cy -= pan
        elif action == 3: cy += pan
        elif action == 4: zoom = min(zoom*1.4, 1e8)
        elif action == 5: zoom = max(zoom/1.4, 1.0)
        elif action == 6: cx, cy, zoom = HOME
        coverage.add((int(cx*50), int(cy*50), int(round(math.log(zoom, 1.4)))))

        # -- next view, reconstruction --
        nv = render_mandelbrot(cx, cy, zoom, view)
        nflat = nv.ravel()
        pred = h_out.data @ Vr + br
        rerr = pred - nflat
        rmse = float(np.mean(rerr**2))
        recon_win += rmse; n_win += 1; recon_all += rmse; n_all += 1

        # recon head numpy SGD + gradient back to h
        g_pred = (2./nflat.size) * rerr
        g_h_recon = g_pred @ Vr.T
        Vr -= lr * np.outer(h_out.data, g_pred)
        br -= lr * g_pred

        # -- REINFORCE (curiosity policy only): reward = novelty (recon error) --
        G_pol = np.zeros(N_ACT, np.float32)
        if policy == 'curiosity':
            reward = rmse
            adv = reward - reward_baseline
            reward_baseline = 0.99*reward_baseline + 0.01*reward
            onehot = np.zeros(N_ACT, np.float32); onehot[action] = 1.
            G_pol = pol_weight * adv * (probs - onehot)

        # -- SINGLE backward via loss proxy (avoids multi-root double-count) --
        proxy = (h_out * g_h_recon).sum() + (l7 * G_pol).sum()
        if aux_h is not None: proxy = proxy + aux_h * aux_weight
        if aux_o is not None: proxy = proxy + aux_o * aux_weight
        proxy.grad = np.ones(1, np.float32) if proxy.data.ndim else np.float32(1.)
        proxy.backward()

        for p in dense_params:
            if p.grad is not None:
                np.clip(p.grad, -1., 1., out=p.grad)
                p.data -= lr * p.grad; p.grad = None

        state = h_out.data.copy()
        view_buf.append(varr); hid_buf.append(state.copy())
        if len(view_buf) > 60: view_buf.pop(0)
        if len(hid_buf) > 60: hid_buf.pop(0)

        if verbose and (step + 1) % report_every == 0:
            el = time.perf_counter() - start
            pm = probe_eval(); probe_history.append(pm)
            print(f"  step {step+1:6d}  pos=({cx:+.3f},{cy:+.3f}) z={zoom:.0f}  "
                  f"recon={recon_win/max(1,n_win):.4f} probe={pm:.4f}  "
                  f"H={entropy_nats(act_counts_win):.2f}  cov={len(coverage)}  "
                  f"vcr={compression_ratio(view_buf):.3f}  "
                  f"{(step+1)/el:.0f} st/s")
            recon_win = 0.; n_win = 0; act_counts_win[:] = 0

    elapsed = time.perf_counter() - start
    steps_done = step + (1 if stop_reason == 'steps' else 0)
    final_probe = probe_eval()
    probe_history.append(final_probe)

    result = dict(
        core=core, policy=policy,
        steps_completed=steps_done, elapsed_sec=elapsed,
        steps_per_sec=steps_done/max(elapsed, 1e-6),
        stop_reason=stop_reason,
        recon_mse_all=recon_all/max(1, n_all),
        probe_mse_first=probe_history[0] if probe_history else float('nan'),
        probe_mse_final=final_probe,
        action_entropy=entropy_nats(act_counts_all),
        coverage=len(coverage),
        view_cr=compression_ratio(view_buf),
        hid_cr=compression_ratio(hid_buf),
    )

    if verbose:
        print(f"\n  stop={stop_reason}  steps={steps_done}  "
              f"{result['steps_per_sec']:.0f} st/s")
        print(f"  probe MSE first->final: {result['probe_mse_first']:.4f} -> "
              f"{result['probe_mse_final']:.4f}  "
              f"({'LEARNING' if result['probe_mse_final'] < result['probe_mse_first'] else 'not improving'})")
        print(f"  online recon (whole run): {result['recon_mse_all']:.4f}")
        print(f"  entropy={result['action_entropy']:.2f} (uniform={math.log(N_ACT):.2f})  "
              f"coverage={result['coverage']}")
    return result


def compare(core, max_steps, timeout, **kw):
    print(f"\n{'='*64}\nCOMPARE: curiosity vs random policy  (core={core})\n{'='*64}")
    r_cur = run(core=core, policy='curiosity', max_steps=max_steps,
                timeout=timeout, **kw)
    r_rnd = run(core=core, policy='random', max_steps=max_steps,
                timeout=timeout, **kw)
    print(f"\n{'='*64}")
    print(f"{'metric':<22}{'curiosity':>14}{'random':>14}")
    for k in ('coverage', 'probe_mse_final', 'recon_mse_all',
              'action_entropy', 'steps_per_sec'):
        print(f"{k:<22}{r_cur[k]:>14.4f}{r_rnd[k]:>14.4f}")
    cov_ok = r_cur['coverage'] != r_rnd['coverage']
    print(f"\ncuriosity {'differs from' if cov_ok else 'matches'} random on coverage"
          f" -- {'policy is steering' if r_cur['coverage'] > r_rnd['coverage'] else 'inspect further'}")
    print(f"{'='*64}")
    return r_cur, r_rnd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--core',    default='zero', choices=['zero', 'mistral'])
    ap.add_argument('--policy',  default='curiosity', choices=['curiosity', 'random'])
    ap.add_argument('--compare', action='store_true',
                    help='run curiosity AND random policies, print side-by-side')
    ap.add_argument('--steps',   type=int,   default=2000)
    ap.add_argument('--timeout', type=float, default=60.0)
    ap.add_argument('--lr',      type=float, default=0.01)
    ap.add_argument('--group',   type=int,   default=4)
    ap.add_argument('--report-every', type=int, default=200)
    a = ap.parse_args()
    kw = dict(lr=a.lr, group=a.group, report_every=a.report_every)
    if a.compare:
        compare(a.core, a.steps, a.timeout, **kw)
    else:
        run(core=a.core, policy=a.policy, max_steps=a.steps,
            timeout=a.timeout, **kw)


if __name__ == '__main__':
    main()
