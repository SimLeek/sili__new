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
  3. Zero/near-zero-init attribution (--core sparse, --core dense): ALL
     weights start at exactly 0 (dense) or near-zero (sparse -- see
     make_near_zero_sparse_layer's docstring for why exact zero doesn't
     survive sparse construction) and energy is applied to hidden AND
     output neurons (pre-charged to 1.9). Any exploration that emerges is
     attributable to energy dynamics, not to random initialisation. Output
     uses GROUPED neurons (4 per action, averaged) to smooth the 2.0-fire
     twitching, per design discussion.
  3b. --core sparse (DEFAULT) vs --core dense (the alternative/comparison):
     sparse is a large, genuinely sparse FoldedLayer-based core (--hidden
     tunable, default 1024, --density controls connectivity fraction).
     dense is the original plain-Tensor version, kept for comparison so
     sparse-vs-dense behavior at the same hidden size can be checked
     directly against each other. Previously the only "large" option was
     dense and the only sparse option was --core mistral (fixed tiny at
     hidden=32) -- neither tested "sparse AND large" together.
  4. Pipeline validation (--core mistral): the folded toy-Mistral weights
     (gen_toy_mistral -> prune -> per-suffix FoldedLayer) form the recurrent
     core, driven by image input, fixed at hidden=32 (folded weights are
     fixed-width). NOTE the "V part": toy Mistral has NO vision weights, so
     the input projection is a new component (Mistral is text-only; VLM
     variants add a separate vision encoder). Single-token attention
     collapses exactly to o_proj(v_proj(x)) since softmax over one position
     is 1, with GQA handled by tiling v from 16 to 32 dims.
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
  python -m tests.integration.test_mandelbrot_rl --core sparse --steps 2000
  python -m tests.integration.test_mandelbrot_rl --core dense --steps 2000
  python -m tests.integration.test_mandelbrot_rl --core mistral --steps 500
  python -m tests.integration.test_mandelbrot_rl --compare --timeout 60
"""

import argparse, math, time, zlib, warnings
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'unit', 'python'))
warnings.filterwarnings('ignore')

import torch  # CSR construction only -- not in the compute path
import sili.cpu
from sili.tensor import Tensor, tanh as sili_tanh, combine_losses
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


def encode_observation(varr: np.ndarray, cx: float, cy: float, zoom: float) -> np.ndarray:
    """
    Raw view + explicit position/zoom -> flat input vector.

    Two fixes to what was here before (mean-pooling all patches into one
    vector, no position signal at all):

    1. Full flatten, not mean-pool. The old pipeline reshaped the view into
       16 separate 8x8 patches then averaged ALL of them into one vector --
       destroying every bit of spatial layout, leaving only "what does an
       average local patch look like" with no information about WHERE any
       feature sits in the frame. Flattening instead keeps every pixel at
       a fixed position in the input vector, so a linear projection can in
       principle learn to treat different spatial regions differently.
    2. Explicit (cx, cy, zoom) as input, not left for the network to infer.
       zoom is log-scaled (it spans 1 to 1e8) and everything is roughly
       unit-scaled so it doesn't dominate the visual features by magnitude.
       Without this, the network had literally no way to know where it is
       except by guessing from ambiguous, now-flattened pixel content.
    """
    pos_feat = np.array([cx / 2.0, cy / 2.0, math.log2(max(zoom, 1.0)) / 10.0],
                        dtype=np.float32)
    return np.concatenate([varr.reshape(-1).astype(np.float32), pos_feat])


# ── Sparse construction (near-zero-init) ──────────────────────────────────────

def make_near_zero_sparse_layer(n_in: int, n_out: int, density: float,
                                lr: float, num_cpus: int = 1,
                                seed: int = 0) -> FoldedLayer:
    """
    A genuinely sparse FoldedLayer with a fixed random connectivity pattern
    and near-zero initial values, for the same "nothing works until energy
    dynamics + backprop shape it" attribution property the exact-zero dense
    cores use -- but exact zero does not survive this construction path.

    CONFIRMED BY DIRECT TESTING (not assumed): building a torch sparse_csr
    tensor with EXPLICIT STORED ZEROS at a fixed connectivity pattern (via
    torch.sparse_csr_tensor with a values tensor of all 0.0, not via
    tensor.to_sparse() on a dense zero tensor, which collapses to 0 nnz
    regardless of any mask) DOES correctly preserve the intended nnz through
    construction of the torch tensor itself -- but FoldedLayer.from_descriptor
    silently drops every entry with value exactly 0.0 during construction
    (verified: source csr had nnz=80, immediately after construction
    layer.nnz == 0, before any forward/backward call at all). This is
    presumably intentional for the NORMAL use case (loading real pretrained
    weights, where a stored zero genuinely means "no meaningful connection"),
    but is incompatible with wanting "connection exists structurally, value
    is deliberately near-zero for now."

    Fix: use a TINY nonzero value (1e-6) instead of exact 0.0. Confirmed by
    direct testing that this survives construction intact down to at least
    1e-8, and that a subsequent backward() call correctly updates the stored
    values (unlike exact zero, which also silently failed to update even
    when construction was worked around, due to a SEPARATE issue: per-row
    value_scale for an all-zero row defaults to a scale that makes any
    single realistic gradient step round back to zero under FP4 quantization
    -- the exact same class of problem already solved for importance_scale
    elsewhere in this codebase (imp_scale = learning_rate / FP4_MAX), just
    not previously applied to the value scale for a from-scratch sparse
    layer). Both fixes are needed together; each alone was insufficient
    (confirmed by testing each in isolation before combining them).

    1e-6 is six orders of magnitude below typical operating signal scale in
    this system (energy fires at 2.0, typical activations O(0.1-1)) --
    negligible relative to any meaningful signal, functionally equivalent to
    zero-init for all practical purposes, but NOT literally exact zero. This
    is stated plainly rather than silently calling it "zero-init": the
    distinction matters for the same attribution reasoning the exact-zero
    dense cores exist for.
    """
    TINY = 1e-6
    FP4_MAX = 6.0
    torch.manual_seed(seed)
    mask = torch.rand(n_out, n_in) < density
    rows, cols = torch.nonzero(mask, as_tuple=True)
    nnz = len(rows)
    order = torch.argsort(rows, stable=True)
    rows_sorted, cols_sorted = rows[order], cols[order]
    counts = torch.bincount(rows_sorted, minlength=n_out)   # vectorized, not
    crow = torch.cat([torch.zeros(1, dtype=torch.int64),     # a Python loop --
                      torch.cumsum(counts, dim=0)])           # matters at scale
    values = torch.full((nnz,), TINY, dtype=torch.float32)
    csr = torch.sparse_csr_tensor(crow, cols_sorted, values, size=(n_out, n_in))

    desc = FoldedBlockDescriptor(
        n_folds=1, block_indices=[0], stacked_weights={'.w': csr},
        out_dims={'.w': n_out}, band_half_widths={'.w': None}, prefix='sparse.')
    layer = FoldedLayer.from_descriptor(desc, learning_rate=lr, num_cpus=num_cpus)

    sub = layer._sili_layers['.w']
    for r in range(n_out):
        sub.set_value_scale_raw(r, lr / FP4_MAX)
    return layer


class DenseProjection:
    """Wraps a dense Tensor with the same call/.params interface as
    SparseProjection below, so call sites don't need to know or care which
    mode (--core dense vs --core sparse) is active."""
    def __init__(self, W: Tensor):
        self.W = W
        self.params = [W]

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W


class SparseProjection:
    """Wraps a FoldedLayer with the same call/.params interface as
    DenseProjection. params is empty because folded layers self-update via
    backward_dense (same reason core_net.params is empty for FoldedLayer-
    based cores) -- the outer SGD loop must not also try to step them."""
    def __init__(self, layer: FoldedLayer):
        self.layer = layer
        self.params = []

    def __call__(self, x: Tensor) -> Tensor:
        return self.layer(x)


# ── Cores ─────────────────────────────────────────────────────────────────────

class DenseCore:
    """
    Dense hidden layer, all zeros. The ALTERNATIVE/comparison test -- see
    SparseCore below for the default. Escape path: pre-charged energy fires
    neurons -> state becomes nonzero -> aux gradient shapes W -> input
    gradient reaches Wp. Any structure that emerges is energy-attributable.
    Exact zero-init works cleanly here since a dense Tensor has no separate
    "connectivity vs value" concept the way a sparse layer does (see
    SparseCore's docstring for why exact zero does NOT survive there).
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


class SparseCore:
    """
    Large, genuinely sparse recurrent cell -- the DEFAULT. sili's actual
    distinguishing capability is sparse computation at scale; pairing
    "sparse" with "tiny" (--core mistral, fixed at hidden=32 by its
    pretrained toy weights) and "large" with "dense" (the old default)
    tested neither property together. This core is sparse AND tunable to
    large hidden sizes via --hidden.

    Two SEPARATE sparse layers (image contribution, recurrent contribution)
    summed, rather than one big lifted-and-concatenated matrix -- avoids the
    lift trick DenseCore needs (which exists specifically to route gradients
    through a single dense matrix multiply) and lets image-sparsity and
    recurrent-sparsity be reasoned about/tuned independently if ever useful.
    Same shape of fix as MistralCore's Wc_img/Wc_state split.

    Near-zero (not exact-zero) init -- see make_near_zero_sparse_layer's
    docstring for the two-part reason (exact zero doesn't survive
    FoldedLayer construction; even a worked-around exact zero doesn't
    reliably escape via backward due to a separate value_scale issue).
    """
    def __init__(self, d_img, hidden, density=0.1, lr=0.01, num_cpus=1, seed=0):
        self.core_img   = make_near_zero_sparse_layer(
            d_img, hidden, density, lr, num_cpus, seed)
        self.core_state = make_near_zero_sparse_layer(
            hidden, hidden, density, lr, num_cpus, seed + 1)
        self.params = []   # folded layers self-update via backward_dense

    def forward(self, x_img: Tensor, state: np.ndarray) -> Tensor:
        return self.core_img(x_img) + self.core_state(Tensor(state))


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
    def __init__(self, sd_sparse, prefix, n_layers, hidden, lr, num_cpus=1):
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
            return FoldedLayer.from_descriptor(desc, learning_rate=lr, num_cpus=num_cpus)

        self.v    = fold('.self_attn.v_proj.weight')   # 32 -> 16
        self.o    = fold('.self_attn.o_proj.weight')   # 32 -> 32
        self.gate = fold('.mlp.gate_proj.weight')      # 32 -> 64
        self.up   = fold('.mlp.up_proj.weight')        # 32 -> 64
        self.down = fold('.mlp.down_proj.weight')      # 64 -> 32
        self.hidden = hidden
        # GQA tile 16 -> 32 as a constant matmul (stays in autograd)
        self._tile = np.hstack([np.eye(16, dtype=np.float32),
                                np.eye(16, dtype=np.float32)])  # (16, 32)
        # Learned image/state combination (replaces raw x_img + state).
        # The folded weights are fixed-width and pretrained-shaped, so they
        # can't be widened, but there's no reason the way image content and
        # recurrent memory get COMBINED before entering them has to be a raw
        # element-wise sum -- that gives the network no way to weight "new
        # visual input" against "old memory" differently per dimension.
        # Two small learned projections, summed, give it that freedom
        # (mathematically the same shape of fix as ZeroCore's separate
        # input/recurrent weight blocks, just written as two matrices
        # instead of one lifted-and-split matrix).
        scale = 1.0 / math.sqrt(hidden)
        self.Wc_img   = Tensor((np.random.randn(hidden, hidden) * scale).astype(np.float32))
        self.Wc_state = Tensor((np.random.randn(hidden, hidden) * scale).astype(np.float32))
        self.params = [self.Wc_img, self.Wc_state]  # folded layers self-update; these don't

    def forward(self, x_img: Tensor, state: np.ndarray) -> Tensor:
        x_in  = x_img @ self.Wc_img + Tensor(state) @ self.Wc_state   # learned combine
        attn  = self.o(self.v(x_in) @ self._tile)   # o(tile(v(x)))
        h1    = x_in + attn
        g     = self.gate(h1)
        u     = self.up(h1)
        silu_g = g * ((sili_tanh(g * 0.5) + 1.0) * 0.5)   # g * sigmoid(g)
        return h1 + self.down(silu_g * u)


# ── Run ───────────────────────────────────────────────────────────────────────

def run(core='sparse', policy='curiosity', agent='reinforce',
        max_steps=2000, timeout=60.0,
        view=32, hidden=1024, density=0.1, num_cpus=1,
        lr=0.01, aux_weight=0.05, pol_weight=0.3,
        group=4, report_every=200, verbose=True, seed=0,
        display=False, gamma=0.99, entropy_scale=0.01,
        loss_alpha=0.2, target_update=0.005):
    N_ACT = 7
    ACT_NAMES = ['pan<','pan>','pan^','panv','zoom+','zoom-','reset']
    np.random.seed(seed); torch.manual_seed(seed)

    if core == 'mistral' and hidden != 32:
        if verbose:
            print(f"    --hidden {hidden} ignored: --core mistral uses folded "
                  f"toy-Mistral weights fixed at hidden=32 (can't be widened)")
        hidden = 32

    # zero_mode covers BOTH zero-init cores (sparse near-zero, dense exact-
    # zero) -- mistral is the only core that starts from nonzero (pretrained
    # toy) weights. Gates the SEPARATE small head matrices (action, value,
    # reconstruction) that stay plain dense Tensors regardless of which core
    # is active -- only the recurrent core + input projection differ between
    # sparse/dense/mistral; the small heads are cheap enough that making
    # them sparse too wasn't part of what was asked for here.
    zero_mode = core in ('sparse', 'dense')
    raw_dim = view * view + 3   # full flattened view + (cx, cy, log-zoom)

    # -- input projection (the "V part" -- new weights, toy Mistral has none) --
    if core == 'sparse':
        Wp = SparseProjection(make_near_zero_sparse_layer(
            raw_dim, hidden, density, lr, num_cpus, seed))
    else:
        Wp = DenseProjection(Tensor(
            np.zeros((raw_dim, hidden), np.float32) if zero_mode else
            (np.random.randn(raw_dim, hidden) * 0.05).astype(np.float32)))

    # -- core --
    if core == 'sparse':
        core_net = SparseCore(hidden, hidden, density, lr, num_cpus, seed + 100)
    elif core == 'dense':
        core_net = DenseCore(hidden, hidden)
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
        core_net = MistralCore(sd_sparse, 'model.layers.', N_LAYERS, hidden, lr, num_cpus)

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

    # -- RTAC value heads: real-time input = [h ; onehot(prev_action)] --
    # (rtrl/rtac.py: model((next_obs, action)) -- action is part of the state)
    # DOUBLE CRITIC (rtrl/rtac.py: reduce(torch.min, next_value_target)):
    # two independently-initialized value heads, both trained toward the
    # same TD target; advantage/bootstrap use min(v1,v2) to counter
    # overestimation bias. Wv_h2/Wv_a2 mirror Wv_h/Wv_a exactly.
    def _mk_value_head():
        wh = Tensor(np.zeros((hidden, 1), np.float32) if zero_mode else
                    (np.random.randn(hidden, 1) * 0.05).astype(np.float32))
        wa = Tensor(np.zeros((N_ACT, 1), np.float32))
        return wh, wa
    Wv_h,  Wv_a  = _mk_value_head()
    Wv_h2, Wv_a2 = _mk_value_head()
    Wvh_t,  Wva_t  = Wv_h.data.copy(),  Wv_a.data.copy()   # EMA target nets
    Wvh_t2, Wva_t2 = Wv_h2.data.copy(), Wv_a2.data.copy()
    from sili.rl_utils import PopArt
    popart = PopArt(beta=0.0003, start_pop=8)
    h_prev = None; a_prev = 0; a_prevprev = 0; r_prev = 0.0

    # -- optional live display: view | reconstruction, 8x nearest upscale --
    # Uses displayarray (ModernGL-backed) rather than cv2.imshow -- OpenCV's
    # HighGUI window backend is known to be slow/flaky to open on some
    # systems. Install with:
    #   pip install git+https://github.com/SimLeek/displayarray.git@moderngl
    # As of writing, that branch is under active development (its own CI
    # isn't wired up yet) and may not import cleanly -- if
    # displayarray.window.mglwindow fails to import (its font submodule is
    # incomplete on this branch), this falls back to headless exactly like
    # a missing dependency would, rather than crashing the run.
    d = None; show = None
    if display:
        try:
            from displayarray import display as _da_display
            d = _da_display(blocking=False)   # no initial source; windows are
                                               # created lazily by the first
                                               # d.update(arr, id) call below
            def show(v, p):
                img = np.hstack([v, np.clip(p, 0, 1).reshape(view, view)])
                img_u8 = (img * 255).astype(np.uint8)
                # No manual upscaling: displayarray scales on the GPU when
                # rendering the window, so a CPU-side repeat() here is both
                # unnecessary and wastes cycles better spent training.
                d.update(img_u8, 'mandelbrot: view | reconstruction')
        except ImportError as e:
            print(f'    --display requested but displayarray not usable '
                  f'({e}); continuing headless')

    # -- reconstruction head (numpy SGD; zero-init learns once hidden fires) --
    Vr = (np.zeros((hidden, view*view), np.float32) if zero_mode else
          (np.random.randn(hidden, view*view) * 0.02).astype(np.float32))
    br = np.zeros(view*view, np.float32)

    dense_params = Wp.params + [Wa, Wv_h, Wv_a, Wv_h2, Wv_a2] + core_net.params

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
            x_img = Wp(Tensor(encode_observation(pv, px, py, pz)))
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
        print(f"\n=== Mandelbrot RL v2  core={core}  policy={policy}  agent={agent} ===")
        print(f"    hidden={hidden}  view={view}x{view}  raw_dim={raw_dim} "
              f"(full view + pos/zoom)  out_neurons={n_out_neurons} "
              f"({N_ACT} actions x {group} grouped)")
        print(f"    limits: steps={max_steps}  timeout={timeout:.0f}s")
        if zero_mode:
            kind = "NEAR-ZERO" if core == 'sparse' else "ZERO"
            print(f"    {kind} INIT everywhere; energy pre-charged 1.9 on hidden+output")
        else:
            print(f"    toy-Mistral folded core (v/o attn-collapse + gated MLP); "
                  f"input projection is NEW (no vision weights in Mistral)")

    for step in range(max_steps):
        if time.perf_counter() - start >= timeout:
            stop_reason = 'timeout'
            break

        # -- observe --
        varr = render_mandelbrot(cx, cy, zoom, view)
        x_img = Wp(Tensor(encode_observation(varr, cx, cy, zoom)))   # (hidden,) graph

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

        if show is not None:
            show(varr, pred)

        # -- intrinsic reward: novelty + homeostatic pressure --
        # (requirements doc 4.1: r = w_r*recon + w_e*energy_aux; the critic
        # learns V of this combined intrinsic return)
        aux_h_val = float(np.asarray(aux_h.data).ravel()[0]) if aux_h is not None else 0.0
        reward = rmse + 0.1 * aux_h_val

        onehot = np.zeros(N_ACT, np.float32); onehot[action] = 1.
        G_pol = np.zeros(N_ACT, np.float32)
        terms = [(h_out, g_h_recon)]
        if aux_h is not None: terms.append((aux_h, aux_weight))
        if aux_o is not None: terms.append((aux_o, aux_weight))

        if policy == 'curiosity' and agent == 'reinforce':
            adv = reward - reward_baseline
            reward_baseline = 0.99*reward_baseline + 0.01*reward
            G_pol = pol_weight * adv * (probs - onehot)

        elif policy == 'curiosity' and agent == 'rtac':
            # rtrl/rtac.py pattern, discrete-online adaptation:
            #   loss_total = alpha*loss_actor + (1-alpha)*loss_critic
            # DOUBLE CRITIC: v_now = min(v1, v2) for the actor's advantage,
            # both heads trained toward the same PopArt-normalized TD target.
            oh_prev = np.zeros(N_ACT, np.float32); oh_prev[a_prev] = 1.
            v1_now = h_out @ Wv_h  + Tensor(oh_prev) @ Wv_a
            v2_now = h_out @ Wv_h2 + Tensor(oh_prev) @ Wv_a2
            v_now_min = min(float(np.asarray(v1_now.data).ravel()[0]),
                            float(np.asarray(v2_now.data).ravel()[0]))
            v_now_orig = popart.unnormalize(v_now_min)

            # actor: advantage vs LEARNED (denormalized) value baseline + entropy
            adv = reward - v_now_orig
            G_pol = loss_alpha * pol_weight * adv * (probs - onehot)
            H = -(probs * np.log(probs + 1e-9)).sum()
            G_pol += loss_alpha * entropy_scale * probs * (np.log(probs + 1e-9) + H)

            # critic: one-step TD on stored previous transition, heads-only
            # recompute (h_prev is a constant leaf; trunk-through-critic
            # remains a follow-up item, see requirements doc 4.2). Target
            # bootstraps from min(target_net_1, target_net_2), PopArt-
            # normalized before computing the critic loss gradient.
            if h_prev is not None:
                oh_pp = np.zeros(N_ACT, np.float32); oh_pp[a_prevprev] = 1.
                v1_prev = Tensor(h_prev) @ Wv_h  + Tensor(oh_pp) @ Wv_a
                v2_prev = Tensor(h_prev) @ Wv_h2 + Tensor(oh_pp) @ Wv_a2

                v1_targ = float((h_out.data @ Wvh_t  + oh_prev @ Wva_t ).ravel()[0])
                v2_targ = float((h_out.data @ Wvh_t2 + oh_prev @ Wva_t2).ravel()[0])
                v_targ_min_norm = min(v1_targ, v2_targ)          # normalized space
                v_targ_min_orig = popart.unnormalize(v_targ_min_norm)

                y_orig = r_prev + gamma * v_targ_min_orig        # raw TD target
                # PopArt update: rescales ALL FOUR value-head weight/bias
                # arrays (both online critics AND both target nets) in the
                # SAME call, so nothing is left numerically inconsistent by
                # this stats update alone -- only actual gradient/polyak
                # steps should move any of them afterward.
                y_norm = popart.update_and_rescale(
                    y_orig,
                    weight_arrays=[Wv_h.data, Wv_h2.data, Wvh_t, Wvh_t2],
                    bias_arrays=[Wv_a.data, Wv_a2.data, Wva_t, Wva_t2])
                g_c1 = (1.0 - loss_alpha) * 2.0 * (float(np.asarray(v1_prev.data).ravel()[0]) - y_norm)
                g_c2 = (1.0 - loss_alpha) * 2.0 * (float(np.asarray(v2_prev.data).ravel()[0]) - y_norm)
                terms.append((v1_prev, np.array([g_c1], np.float32)))
                terms.append((v2_prev, np.array([g_c2], np.float32)))

            Wvh_t  += target_update * (Wv_h.data  - Wvh_t)
            Wva_t  += target_update * (Wv_a.data  - Wva_t)
            Wvh_t2 += target_update * (Wv_h2.data - Wvh_t2)
            Wva_t2 += target_update * (Wv_a2.data - Wva_t2)

        terms.append((l7, G_pol))
        combine_losses(*terms).backward()

        for p in dense_params:
            if p.grad is not None:
                np.clip(p.grad, -1., 1., out=p.grad)
                p.data -= lr * p.grad; p.grad = None

        state = h_out.data.copy()
        h_prev = state.copy(); a_prevprev = a_prev; a_prev = action; r_prev = reward
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

    if d is not None:
        d.end()   # explicit cleanup rather than relying on __del__ timing

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
    ap.add_argument('--core',    default='sparse',
                    choices=['sparse', 'dense', 'mistral'],
                    help='sparse (default): large, genuinely sparse '
                         'FoldedLayer core, near-zero-init. dense: the '
                         'alternative/comparison, plain Tensor, exact-zero-'
                         'init. mistral: folded toy-Mistral weights, fixed '
                         'at hidden=32.')
    ap.add_argument('--agent',   default='reinforce', choices=['reinforce', 'rtac'])
    ap.add_argument('--display', action='store_true',
                    help='displayarray window: current view | reconstruction')
    ap.add_argument('--policy',  default='curiosity', choices=['curiosity', 'random'])
    ap.add_argument('--compare', action='store_true',
                    help='run curiosity AND random policies, print side-by-side')
    ap.add_argument('--steps',   type=int,   default=2000)
    ap.add_argument('--timeout', type=float, default=60.0)
    ap.add_argument('--hidden',  type=int,   default=1024,
                    help='hidden size; forced to 32 for --core mistral '
                         '(folded weights are fixed-width)')
    ap.add_argument('--density', type=float, default=0.1,
                    help='connectivity fraction for --core sparse (default '
                         '10%%); ignored for dense/mistral')
    ap.add_argument('--num-cpus', type=int, default=1,
                    help='OpenMP threads for sparse layers. 1 is correct at '
                         'small hidden (thread-wakeup overhead exceeds any '
                         'FLOP savings); worth experimenting with at large '
                         '--hidden (1024+) where real compute starts to '
                         'dominate per-call overhead.')
    ap.add_argument('--lr',      type=float, default=0.01)
    ap.add_argument('--group',   type=int,   default=4)
    ap.add_argument('--report-every', type=int, default=200)
    a = ap.parse_args()
    kw = dict(lr=a.lr, group=a.group, report_every=a.report_every,
              agent=a.agent, display=a.display, hidden=a.hidden,
              density=a.density, num_cpus=a.num_cpus)
    if a.compare:
        compare(a.core, a.steps, a.timeout, **kw)
    else:
        run(core=a.core, policy=a.policy, max_steps=a.steps,
            timeout=a.timeout, **kw)


if __name__ == '__main__':
    main()
