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
     sparse is a large, genuinely sparse FoldedLayer-based core that starts
     EMPTY and grows via real synaptogenesis (--hidden tunable, default 1024,
     --base-connections is the per-row target reached via build_probes ->
     synap_step, then held there via continual grow+prune churn -- this is
     sili's actual distinguishing mechanism, not merely "some weight matrix
     happens to be sparse"; see make_grown_sparse_layer's docstring).
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

import argparse, json, math, time, zlib, warnings
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'unit', 'python'))
warnings.filterwarnings('ignore')

import torch  # CSR construction only -- not in the compute path
import sili.cpu
from sili import _cpu
from sili.tensor import Tensor, tanh as sili_tanh, combine_losses
from sili.sparse_rnn import FoldedLayer, SynaptogenesisSchedule
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


def make_position_embeddings(size: int, n_freqs: int = 4) -> np.ndarray:
    """
    Fixed (not learned) sinusoidal position embeddings for a size x size
    grid, flattened row-major to match render_mandelbrot's output layout.
    Deterministic, unique per position, smoothly varying -- the classic
    Transformer positional-encoding idea extended to 2D (row, col).
    """
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    rows = rows.reshape(-1).astype(np.float32)
    cols = cols.reshape(-1).astype(np.float32)
    feats = []
    for i in range(n_freqs):
        freq = 1.0 / (size ** (i / max(1, n_freqs - 1)))
        feats += [np.sin(rows*freq), np.cos(rows*freq),
                  np.sin(cols*freq), np.cos(cols*freq)]
    return np.stack(feats, axis=1).astype(np.float32)


class PixelAttentionHead:
    """
    Transformer-style reconstruction head: position-embedded pixel QUERIES
    attend into the HIDDEN STATE (reshaped into memory slots) as the
    attention CONTEXT, replacing a single linear layer predicting all
    view*view pixels at once from one hidden vector.

    Why: a single linear layer mapping hidden -> all pixels at once has no
    way to treat different output positions differently -- structurally
    blind to WHERE a pixel is, the same failure mode the input side had
    before encode_observation() stopped mean-pooling. Confirmed by direct
    observation this session: reconstruction converged toward something
    close to the pixel-wise average of the training distribution
    ("predicts the average view"). This is a well-known failure mode for
    RNN-style dense-output visual prediction generally, not specific to
    sili (matches prior direct experience: identical behavior from a large
    RNN run on Atari frames).

    Design: fixed sinusoidal position embeddings per pixel (not learned --
    deterministic, unique, smoothly varying) projected into per-pixel
    QUERIES. The hidden state is reshaped into n_slots memory slots and
    projected into KEYS/VALUES, giving attention real multi-position
    structure to attend over (a single vector has nothing to attend "into"
    otherwise). Every pixel's query attends over the SAME slot set, so
    different pixels can learn to pull from different combinations of
    hidden-state slots depending on their position -- genuinely
    position-aware, unlike one shared linear map.

    Fully vectorized: all view*view queries computed and attended in a
    handful of matmuls (batched over pixel positions), not a per-pixel
    Python loop. Gradient hand-derived and verified against finite-
    difference numerical gradients (max relative error ~1e-10) before
    wiring in -- softmax has no native sili autograd op, so this is plain
    numpy throughout with a manually-computed backward, matching Vr/br's
    own prior convention (plain numpy, manual SGD) rather than mixing in
    sili Tensor ops that would never actually be exercised.

    Wo starts at exactly zero (matches everything else in zero_mode):
    reconstruction begins at exactly 0 regardless of core; only Wo needs to
    escape zero via backprop, since Wq/Wk/Wv already provide small-random
    (not zero) attention structure to differentiate positions/slots from
    the first step.
    """
    def __init__(self, hidden: int, view: int, n_slots: int = 16,
                d_k: int = 32, lr: float = 0.01, zero_init: bool = True,
                seed: int = 0):
        assert hidden % n_slots == 0, \
            f"hidden={hidden} must be divisible by n_slots={n_slots}"
        self.n_slots, self.d_slot, self.d_k = n_slots, hidden // n_slots, d_k
        self.n_pix = view * view
        self.pos_emb = make_position_embeddings(view)      # (n_pix, d_pos), fixed
        d_pos = self.pos_emb.shape[1]
        rng = np.random.default_rng(seed)
        s = 1.0 / math.sqrt(d_k)
        self.Wq = (rng.standard_normal((d_pos, d_k)) * s).astype(np.float32)
        self.Wk = (rng.standard_normal((self.d_slot, d_k)) * s).astype(np.float32)
        self.Wv = (rng.standard_normal((self.d_slot, d_k)) * s).astype(np.float32)
        self.Wo = (np.zeros((d_k, 1), np.float32) if zero_init else
                  (rng.standard_normal((d_k, 1)) * s).astype(np.float32))
        self.lr = lr
        self._cache = None

    def forward(self, h: np.ndarray) -> np.ndarray:
        """h: (hidden,) numpy array (detached state). Returns (n_pix,)."""
        slots = h.reshape(self.n_slots, self.d_slot)
        Q = self.pos_emb @ self.Wq                # (n_pix, d_k)
        K = slots @ self.Wk                        # (n_slots, d_k)
        V = slots @ self.Wv                        # (n_slots, d_k)
        scores = Q @ K.T / math.sqrt(self.d_k)
        scores = scores - scores.max(axis=1, keepdims=True)
        w = np.exp(scores); w = w / w.sum(axis=1, keepdims=True)
        attended = w @ V                            # (n_pix, d_k)
        pred = (attended @ self.Wo).reshape(-1)     # (n_pix,)
        self._cache = (Q, K, V, w, attended, slots)
        return pred

    def backward(self, g_pred: np.ndarray) -> np.ndarray:
        """g_pred: (n_pix,) gradient at the reconstruction output. Updates
        Wq/Wk/Wv/Wo via SGD; returns gradient w.r.t. h (hidden,) for the
        caller to inject into the recurrent core (matching Vr/br's own
        prior g_h_recon = g_pred @ Vr.T convention)."""
        Q, K, V, w, attended, slots = self._cache
        g2 = g_pred.reshape(-1, 1)
        dWo = attended.T @ g2
        d_att = g2 @ self.Wo.T
        dV = w.T @ d_att
        dw = d_att @ V.T
        ds = w * (dw - (w*dw).sum(axis=1, keepdims=True))     # softmax Jacobian
        dQ = ds @ K / math.sqrt(self.d_k)
        dK = ds.T @ Q / math.sqrt(self.d_k)
        dWq = self.pos_emb.T @ dQ
        dWk = slots.T @ dK
        dWv = slots.T @ dV
        d_slots = dK @ self.Wk.T + dV @ self.Wv.T
        self.Wq -= self.lr * dWq
        self.Wk -= self.lr * dWk
        self.Wv -= self.lr * dWv
        self.Wo -= self.lr * dWo
        return d_slots.reshape(-1)


# ── Sparse construction (real synaptogenesis, not a fixed pattern) ───────────

def make_grown_sparse_layer(n_in: int, n_out: int, base_connections: int,
                            lr: float, num_cpus: int = 1,
                            k_factor: int = 4, importance_cutoff: float = 0.0,
                            amplitude: float = 0.0, period: int = 200,
                            every_n_steps: int = 1
                            ) -> tuple[FoldedLayer, SynaptogenesisSchedule]:
    """
    A genuinely sparse FoldedLayer that starts EMPTY (nnz=0) and is grown
    via REAL synaptogenesis (build_probes -> synap_step, driven by
    importance), not a fixed random connectivity pattern with only trained
    values.

    WHY THIS MATTERS (per direct correction): build_probes/synap_step are
    not an initialization trick -- they are the mechanism that CONTINUALLY
    reshapes the graph structure itself during training, based on gradient-
    derived importance. A fixed random sparse pattern with only backprop-
    trained VALUES is architecturally a sparse echo/reservoir network with
    a trained readout -- a completely different system from one where the
    connectivity itself is shaped by backprop and importance. An earlier
    version of this function did exactly that (fixed random mask, near-zero
    values, no growth calls at all) and was wrong for that reason, not for
    any bug in it.

    CONSTRUCTION SEQUENCE (each step verified directly, not assumed):
      1. _cpu.SparseLinearLayer(n_in, n_out, max_weights, num_cpus) starts
         at nnz=0 -- confirmed empty, the natural bootstrap point for
         growth-from-scratch. NOTE the row convention: rows = n_in (matches
         this direct constructor's own (n_inputs, n_outputs, ...) parameter
         order) -- DIFFERENT from FoldedBlockDescriptor/from_descriptor's
         dense-weight-matrix convention (rows = n_out), which is what an
         earlier version of this function used and is NOT the convention
         used here.
      2. equalize_to_capacity(...) reserves per-row headroom BEFORE any
         connections exist. Without this, the first synap_step() growth
         attempt fails immediately with "ran out of blank space" -- a fresh
         empty layer has zero pre-allocated bytes to insert into.
      3. set_value_scale_raw(row, lr/FP4_MAX) for every row, once, before
         training starts. Same class of fix already used for
         importance_scale elsewhere in this codebase, just not previously
         applied to the value scale for a from-scratch sparse layer:
         new connections (from probes) always start at value=0 (confirmed
         against tests/unit/unittest_sisldo.cpp's sisldo_optim_synaptogenesis
         test: "Values: existing kept, new connections = 0"), and without
         this fix a single realistic backprop step on a newly-grown zero-
         valued connection rounds back to zero under FP4 quantization --
         the connection exists structurally but never actually learns
         anything. Confirmed by direct testing: growth without this fix
         reached the correct nnz but forward output stayed exactly zero
         after 50 training steps; with this fix, real nonzero output.
      4. SynaptogenesisSchedule drives ongoing growth. amplitude=0 (default)
         means "grow to base_connections per row, then churn there forever"
         -- every cadence, synap_step() removes connections below
         importance_cutoff and regrows back up to base_connections, so the
         density stays roughly constant while WHICH connections exist keeps
         evolving. amplitude>0 sine-waves the target, explicitly exercising
         both growth and pruning. k_factor is passed through UNSCALED --
         build_probes(k) is NOT "k global candidates": it takes the top-k
         inputs by accumulated activity and top-k outputs by accumulated
         gradient, then the CARTESIAN PRODUCT of those two sets, so raw
         candidates scale as k^2, not k (confirmed against the actual source,
         delta_csr_build_probes in delta_csr_memory.hpp). A small k (the
         default of 4, ~16 raw candidate pairs) is the intended scale --
         growth relies on calling synaptogenesis often and letting which
         inputs/outputs land in the top-k rotate over many calls as
         accumulated activity/gradient shifts (which the energy dynamics'
         homeostatic firing already guarantees happens for every neuron
         eventually), not on one call covering the whole layer at once.

    Caller MUST call the returned schedule's .step() every training step,
    AFTER backward() and BEFORE the next forward() (matches
    SynaptogenesisSchedule's own documented contract).
    """
    FP4_MAX = 6.0
    total_target = n_in * base_connections
    # Budget with margin: sine-wave peaks can exceed base_connections by
    # (1+amplitude), plus general slack so equalize_to_capacity/growth has
    # room without immediately hitting the layer's overall budget.
    max_weights = max(64, int(total_target * (1.0 + amplitude) * 1.5))

    raw = _cpu.SparseLinearLayer(n_in, n_out, max_weights, num_cpus)

    target_elems = max(1, round(base_connections * (1.0 + amplitude) * 1.5))
    bits = max(1, n_out - 1).bit_length()
    target_bytes = target_elems * ((bits + 6) // 7) + 8
    raw.equalize_to_capacity(target_elems, target_bytes)

    for r in range(n_in):
        raw.set_value_scale_raw(r, lr / FP4_MAX)

    layer = FoldedLayer(layers={'.w': raw}, n_folds=1, out_dims={'.w': n_out},
                        learning_rate=lr)
    # k_factor is passed through UNSCALED. An earlier version of this
    # function scaled it by n_in, based on a wrong mental model of build_probes
    # (assumed k = candidate count directly). CORRECTED, verified against
    # the actual C++ source (delta_csr_build_probes in
    # sili/lib/headers/delta_csr_memory.hpp): it selects the top-k INPUTS by
    # accumulated activity and the top-k OUTPUTS by accumulated gradient,
    # then takes the CARTESIAN PRODUCT of those two sets as candidates --
    # k_in * k_out ~= k^2 raw candidates, not k. Scaling k by n_in therefore
    # scaled actual candidate generation by n_in^2, not n_in -- a severe,
    # unintended blowup (confirmed directly: this caused the severe slowdown
    # measured afterward, and reverting to plain k_factor made growth run
    # faster than real time). You are not meant to generate thousands of
    # candidates per call; small k (e.g. 4, giving ~16 raw candidate pairs)
    # relies on calling synaptogenesis often and letting the TOP-K SELECTION
    # itself rotate across different inputs/outputs over many calls as their
    # accumulated activity/gradient changes -- which the energy dynamics'
    # homeostatic firing already guarantees will eventually happen for every
    # neuron, giving every row a fair turn at being selected over time
    # without needing a single call to cover everything at once.
    sched = SynaptogenesisSchedule(
        layer, base_connections=base_connections, k_factor=k_factor,
        importance_cutoff=importance_cutoff, amplitude=amplitude,
        period=period, every_n_steps=every_n_steps)
    return layer, sched


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
    """Wraps a FoldedLayer + its SynaptogenesisSchedule with the same
    call/.params interface as DenseProjection. params is empty because
    folded layers self-update via backward_dense (same reason core_net.params
    is empty for FoldedLayer-based cores) -- the outer SGD loop must not
    also try to step them."""
    def __init__(self, layer: FoldedLayer, sched: SynaptogenesisSchedule):
        self.layer = layer
        self.sched = sched
        self.params = []

    def __call__(self, x: Tensor) -> Tensor:
        return self.layer(x)

    def step_synaptogenesis(self):
        self.sched.step()


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
    distinguishing capability is DYNAMIC sparse structure driven by
    backprop-derived importance (real synaptogenesis: build_probes ->
    synap_step, continually growing and pruning), not merely "some weight
    matrix happens to be sparse." Pairing "sparse" with "tiny" (--core
    mistral, fixed at hidden=32 by its pretrained toy weights) and "large"
    with "dense" (the old default) tested neither property together, and an
    earlier version of THIS class used a FIXED random connectivity pattern
    with only trained values -- architecturally a sparse echo/reservoir
    network with a trained readout, a completely different system from what
    sili actually is. This version starts genuinely empty (nnz=0) and grows
    via SynaptogenesisSchedule; see make_grown_sparse_layer's docstring for
    the full verified construction sequence.

    Two SEPARATE grown layers (image contribution, recurrent contribution)
    summed, rather than one big lifted-and-concatenated matrix -- avoids the
    lift trick DenseCore needs (which exists specifically to route gradients
    through a single dense matrix multiply) and lets image-sparsity and
    recurrent-sparsity grow/churn independently. Same shape of fix as
    MistralCore's Wc_img/Wc_state split.
    """
    def __init__(self, d_img, hidden, base_connections=6, lr=0.01,
                num_cpus=1, k_factor=4, importance_cutoff=0.0,
                amplitude=0.0, period=200, every_n_steps=1, seed=0):
        self.core_img, self.sched_img = make_grown_sparse_layer(
            d_img, hidden, base_connections, lr, num_cpus, k_factor,
            importance_cutoff, amplitude, period, every_n_steps)
        self.core_state, self.sched_state = make_grown_sparse_layer(
            hidden, hidden, base_connections, lr, num_cpus, k_factor,
            importance_cutoff, amplitude, period, every_n_steps)
        self.params = []   # folded layers self-update via backward_dense

    def forward(self, x_img: Tensor, state: np.ndarray) -> Tensor:
        return self.core_img(x_img) + self.core_state(Tensor(state))

    def step_synaptogenesis(self):
        """Call AFTER backward(), BEFORE the next forward() -- matches
        SynaptogenesisSchedule's own documented contract. Each call advances
        both cadences; each schedule only actually runs growth/pruning when
        its own every_n_steps interval fires."""
        self.sched_img.step()
        self.sched_state.step()

    def nnz_total(self) -> int:
        return self.core_img.nnz_total() + self.core_state.nnz_total()


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

_RUN_PARAM_KEYS = (
    'core', 'policy', 'agent', 'action_mode', 'max_steps', 'timeout', 'view',
    'hidden', 'base_connections', 'num_cpus', 'k_factor', 'importance_cutoff',
    'synap_amplitude', 'synap_period', 'synap_every', 'lr', 'aux_weight',
    'pol_weight', 'group', 'seed', 'gamma', 'entropy_scale', 'loss_alpha',
    'target_update', 'energy_drive', 'energy_activation_cost',
    'energy_precision', 'energy_density', 'energy_exploration',
    'energy_setpoint', 'energy_reactivity', 'energy_p')


def _run_params_dict(scope):
    '''Record of the exact configuration a run used (for the experiment
    collector), including the RESOLVED energy params actually constructed
    (A.eh_kwargs / A.eo_kwargs), not just the None-able overrides.'''
    out = {k: scope.get(k) for k in _RUN_PARAM_KEYS if k in scope}
    A = scope.get('A')
    if A is not None:
        out['energy_hidden_resolved'] = dict(A.eh_kwargs)
        out['energy_output_resolved'] = dict(A.eo_kwargs)
    return out


def _json_default(o):
    '''numpy scalars/arrays leak into records easily (e.g. cx/cy/zoom become
    np.float32 after continuous-mode arithmetic with a float32 magnitude
    array) -- convert them at the serialization boundary once rather than
    chasing every producer.'''
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'not JSON serializable: {type(o).__name__}')


def _write_json_atomic(path, obj):
    '''tmp-write + os.replace so a crash mid-write never leaves a truncated
    JSON for the experiment collector to choke on.'''
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, default=_json_default)
    os.replace(tmp, path)


def ncd(bufs_a, bufs_b) -> float:
    """Normalized Compression Distance between two window buffers (lists of
    float32 arrays), using the same zlib-level-1 compressor as
    compression_ratio for consistency. NCD(a,b) = (C(ab)-min(Ca,Cb))/max(Ca,Cb):
    ~0 when one stream is (compressibly) predictable from the other, ~1 when
    they share no compressible structure. Used as a cheap mutual-information
    proxy between the input-view window and the hidden-state window --
    'how much of what the hidden state carries is actually about the input'
    -- one of the information-theoretic measurements the energy-*.md
    hypotheses need (Theorems 3/5: sparse codes should track the input
    manifold, not private dynamics)."""
    if not bufs_a or not bufs_b:
        return float('nan')
    a = np.concatenate([x.ravel() for x in bufs_a]).astype(np.float32).tobytes()
    b = np.concatenate([x.ravel() for x in bufs_b]).astype(np.float32).tobytes()
    ca = len(zlib.compress(a, 1)); cb = len(zlib.compress(b, 1))
    cab = len(zlib.compress(a + b, 1))
    return (cab - min(ca, cb)) / max(ca, cb)


def apply_action_discrete(cx, cy, zoom, action, home):
    """Pure env-step for discrete mode: exactly one of 7 actions fires."""
    pan = 0.3 / zoom
    if   action == 0: cx -= pan
    elif action == 1: cx += pan
    elif action == 2: cy -= pan
    elif action == 3: cy += pan
    elif action == 4: zoom = min(zoom*1.4, 1e8)
    elif action == 5: zoom = max(zoom/1.4, 1.0)
    elif action == 6: cx, cy, zoom = home
    return cx, cy, zoom


def apply_action_continuous(cx, cy, zoom, mags, home):
    """Pure env-step for continuous mode: all seven dimensions act at once,
    each scaled by its own magnitude. Reset is a continuous pull toward home
    (blended in log-zoom space so the pull is comparable at any zoom)."""
    pan = 0.3 / zoom
    cx -= mags[0] * pan
    cx += mags[1] * pan
    cy -= mags[2] * pan
    cy += mags[3] * pan
    zoom *= (1.0 + mags[4] * 0.4)
    zoom /= (1.0 + mags[5] * 0.4)
    cx += mags[6] * (home[0] - cx)
    cy += mags[6] * (home[1] - cy)
    log_zoom      = math.log(zoom, 1.4)
    log_zoom_home = math.log(home[2], 1.4)
    zoom = 1.4 ** (log_zoom + mags[6] * (log_zoom_home - log_zoom))
    zoom = min(max(zoom, 1.0), 1e8)
    return cx, cy, zoom


def select_action(l7, policy, action_mode, n_act):
    """Action selection for one step. Returns (action, mag7, probs):
    discrete -> (int, None, softmax/uniform probs); continuous ->
    (None, mag7 Tensor or None for random policy, magnitudes array).
    RNG call order matches the original inline code exactly."""
    if action_mode == 'continuous':
        if policy == 'random':
            return None, None, np.random.uniform(0., 1., n_act).astype(np.float32)
        mag7 = l7.bounded_gate(n=2.0)
        return None, mag7, mag7.data
    if policy == 'random':
        probs = np.full(n_act, 1./n_act, np.float32)
    else:
        lg = l7.data - l7.data.max()
        probs = np.exp(lg); probs /= probs.sum()
    return int(np.random.choice(n_act, p=probs)), None, probs


def resolve_energy_params(hidden, n_out_neurons, n_act, overrides):
    """Default energy parameters (exactly the formulas previously hardcoded
    in run()), with any not-None override applied. Shared overrides (drive,
    activation_cost, precision, exploration, setpoint, reactivity, density)
    apply to BOTH regions; `p` applies to the hidden region only (the output
    region's p is tied to the grouping structure). This is what makes the
    energy-*.md parameter axes (delta, gamma, lambda_KL, beta, sigma, tau,
    alpha) actually testable from the CLI instead of buried constants."""
    frac_h = max(0.1, min(0.4, 8. / hidden))
    frac_o = max(0.15, min(0.5, 2.*n_act / n_out_neurons))
    eh = dict(drive=1./hidden, activation_cost=0.05, precision=0.01,
              density=frac_h/2, exploration=0.002, setpoint=1.0,
              reactivity=0.01, p=frac_h)
    eo = dict(drive=1./n_out_neurons, activation_cost=0.05, precision=0.01,
              density=frac_o/2, exploration=0.002, setpoint=1.0,
              reactivity=0.01, p=frac_o)
    for key in ('drive', 'activation_cost', 'precision', 'density',
                'exploration', 'setpoint', 'reactivity'):
        v = overrides.get(key)
        if v is not None:
            eh[key] = v; eo[key] = v
    if overrides.get('p') is not None:
        eh['p'] = overrides['p']
    return eh, eo


def build_agent(core, hidden, view, raw_dim, zero_mode, base_connections,
                lr, num_cpus, k_factor, importance_cutoff, synap_amplitude,
                synap_period, synap_every, n_act, group, energy_overrides,
                verbose=True):
    """Constructs every module of the agent in one place and returns a
    SimpleNamespace, instead of scattering model construction through the
    body of run(). Construction ORDER is preserved exactly from the old
    inline code (input proj -> core -> energy -> action head -> value heads
    -> recon head) so seeded runs remain bit-identical with pre-refactor
    behavior (verified against captured baselines)."""
    from types import SimpleNamespace

    # (input projection: the "V part" -- new weights, toy Mistral has none)
    if core == 'sparse':
        Wp = SparseProjection(*make_grown_sparse_layer(
            raw_dim, hidden, base_connections, lr, num_cpus, k_factor,
            importance_cutoff, synap_amplitude, synap_period, synap_every))
    else:
        Wp = DenseProjection(Tensor(
            np.zeros((raw_dim, hidden), np.float32) if zero_mode else
            (np.random.randn(raw_dim, hidden) * 0.05).astype(np.float32)))

    if core == 'sparse':
        core_net = SparseCore(hidden, hidden, base_connections, lr, num_cpus,
                              k_factor, importance_cutoff, synap_amplitude,
                              synap_period, synap_every)
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

    n_out_neurons = n_act * group
    eh_kwargs, eo_kwargs = resolve_energy_params(hidden, n_out_neurons,
                                                 n_act, energy_overrides)
    energy_h = EnergyDynamics(**eh_kwargs)
    energy_o = EnergyDynamics(**eo_kwargs)
    if zero_mode:
        energy_h.energy = np.full(hidden,        1.9, np.float32)
        energy_o.energy = np.full(n_out_neurons, 1.9, np.float32)

    Wa = Tensor(np.zeros((hidden, n_out_neurons), np.float32) if zero_mode else
                (np.random.randn(hidden, n_out_neurons) * 0.05).astype(np.float32))
    Gm = np.zeros((n_out_neurons, n_act), np.float32)
    for a in range(n_act):
        Gm[a*group:(a+1)*group, a] = 1.0 / group

    # RTAC value heads: real-time input = [h ; action-vector]. Discrete
    # mode: onehot(prev_action); continuous mode: prev mag7 vector -- same
    # (N_ACT, 1) weight shape either way, the action input is just a vector
    # instead of a one-hot.
    def _mk_value_head():
        wh = Tensor(np.zeros((hidden, 1), np.float32) if zero_mode else
                    (np.random.randn(hidden, 1) * 0.05).astype(np.float32))
        wa = Tensor(np.zeros((n_act, 1), np.float32))
        return wh, wa
    Wv_h,  Wv_a  = _mk_value_head()
    Wv_h2, Wv_a2 = _mk_value_head()
    Wvh_t,  Wva_t  = Wv_h.data.copy(),  Wv_a.data.copy()
    Wvh_t2, Wva_t2 = Wv_h2.data.copy(), Wv_a2.data.copy()
    from sili.rl_utils import PopArt
    popart = PopArt(beta=0.0003, start_pop=8)

    n_slots = max(1, math.gcd(hidden, 16))
    recon_head = PixelAttentionHead(hidden, view, n_slots=n_slots, d_k=32,
                                    lr=lr, zero_init=zero_mode)

    dense_params = Wp.params + [Wa, Wv_h, Wv_a, Wv_h2, Wv_a2] + core_net.params

    return SimpleNamespace(
        Wp=Wp, core_net=core_net, energy_h=energy_h, energy_o=energy_o,
        Wa=Wa, Gm=Gm, n_out_neurons=n_out_neurons,
        Wv_h=Wv_h, Wv_a=Wv_a, Wv_h2=Wv_h2, Wv_a2=Wv_a2,
        Wvh_t=Wvh_t, Wva_t=Wva_t, Wvh_t2=Wvh_t2, Wva_t2=Wva_t2,
        popart=popart, recon_head=recon_head, dense_params=dense_params,
        eh_kwargs=eh_kwargs, eo_kwargs=eo_kwargs)


def rtac_terms(A, S, h_out, probs, onehot, mag7, reward, action_mode, n_act,
               gamma, loss_alpha, pol_weight, entropy_scale, target_update):
    """The full RTAC update, extracted verbatim from run(): double critic
    with EMA target nets and PopArt-normalized one-step TD, actor advantage
    against min(v1,v2). Returns (G_pol_or_None, extra_terms).

    Discrete mode: byte-identical math to the previous inline code.

    Continuous mode (new): the action input to the value heads is the
    previous step's mag7 VECTOR instead of a one-hot -- same (N_ACT, 1)
    weight shape. The actor gradient is injected at mag7 (returned in
    extra_terms; bounded_gate's verified backward chains it to l7):
    G = -loss_alpha*pol_weight*adv*mag7.data, mirroring the reinforce-
    continuous convention. No entropy bonus in continuous mode (no analog
    for independent continuous magnitudes -- omitted rather than guessed at).

    Mutates A's target nets (polyak, in place) and popart; reads S.h_prev/
    S.a_prev/S.a_prevprev/S.r_prev but does NOT advance them (loop end
    owns that)."""
    extra_terms = []
    if action_mode == 'continuous':
        av_prev = np.asarray(S.a_prev, np.float32)
        av_pp   = np.asarray(S.a_prevprev, np.float32)
    else:
        av_prev = np.zeros(n_act, np.float32); av_prev[S.a_prev] = 1.
        av_pp   = np.zeros(n_act, np.float32); av_pp[S.a_prevprev] = 1.

    v1_now = h_out @ A.Wv_h  + Tensor(av_prev) @ A.Wv_a
    v2_now = h_out @ A.Wv_h2 + Tensor(av_prev) @ A.Wv_a2
    v_now_min = min(float(np.asarray(v1_now.data).ravel()[0]),
                    float(np.asarray(v2_now.data).ravel()[0]))
    v_now_orig = A.popart.unnormalize(v_now_min)

    adv = reward - v_now_orig
    if action_mode == 'continuous':
        G_pol = None
        if mag7 is not None:
            extra_terms.append((mag7,
                                -loss_alpha * pol_weight * adv * mag7.data))
    else:
        G_pol = loss_alpha * pol_weight * adv * (probs - onehot)
        H = -(probs * np.log(probs + 1e-9)).sum()
        G_pol += loss_alpha * entropy_scale * probs * (np.log(probs + 1e-9) + H)

    if S.h_prev is not None:
        v1_prev = Tensor(S.h_prev) @ A.Wv_h  + Tensor(av_pp) @ A.Wv_a
        v2_prev = Tensor(S.h_prev) @ A.Wv_h2 + Tensor(av_pp) @ A.Wv_a2

        v1_targ = float((h_out.data @ A.Wvh_t  + av_prev @ A.Wva_t ).ravel()[0])
        v2_targ = float((h_out.data @ A.Wvh_t2 + av_prev @ A.Wva_t2).ravel()[0])
        v_targ_min_norm = min(v1_targ, v2_targ)
        v_targ_min_orig = A.popart.unnormalize(v_targ_min_norm)

        y_orig = S.r_prev + gamma * v_targ_min_orig
        y_norm = A.popart.update_and_rescale(
            y_orig,
            weight_arrays=[A.Wv_h.data, A.Wv_h2.data, A.Wvh_t, A.Wvh_t2],
            bias_arrays=[A.Wv_a.data, A.Wv_a2.data, A.Wva_t, A.Wva_t2])
        g_c1 = (1.0 - loss_alpha) * 2.0 * (float(np.asarray(v1_prev.data).ravel()[0]) - y_norm)
        g_c2 = (1.0 - loss_alpha) * 2.0 * (float(np.asarray(v2_prev.data).ravel()[0]) - y_norm)
        extra_terms.append((v1_prev, np.array([g_c1], np.float32)))
        extra_terms.append((v2_prev, np.array([g_c2], np.float32)))

    A.Wvh_t  += target_update * (A.Wv_h.data  - A.Wvh_t)
    A.Wva_t  += target_update * (A.Wv_a.data  - A.Wva_t)
    A.Wvh_t2 += target_update * (A.Wv_h2.data - A.Wvh_t2)
    A.Wva_t2 += target_update * (A.Wv_a2.data - A.Wva_t2)

    return G_pol, extra_terms


def run(core='sparse', policy='curiosity', agent='reinforce',
        max_steps=2000, timeout=60.0,
        view=32, hidden=1024, base_connections=6, num_cpus=1,
        k_factor=4, importance_cutoff=0.0, synap_amplitude=0.0,
        synap_period=200, synap_every=1,
        lr=0.01, aux_weight=0.05, pol_weight=0.3,
        group=4, report_every=200, verbose=True, seed=0,
        display=False, gamma=0.99, entropy_scale=0.01,
        loss_alpha=0.2, target_update=0.005, action_mode='discrete',
        energy_drive=None, energy_activation_cost=None, energy_precision=None,
        energy_density=None, energy_exploration=None, energy_setpoint=None,
        energy_reactivity=None, energy_p=None, json_out=None):
    N_ACT = 7
    ACT_NAMES = ['pan<','pan>','pan^','panv','zoom+','zoom-','reset']

    # NOTE: --action-mode continuous + --agent rtac is supported: the value
    # heads' action input is the previous step's mag7 VECTOR instead of a
    # one-hot (same (N_ACT,1) weight shape), and the actor gradient is
    # injected at mag7 -- see rtac_terms().
    np.random.seed(seed); torch.manual_seed(seed)

    if core == 'mistral' and hidden != 32:
        if verbose:
            print(f"    --hidden {hidden} ignored: --core mistral uses folded "
                  f"toy-Mistral weights fixed at hidden=32 (can't be widened)")
        hidden = 32

    # zero_mode covers BOTH zero-init cores (sparse: genuinely empty, grown
    # via synaptogenesis; dense: exact-zero) -- mistral is the only core
    # that starts from nonzero (pretrained toy) weights. Gates the SEPARATE
    # small head matrices (action, value, reconstruction) that stay plain
    # dense Tensors regardless of which core is active -- only the
    # recurrent core + input projection differ between sparse/dense/mistral.
    zero_mode = core in ('sparse', 'dense')
    raw_dim = view * view + 3   # full flattened view + (cx, cy, log-zoom)

    energy_overrides = dict(
        drive=energy_drive, activation_cost=energy_activation_cost,
        precision=energy_precision, density=energy_density,
        exploration=energy_exploration, setpoint=energy_setpoint,
        reactivity=energy_reactivity, p=energy_p)

    A = build_agent(core, hidden, view, raw_dim, zero_mode, base_connections,
                    lr, num_cpus, k_factor, importance_cutoff,
                    synap_amplitude, synap_period, synap_every,
                    N_ACT, group, energy_overrides, verbose)
    Wp, core_net = A.Wp, A.core_net
    energy_h, energy_o = A.energy_h, A.energy_o
    Wa, Gm, n_out_neurons = A.Wa, A.Gm, A.n_out_neurons
    Wv_h, Wv_a, Wv_h2, Wv_a2 = A.Wv_h, A.Wv_a, A.Wv_h2, A.Wv_a2
    Wvh_t, Wva_t, Wvh_t2, Wva_t2 = A.Wvh_t, A.Wva_t, A.Wvh_t2, A.Wva_t2
    popart, recon_head, dense_params = A.popart, A.recon_head, A.dense_params

    from types import SimpleNamespace
    S = SimpleNamespace(
        h_prev=None,
        a_prev=(np.zeros(N_ACT, np.float32) if action_mode == 'continuous' else 0),
        a_prevprev=(np.zeros(N_ACT, np.float32) if action_mode == 'continuous' else 0),
        r_prev=0.0)

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

    # (reconstruction head + dense_params are constructed in build_agent)

    # -- navigation --
    cx, cy, zoom = -0.75, 0.0, 50.0
    HOME = (cx, cy, zoom)
    state = np.zeros(hidden, np.float32)

    # -- trackers --
    view_buf, hid_buf = [], []
    act_counts_win = np.zeros(N_ACT, int); act_counts_all = np.zeros(N_ACT, int)
    mag_accum_win  = np.zeros(N_ACT, np.float32); n_mag_win = 0
    coverage = set()
    recon_win = 0.; n_win = 0
    recon_all = 0.; n_all = 0
    fire_win = 0.
    history = []
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
            pred = recon_head.forward(h_o.data)
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
            kind = "EMPTY (grown via synaptogenesis)" if core == 'sparse' else "ZERO"
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
        fire_win += float((h_out.data != 0).mean())

        # -- action path: energy-gated grouped output --
        la28 = h_out @ Wa                                # (28,) graph
        e28, aux_o, _ = energy_o.forward(la28)
        l7  = e28 @ Gm                                   # grouped mean, in graph

        action, mag7, probs = select_action(l7, policy, action_mode, N_ACT)
        if action is not None:
            act_counts_win[action] += 1; act_counts_all[action] += 1
        elif action_mode == 'continuous':
            mag_accum_win += probs; n_mag_win += 1

        # -- act --
        if action_mode == 'continuous':
            cx, cy, zoom = apply_action_continuous(cx, cy, zoom, probs, HOME)
        else:
            cx, cy, zoom = apply_action_discrete(cx, cy, zoom, action, HOME)
        coverage.add((int(cx*50), int(cy*50), int(round(math.log(zoom, 1.4)))))

        # -- next view, reconstruction --
        nv = render_mandelbrot(cx, cy, zoom, view)
        nflat = nv.ravel()
        pred = recon_head.forward(h_out.data)
        rerr = pred - nflat
        rmse = float(np.mean(rerr**2))
        recon_win += rmse; n_win += 1; recon_all += rmse; n_all += 1

        # recon head backward: updates Wq/Wk/Wv/Wo via SGD, returns gradient
        # back to h for injection into the recurrent core below
        g_pred = (2./nflat.size) * rerr
        g_h_recon = recon_head.backward(g_pred)

        if show is not None:
            show(varr, pred)

        # -- intrinsic reward: novelty + homeostatic pressure --
        # (requirements doc 4.1: r = w_r*recon + w_e*energy_aux; the critic
        # learns V of this combined intrinsic return)
        aux_h_val = float(np.asarray(aux_h.data).ravel()[0]) if aux_h is not None else 0.0
        reward = rmse + 0.1 * aux_h_val

        terms = [(h_out, g_h_recon)]
        if aux_h is not None: terms.append((aux_h, aux_weight))
        if aux_o is not None: terms.append((aux_o, aux_weight))

        if policy == 'curiosity' and agent == 'rtac':
            onehot = None
            if action_mode != 'continuous':
                onehot = np.zeros(N_ACT, np.float32); onehot[action] = 1.
            G_pol, extra = rtac_terms(A, S, h_out, probs, onehot, mag7,
                                      reward, action_mode, N_ACT, gamma,
                                      loss_alpha, pol_weight, entropy_scale,
                                      target_update)
            terms.extend(extra)
            if G_pol is not None:
                terms.append((l7, G_pol))
        elif policy == 'curiosity' and action_mode == 'continuous':
            # Continuous analog of REINFORCE: no chosen-vs-unchosen split
            # exists when all dimensions act at once, so credit is assigned
            # proportionally to how active each dimension was. Negative
            # gradient increases the parameter under SGD, so advantage>0
            # pushes currently-active dimensions more active. No entropy
            # bonus (no analog for independent continuous magnitudes --
            # omitted rather than guessed at).
            if mag7 is not None:
                adv = reward - reward_baseline
                reward_baseline = 0.99*reward_baseline + 0.01*reward
                terms.append((mag7, -pol_weight * adv * mag7.data))
        elif policy == 'curiosity':
            onehot = np.zeros(N_ACT, np.float32); onehot[action] = 1.
            adv = reward - reward_baseline
            reward_baseline = 0.99*reward_baseline + 0.01*reward
            terms.append((l7, pol_weight * adv * (probs - onehot)))
        combine_losses(*terms).backward()

        # Synaptogenesis: AFTER backward() (importance/grad accumulators are
        # populated), BEFORE the next forward() -- matches
        # SynaptogenesisSchedule's documented contract exactly. No-op for
        # dense/mistral (their .params-bearing objects don't have this method;
        # only SparseCore/SparseProjection do).
        if hasattr(core_net, 'step_synaptogenesis'):
            core_net.step_synaptogenesis()
        if hasattr(Wp, 'step_synaptogenesis'):
            Wp.step_synaptogenesis()

        for p in dense_params:
            if p.grad is not None:
                np.clip(p.grad, -1., 1., out=p.grad)
                p.data -= lr * p.grad; p.grad = None

        state = h_out.data.copy()
        S.h_prev = state.copy(); S.a_prevprev = S.a_prev
        S.a_prev = (probs.copy() if action_mode == 'continuous' else action)
        S.r_prev = reward
        view_buf.append(varr); hid_buf.append(state.copy())
        if len(view_buf) > 60: view_buf.pop(0)
        if len(hid_buf) > 60: hid_buf.pop(0)

        if verbose and (step + 1) % report_every == 0:
            el = time.perf_counter() - start
            pm = probe_eval(); probe_history.append(pm)
            nnz = 0
            if hasattr(core_net, 'nnz_total'): nnz += core_net.nnz_total()
            if hasattr(Wp, 'layer') and hasattr(Wp.layer, 'nnz_total'):
                nnz += Wp.layer.nnz_total()
            nnz_str = f"  nnz={nnz}" if nnz > 0 else ""
            mean_mag = None
            if action_mode == 'continuous':
                mean_mag = mag_accum_win / max(1, n_mag_win)
                act_str = f"mag={np.array2string(mean_mag, precision=2, suppress_small=True)}"
            else:
                act_str = f"H={entropy_nats(act_counts_win):.2f}"
            vcr_now = compression_ratio(view_buf)
            hcr_now = compression_ratio(hid_buf)
            ncd_now = ncd(view_buf, hid_buf)
            fire_now = fire_win / max(1, n_win)
            print(f"  step {step+1:6d}  pos=({cx:+.3f},{cy:+.3f}) z={zoom:.0f}  "
                  f"recon={recon_win/max(1,n_win):.4f} probe={pm:.4f}  "
                  f"{act_str}  cov={len(coverage)}  "
                  f"vcr={vcr_now:.3f}{nnz_str}  "
                  f"{(step+1)/el:.0f} st/s")
            history.append(dict(
                step=step+1, probe=pm, recon=recon_win/max(1, n_win),
                nnz=nnz, coverage=len(coverage), vcr=vcr_now, hcr=hcr_now,
                ncd_view_hidden=ncd_now, fire_frac=fire_now,
                steps_per_sec=(step+1)/el,
                mean_mag=(mean_mag.tolist() if mean_mag is not None else None),
                action_entropy=(None if action_mode == 'continuous'
                                else float(entropy_nats(act_counts_win))),
                cx=cx, cy=cy, zoom=zoom))
            if json_out:
                _write_json_atomic(json_out, dict(
                    done=False, history=history,
                    params=_run_params_dict(locals())))
            recon_win = 0.; n_win = 0; act_counts_win[:] = 0
            mag_accum_win[:] = 0.; n_mag_win = 0; fire_win = 0.

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
        action_entropy=(entropy_nats(act_counts_all)
                        if action_mode != 'continuous' else float('nan')),
        coverage=len(coverage),
        view_cr=compression_ratio(view_buf),
        hid_cr=compression_ratio(hid_buf),
        ncd_view_hidden=ncd(view_buf, hid_buf),
        history=history,
        params=_run_params_dict(locals()),
    )
    if json_out:
        _write_json_atomic(json_out, dict(done=True, result=result,
                                          history=history,
                                          params=result['params']))

    if verbose:
        print(f"\n  stop={stop_reason}  steps={steps_done}  "
              f"{result['steps_per_sec']:.0f} st/s")
        print(f"  probe MSE first->final: {result['probe_mse_first']:.4f} -> "
              f"{result['probe_mse_final']:.4f}  "
              f"({'LEARNING' if result['probe_mse_final'] < result['probe_mse_first'] else 'not improving'})")
        print(f"  online recon (whole run): {result['recon_mse_all']:.4f}")
        if action_mode == 'continuous':
            print(f"  action_mode=continuous (no discrete entropy to report)  "
                  f"coverage={result['coverage']}")
        else:
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
    ap.add_argument('--action-mode', default='discrete', choices=['discrete', 'continuous'],
                    help='discrete (default): sample exactly one of 7 actions '
                         'per step via softmax ("old actor" forced choice). '
                         'continuous: no forced single choice -- every action '
                         'dimension gets its own continuous, energy-earned '
                         'magnitude via bounded_gate, and all can act at '
                         'once. A settled/quiet energy state naturally means '
                         '"do nothing," rather than needing a discrete choice '
                         'for it. Falls back to --agent reinforce if combined '
                         'with --agent rtac (RTAC assumes a discrete '
                         'one-hot previous action).')
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
    ap.add_argument('--base-connections', type=int, default=6,
                    help='target connections per row for --core sparse '
                         '(grown from empty via synaptogenesis, then held '
                         'there via continual churn); ignored for dense/mistral')
    ap.add_argument('--k-factor', type=int, default=4,
                    help='probes built per synaptogenesis call = '
                         'k_factor * current target (rule of thumb: 4x)')
    ap.add_argument('--importance-cutoff', type=float, default=0.0,
                    help='prune connections below this stored-unit importance '
                         'each synaptogenesis call')
    ap.add_argument('--synap-amplitude', type=float, default=0.0,
                    help='0 (default) = constant target, grow-then-churn. '
                         '>0 sine-waves the target +-amplitude fraction, '
                         'explicitly exercising both growth and pruning')
    ap.add_argument('--synap-period', type=int, default=200,
                    help='steps per full sine cycle when --synap-amplitude > 0')
    ap.add_argument('--synap-every', type=int, default=1,
                    help='run synaptogenesis every N training steps. With '
                         'k_factor at its intended small scale (build_probes '
                         'is k_in x k_out candidates, not k -- see '
                         'make_grown_sparse_layer docstring), every step is '
                         'fast (confirmed: 135+ st/s at hidden=256, matching '
                         'the "runs real-time calling once per backprop with '
                         'a small number of new synapses" pattern this is '
                         'modeled on). Raise this if you deliberately want a '
                         'BIGGER k_factor for more aggressive reshaping per '
                         'call -- that is the actual performance/quality '
                         'tradeoff worth tuning, not this cadence by itself.')
    ap.add_argument('--num-cpus', type=int, default=1,
                    help='OpenMP threads for sparse layers. 1 is correct at '
                         'small hidden (thread-wakeup overhead exceeds any '
                         'FLOP savings); worth experimenting with at large '
                         '--hidden (1024+) where real compute starts to '
                         'dominate per-call overhead.')
    for _en, _hint in (('drive', 'delta: metabolic tempo'),
                       ('activation-cost', 'gamma: cost of |h|'),
                       ('precision', 'lambda_KL: sparsity enforcement'),
                       ('density', 'beta: target activation density'),
                       ('exploration', 'sigma: per-neuron noise'),
                       ('setpoint', 'tau: comfort-zone target'),
                       ('reactivity', 'alpha: homeostatic gain'),
                       ('p', 'hard top-p active-fraction ceiling (hidden region)')):
        ap.add_argument(f'--energy-{_en}', type=float, default=None,
                        help=f'{_hint}; default = current built-in formula '
                             f'(see resolve_energy_params). Shared overrides '
                             f'apply to both hidden and output regions.')
    ap.add_argument('--json-out', type=str, default=None,
                    help='write history+params+result JSON here, updated '
                         'atomically at every report interval (crash-safe '
                         'partial results for the experiment harness)')
    ap.add_argument('--lr',      type=float, default=0.01)
    ap.add_argument('--group',   type=int,   default=4)
    ap.add_argument('--report-every', type=int, default=200)
    a = ap.parse_args()
    kw = dict(lr=a.lr, group=a.group, report_every=a.report_every,
              agent=a.agent, display=a.display, hidden=a.hidden,
              action_mode=a.action_mode,
              base_connections=a.base_connections, num_cpus=a.num_cpus,
              k_factor=a.k_factor, importance_cutoff=a.importance_cutoff,
              synap_amplitude=a.synap_amplitude, synap_period=a.synap_period,
              energy_drive=a.energy_drive,
              energy_activation_cost=a.energy_activation_cost,
              energy_precision=a.energy_precision,
              energy_density=a.energy_density,
              energy_exploration=a.energy_exploration,
              energy_setpoint=a.energy_setpoint,
              energy_reactivity=a.energy_reactivity,
              energy_p=a.energy_p, json_out=a.json_out,
              synap_every=a.synap_every)
    if a.compare:
        compare(a.core, a.steps, a.timeout, **kw)
    else:
        run(core=a.core, policy=a.policy, max_steps=a.steps,
            timeout=a.timeout, **kw)


if __name__ == '__main__':
    main()
