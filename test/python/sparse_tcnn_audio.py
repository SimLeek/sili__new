"""
sparse_tcnn_audio.py
─────────────────────
Sparse temporal CNN audio encoder using SILi DISLDO conv layers.

Architecture
────────────
Three conv layers with increasing temporal context, each followed by
top-k energy sparsity.  All layers start from zero weights; synaptogenesis
grows connections where input × gradient activity is concentrated.

    Audio chunk  [T × C_in]
         │  Layer 1: kT=KT1, stride=ST1  →  [T1 × H1]  top-k sparse
         │  Layer 2: kT=KT2, stride=ST2  →  [T2 × H2]  top-k sparse
         │  Layer 3: kT=KT3, stride=ST3  →  [T3 × H3]  top-k sparse
         │
         └── Concat activations → 2D array  [max_T × total_H]
             (rows = time positions, cols = all feature channels, zero-padded)
             Printed to stdout — wrap in your own display/logger.

Why TCNN without a conv op
───────────────────────────
Each layer is a SparseLinearLayer where n_inputs = kT × C_in.
The conv1d_forward call strides a pointer through the input without copying
(1-D windows are contiguous in memory).  Reshape/ravel only happen on the
output, not the input.

Sparsity and learning
──────────────────────
- Activation sparsity: top-k by absolute value after each layer.
- Weight sparsity:     SILi FP4 + synaptogenesis (starts from 0 synapses via
                       half_bandwidth=0 diagonal init, grows during training).
- Energy/homeostasis:  the top-k threshold acts as the "max" gate; the
                       running_energy EMA tracks how active each layer is
                       so you can adapt k dynamically.

Optional reconstruction
───────────────────────
Pass reconstruct=True to add a decoder layer that predicts the original
chunk from the deepest encoding.  Off by default — the main use case is
expansion for a downstream 2-D sparse transformer, not autoencoding.

Usage
─────
    # Synthetic waveform example (no mic needed):
    python3 sparse_tcnn_audio.py

    # With live microphone (requires sounddevice):
    python3 sparse_tcnn_audio.py --mic --seconds 5

    # Custom chunk size and topology:
    python3 sparse_tcnn_audio.py --chunk 2048 --hidden 128 64 32 --k 32 16 8
"""
from __future__ import annotations

import argparse
import sys
import time
import numpy as np

try:
    import _cpu
except ImportError:
    from sili import _cpu

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _budget(n_in: int, n_out: int, density: float = 0.05) -> int:
    """Byte budget: enough for density% of all possible synapses at 2 bytes each."""
    return max(int(n_in * n_out * density * 2), 64 * 1024)

def _make_layer(n_in: int, n_out: int, cpus: int = 1) -> "_cpu.SparseLinearLayer":
    layer = _cpu.SparseLinearLayer(n_in, n_out, 0, _budget(n_in, n_out), cpus)
    # Pre-energize diagonal with fixed-random non-zero FP4 weights near ±2.0.
    # Values near 2.0 mean the first forward pass produces outputs near 2×input,
    # which gives the Hebbian trace enough signal to update importance and let
    # synaptogenesis start finding useful connections.
    # Fixed seed → reproducible init without requiring the energy function.
    # None of these are zero (guaranteed by the table choice).
    FP4_NONZERO_NEAR_2 = [1.5, -1.5, 2.0, -2.0]
    rng = np.random.default_rng(seed=42)
    for r in range(layer.n_inputs):
        n = layer.buffer.neuron[r]
        for k in range(n.nnz):
            n.synapse[k].weight = float(rng.choice(FP4_NONZERO_NEAR_2))
    return layer


# ─────────────────────────────────────────────────────────────────────────────
# SparseTCNNAudioEncoder
# ─────────────────────────────────────────────────────────────────────────────

class SparseTCNNAudioEncoder:
    """
    Sparse TCNN audio encoder.

    Parameters
    ──────────
    chunk_size   : Audio samples per forward call.
    C_in         : Input channels (1 = mono, 2 = stereo).
    kernels      : Sequence of (kT, stride, hidden_size) per layer.
    top_k        : Top-k active neurons per output position per layer.
    reconstruct  : Add a linear decoder predicting the input from layer-3 output.
    lr           : Learning rate (importance update and weight gradient).
    synap_k      : Top-k probe pairs for synaptogenesis per cycle.
    synap_cutoff : Minimum importance for a new synapse to be accepted.
    synap_max    : Max synapses per input neuron.
    synap_every  : Run synaptogenesis every N forward calls.
    cpus         : OpenMP thread count.
    """

    def __init__(
        self,
        chunk_size  : int = 1024,
        C_in        : int = 1,
        kernels     : list[tuple[int,int,int]] = ((16,4,64),(8,2,128),(4,2,256)),
        top_k       : list[int] = (16, 32, 64),
        reconstruct : bool = False,
        lr          : float = 0.01,
        synap_k     : int = 32,
        synap_cutoff: float = 0.05,
        synap_max   : int = 64,
        synap_every : int = 32,
        cpus        : int = 1,
    ):
        self.chunk_size   = chunk_size
        self.C_in         = C_in
        self.kernels      = list(kernels)
        self.top_k        = list(top_k)
        self.reconstruct  = reconstruct
        self.lr           = lr
        self.synap_k      = synap_k
        self.synap_cutoff = synap_cutoff
        self.synap_max    = synap_max
        self.synap_every  = synap_every
        self.cpus         = cpus
        self._step        = 0

        # Build layers.
        T_cur = chunk_size
        C_cur = C_in
        self.layers: list[_cpu.SparseLinearLayer] = []
        self.T_sizes: list[int] = []   # T_out per layer

        for kT, stride, hidden in kernels:
            n_in  = kT * C_cur
            n_out = hidden
            layer = _make_layer(n_in, n_out, cpus)
            self.layers.append(layer)
            T_out = (T_cur - kT) // stride + 1
            self.T_sizes.append(T_out)
            T_cur = T_out
            C_cur = hidden

        # Optional reconstruction decoder (linear from deepest layer).
        self.decoder: "_cpu.SparseLinearLayer | None" = None
        if reconstruct:
            n_deep = kernels[-1][2]
            T_deep = self.T_sizes[-1]
            n_dec_in  = n_deep         # one position at a time
            n_dec_out = chunk_size     # reconstruct full input
            self.decoder = _make_layer(n_dec_in, n_dec_out, cpus)

        # Per-layer activation cache (for backward pass).
        self._acts: list[np.ndarray | None] = [None] * len(kernels)
        self._x_in: np.ndarray | None = None

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, audio_chunk: np.ndarray, train: bool = True) -> np.ndarray:
        """
        Process one audio chunk.

        Parameters
        ──────────
        audio_chunk : (chunk_size,) or (chunk_size, C_in) float32.
        train       : If True, update importance (Hebbian) and run synaptogenesis.

        Returns
        ───────
        features : 2-D float32 array [max_T × total_H] where:
            rows  = output time positions (zero-padded to max_T)
            cols  = all feature channels across layers (zero-padded)
        Pass to a 2-D sparse transformer as the context sequence.
        """
        x = np.asarray(audio_chunk, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, np.newaxis]                    # (T, 1)
        assert x.shape[0] == self.chunk_size

        lr = self.lr if train else 0.0
        self._x_in = x
        acts = []

        cur_x = x     # (T_cur, C_cur)
        for i, ((kT, stride, hidden), k) in enumerate(
                zip(self.kernels, self.top_k)):
            T_cur, C_cur = cur_x.shape
            # Flatten to (1, T_cur * C_cur) — conv1d reads strides into this.
            x_flat = cur_x.ravel()[np.newaxis, :]   # (1, T_cur*C_cur)
            out = self.layers[i].conv1d_forward(
                x_flat, T=T_cur, C_in=C_cur, kT=kT, stride=stride, lr=lr)
            # out: (1, T_out, hidden) → (T_out, hidden)
            out = out[0]
            # Top-k activation sparsity.
            if k > 0:
                self.layers[i].apply_top_k_sparsity(out, k=k)
            acts.append(out)
            self._acts[i] = out
            cur_x = out   # feed into next layer

        # ── Synaptogenesis ────────────────────────────────────────────────────
        if train:
            self._step += 1
            if self._step % self.synap_every == 0:
                self._synaptogenesis_cycle()

        # ── Concat into 2-D feature array ─────────────────────────────────────
        return self._concat_activations(acts)

    def _concat_activations(
            self, acts: list[np.ndarray]) -> np.ndarray:
        """
        Stack layer activations into a 2-D array [max_T × total_H].

        Each layer produces (T_i, H_i).  We zero-pad T_i to max_T (the
        largest T across layers) and concatenate along the feature axis.
        The result is a single context window ready for a sparse 2-D
        transformer:  rows = time, cols = features.
        """
        max_T   = max(a.shape[0] for a in acts)
        total_H = sum(a.shape[1] for a in acts)
        out = np.zeros((max_T, total_H), dtype=np.float32)
        col = 0
        for a in acts:
            T_i, H_i = a.shape
            out[:T_i, col:col+H_i] = a
            col += H_i
        return out

    # ── Backward (optional) ───────────────────────────────────────────────────

    def backward(
            self,
            target: np.ndarray | None = None,
            dy_ext: np.ndarray | None = None):
        """
        Backward pass through all conv layers.

        Provide either:
          target  : (chunk_size,) reconstruction target — requires reconstruct=True.
          dy_ext  : External gradient for the deepest layer output
                    (e.g., from a downstream transformer), shape (T_deep, H_deep).

        Updates weight and importance in all layers.
        """
        if self._acts[0] is None:
            return

        # Gradient for the deepest layer.
        if dy_ext is not None:
            dy = np.asarray(dy_ext, dtype=np.float32)[np.newaxis, :, :]  # (1,T,H)
        elif target is not None and self.decoder is not None:
            # Reconstruction loss: MSE.
            deep_act = self._acts[-1]   # (T_deep, H_deep)
            recon = self.decoder.forward_disldo(deep_act, lr=0.0)
            t = np.asarray(target, dtype=np.float32).ravel()
            diff = recon[0] - t                                  # (chunk_size,)
            dy_dec = (2.0 / t.size) * diff[np.newaxis, :]       # (1, chunk_size)
            self.decoder.backward_disldo(deep_act, dy_dec, lr=self.lr)
            # Backprop through decoder → gradient for deepest layer.
            dy_deep = np.zeros_like(deep_act)[np.newaxis, :, :]
            # (simplified: use decoder's output gradient as signal)
            dy = dy_deep
        else:
            return

        # Backprop through conv layers (deepest → shallowest).
        cur_dy = dy   # (1, T_i, H_i)
        for i in range(len(self.layers)-1, -1, -1):
            kT, stride, _ = self.kernels[i]
            C_in_i = (self._x_in.shape[1] if i == 0
                      else self.kernels[i-1][2])
            T_in_i = (self.chunk_size if i == 0
                      else self.T_sizes[i-1])
            x_i = (self._x_in if i == 0
                   else self._acts[i-1]).ravel()[np.newaxis, :]
            dx_i = self.layers[i].conv1d_backward(
                x_i, cur_dy,
                T=T_in_i, C_in=C_in_i, kT=kT, stride=stride,
                lr=self.lr)
            # dx_i: (1, T_in_i, C_in_i) → (T_in_i, C_in_i) for next layer back
            cur_dy = dx_i

    # ── Synaptogenesis ────────────────────────────────────────────────────────

    def _synaptogenesis_cycle(self):
        for layer in self.layers:
            layer.build_probes(self.synap_k)
            for _ in range(layer.n_inputs):
                layer.synap_step(self.synap_cutoff, self.synap_max)
            layer.zero_accum()
            layer.equalizer_step()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def info(self) -> str:
        lines = ["SparseTCNNAudioEncoder:"]
        for i, ((kT, stride, hidden), k) in enumerate(
                zip(self.kernels, self.top_k)):
            T_out = self.T_sizes[i]
            nnz   = self.layers[i].nnz
            n_in  = self.layers[i].n_inputs
            lines.append(
                f"  L{i+1}: kT={kT} stride={stride} "
                f"{n_in}→{hidden}  T_out={T_out}  "
                f"nnz={nnz}  top_k={k}")
        total_features = sum(h for _,_,h in self.kernels)
        max_T = max(self.T_sizes)
        lines.append(f"  output: {max_T} × {total_features}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Example: synthetic waveform or microphone
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_audio(chunk_size: int, sample_rate: int = 16000,
                         t_offset: float = 0.0) -> np.ndarray:
    """
    Mixture of sinusoids with amplitude envelope — a rich test signal.
    Returns (chunk_size,) float32.
    """
    t = np.linspace(t_offset, t_offset + chunk_size/sample_rate,
                    chunk_size, endpoint=False)
    x = (0.40 * np.sin(2*np.pi * 220  * t) +   # A3
         0.30 * np.sin(2*np.pi * 440  * t) +   # A4
         0.15 * np.sin(2*np.pi * 880  * t) +   # A5
         0.10 * np.sin(2*np.pi * 1760 * t) +   # A6
         0.05 * np.random.randn(chunk_size))    # noise
    return x.astype(np.float32)


def print_feature_map(feat: np.ndarray, label: str = ""):
    """
    Print a 2-D feature map as a compact ASCII heatmap.
    Rows = time, cols = features.  Non-zero entries shown as symbols.
    """
    T, H = feat.shape
    # Subsample columns if very wide
    max_cols = 80
    step_c = max(1, H // max_cols)
    step_r = max(1, T // 24)
    sampled = feat[::step_r, ::step_c]
    vmax = np.abs(sampled).max() or 1.0
    chars = " ·+#@"
    header = f"{'─'*50}"
    if label:
        header = f"── {label} {'─'*(max(0,46-len(label)))}"
    print(header)
    print(f"  shape={feat.shape}  nnz={np.count_nonzero(feat)}  "
          f"|max|={vmax:.3f}  sample[:{sampled.shape[0]},:{sampled.shape[1]}]:")
    for row in sampled:
        line = ""
        for v in row:
            idx = int(abs(v) / vmax * (len(chars)-1))
            c   = chars[idx] if v >= 0 else chars[idx].lower()
            line += c
        print("  " + line)
    print()


def run_example(args):
    SAMPLE_RATE = 16000
    CHUNK       = args.chunk
    N_CHUNKS    = args.n_chunks

    # Build encoder.
    kernels = [(args.k[i], args.s[i], args.hidden[i])
               for i in range(len(args.hidden))]
    top_k   = args.top_k

    enc = SparseTCNNAudioEncoder(
        chunk_size   = CHUNK,
        C_in         = 1,
        kernels      = kernels,
        top_k        = top_k,
        reconstruct  = args.reconstruct,
        lr           = args.lr,
        synap_k      = 16,
        synap_cutoff = 0.05,
        synap_max    = 32,
        synap_every  = 8,
        cpus         = args.cpus,
    )
    print(enc.info())
    print()

    if args.mic:
        # ── Live microphone input ─────────────────────────────────────────────
        try:
            import sounddevice as sd
        except ImportError:
            print("sounddevice not installed.  Install with: pip install sounddevice")
            sys.exit(1)

        print(f"Recording from mic for {args.seconds}s at {SAMPLE_RATE} Hz ...")
        buf: list[np.ndarray] = []

        def _callback(indata, frames, t, status):
            buf.append(indata[:,0].copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            blocksize=CHUNK, dtype='float32',
                            callback=_callback):
            time.sleep(args.seconds)

        chunks = buf
    else:
        # ── Synthetic waveform ────────────────────────────────────────────────
        print(f"Using synthetic audio ({N_CHUNKS} chunks of {CHUNK} samples).")
        chunks = [make_synthetic_audio(CHUNK, SAMPLE_RATE, i*CHUNK/SAMPLE_RATE)
                  for i in range(N_CHUNKS)]

    # ── Process chunks ────────────────────────────────────────────────────────
    for ci, chunk in enumerate(chunks):
        feat = enc.forward(chunk, train=True)
        print_feature_map(feat, label=f"chunk {ci+1}/{len(chunks)}")

    # ── Final layer stats ─────────────────────────────────────────────────────
    print("After processing:")
    print(enc.info())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Sparse TCNN audio encoder example.")
    p.add_argument("--mic",       action="store_true",
                   help="Use live microphone (requires sounddevice).")
    p.add_argument("--seconds",   type=float, default=3.0,
                   help="Seconds to record from mic (default 3).")
    p.add_argument("--chunk",     type=int, default=512,
                   help="Audio samples per chunk (default 512).")
    p.add_argument("--n-chunks",  type=int, default=6,
                   help="Number of synthetic chunks to process (default 6).")
    p.add_argument("--hidden",    type=int, nargs="+", default=[32, 64, 128],
                   help="Hidden size per layer (default 32 64 128).")
    p.add_argument("--k",         type=int, nargs="+", default=[8, 4, 4],
                   help="Kernel temporal size per layer (default 8 4 4).")
    p.add_argument("--s",         type=int, nargs="+", default=[4, 2, 2],
                   help="Stride per layer (default 4 2 2).")
    p.add_argument("--top-k",     type=int, nargs="+", default=[8, 16, 32],
                   help="Top-k active neurons per layer (default 8 16 32).")
    p.add_argument("--reconstruct", action="store_true",
                   help="Add reconstruction decoder (disabled by default).")
    p.add_argument("--lr",        type=float, default=0.01)
    p.add_argument("--cpus",      type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    np.random.seed(42)
    run_example(_parse())
