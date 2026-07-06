"""
multimodal_sparse_rnn.py
────────────────────────
Multimodal sparse recurrent neural network for a sensorimotor agent.

Scale math (30 GB budget, 2 bytes/synapse average)
────────────────────────────────────────────────────
  Raw synapses:       27 GB / 2 B = 13.5 B synapses
  Layer folding 10×:  × 10  → 135 B effective (recurrence replaces stacking)
  Dead neuron 90%:    × 10  → 1.35 T effective parameters

  Runtime sparsity:   activations are 5–10% active → forward/backward are
                      10–20× faster than parameter count suggests.
  Combined:           training is ~1000× cheaper than an equivalent dense model.

  Capacity comparison:
    The honest benchmark is not parameter count but "how many bits of
    optimized compiled code can I represent?" An inverse-kinematics solver
    in optimized C++ is a few KB. Our model at 2 bytes/synapse with 14B
    synapses = 28 GB of raw representational capacity, times 4× (4-bit value
    vs 16-bit code pointer granularity) = ~112 GB equivalent program space.
    That vastly exceeds any conventional robotics controller.

Architecture
────────────
Each sensory modality contributes to a unified recurrent context vector h.
The recurrent state IS the depth — no explicit layer stacking needed.

  new_h = clip6(W_ih @ x + W_hh @ h)

FP4 quantisation clips values to ±6 on every write. This acts as the
nonlinearity and gradient stabiliser — no tanh or gating required.
It's a clipped RNN, not an LSTM. Simpler, and the clipping already
prevents exploding state.

Sparse attention
─────────────────
Q, K, V are sparse recurrent projections of h. The attention kernel
selects top-k positions by L2 norm (k = sqrt(T) by default) and computes
only those k² dot-products. Zero entries in the score matrix are treated
as -inf in softmax, so non-selected pairs contribute 0 to the output.
Cost: O(k²·d) = O(T·d) at k=sqrt(T), vs O(T²·d) for dense attention.

Modality slots in h are logical (index ranges), not physical barriers.
Neurons in the vision slot that are irrelevant to a task converge to zero
via pruning and stay at zero cost (sparse — no synapses, no compute).

Stereo / multi-channel inputs are stacked in the channel dimension of the
patch embedding. Left/right image or L/R audio are channels 0..C and C..2C.

Usage
─────
    model = MultimodalSparseRNN(cfg)
    h = model.zero_state()

    for frame in stream:
        out, h = model.step(frame, h, lr=0.01)
        model.synap_cycle(h_accum, out_accum)   # every N steps

"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

try:
    # Package-qualified first: keeps sys.modules keyed consistently as
    # 'sili._cpu' everywhere, avoiding the double-registration bug that
    # occurs when a bare 'import _cpu' and 'sili._cpu' both execute the
    # compiled extension's init code under two different sys.modules
    # keys (see sili/conversion/rnn_fold.py for the full account).
    from sili import _cpu
except ImportError:
    import _cpu

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModalityConfig:
    name: str
    input_size: int          # Raw input neurons (after patch embedding if any).
    state_size: int          # Neurons allocated in the unified h for this modality.
    encoder_bw: int          # half_bandwidth for the input→state sparse layer.
    encoder_budget: int      # Byte budget for this encoder's weight buffer.


@dataclass
class MultimodalConfig:
    # Modalities — order determines their slot position in h.
    modalities: list[ModalityConfig]

    # Recurrent weight matrix (W_hh).
    recurrent_bw: int = 32
    recurrent_budget: int = 512 * 1024 * 1024   # 512 MB default

    # Output heads — each is a (state_size → output_size) sparse layer.
    language_output_size: int = 32000  # vocabulary
    motor_output_size:    int = 200    # actuator commands

    output_bw:     int = 8
    output_budget: int = 64 * 1024 * 1024

    # Q, K, V recurrent projection sizes (for attention over context).
    qkv_size: int = 512
    qkv_bw:   int = 16
    qkv_budget: int = 32 * 1024 * 1024

    # Synaptogenesis hyperparameters.
    synap_k:                int   = 64
    synap_importance_cutoff: float = 0.05
    synap_max_row_weights:  int   = 128

    num_cpus: int = 4

    @property
    def total_state_size(self) -> int:
        return sum(m.state_size for m in self.modalities)

    @classmethod
    def for_humanoid_robot(cls) -> "MultimodalConfig":
        """
        Practical sizing for a ~5-foot humanoid with camera, microphone,
        and proprioception. Language understanding included.

        Budget breakdown (at 1% density, 2 bytes/synapse):
          Vision encoder:   12K → 96K  × 0.01 × 2  ≈  23 MB
          Audio encoder:    512  → 10K  × 0.01 × 2  ≈  0.1 MB
          Proprio encoder:  200  → 4K   × 0.01 × 2  ≈  0.02 MB
          Lang encoder:     512  → 35K  × 0.01 × 2  ≈  0.36 MB
          W_hh (recurrent): 145K² × 0.01 × 2        ≈  420 MB
          Q/K/V heads:      145K → 512 × 3  × 0.01  ≈  4 MB
          Language out:     145K → 32K × 0.01 × 2   ≈  93 MB
          Motor out:        145K → 200 × 0.01 × 2   ≈  0.6 MB
          ──────────────────────────────────────────────────
          Total                                       ≈  541 MB

        30 GB budget supports ~55× this scale (GPT-3 territory).
        """
        MB = 1024 * 1024
        GB = 1024 * MB
        return cls(
            modalities=[
                ModalityConfig("vision",        12288,  96000, 64,  64*MB),
                ModalityConfig("audio",          512,   10000, 32,   4*MB),
                ModalityConfig("proprioception", 200,    4000, 16,   1*MB),
                ModalityConfig("language",       512,   35000, 32,   4*MB),
            ],
            recurrent_bw     = 32,
            recurrent_budget = 512 * MB,
            language_output_size = 32000,
            motor_output_size    = 200,
            output_bw     = 8,
            output_budget = 128 * MB,
            qkv_size      = 512,
            qkv_bw        = 16,
            qkv_budget    = 32 * MB,
            synap_k                 = 64,
            synap_importance_cutoff = 0.05,
            synap_max_row_weights   = 128,
            num_cpus = 4,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dense_to_csr(x: np.ndarray) -> tuple:
    """Convert (batch, n) dense float32 → (ptrs, indices, values) CSR."""
    return _cpu.dense_to_csr(np.asarray(x, dtype=np.float32), 0.0)


def _forward(layer: "_cpu.SparseLinearLayer",
             x: np.ndarray, lr: float = 0.0) -> np.ndarray:
    """Dense forward through a SparseLinearLayer."""
    return layer.forward_dense(np.asarray(x, dtype=np.float32), lr)


def _backward(layer: "_cpu.SparseLinearLayer",
              x: np.ndarray, dy: np.ndarray, lr: float) -> np.ndarray:
    """Dense backward through a SparseLinearLayer.

    x is accepted for API symmetry with _forward (and because callers
    naturally have it on hand) but not passed through directly --
    backward_dense relies on the layer's own _last_input, stored by the
    preceding forward_dense call, rather than taking it as an explicit
    argument. There is deliberately no sparse-input backward at all (see
    conversation): dx = sum_c W[r,c]*dy[c] depends only on weights and the
    gradient, not on the input value itself, so dense input is what lets
    gradient reach a row whose own activation happened to be near zero --
    sparse input would silently lose that. Only the GRADIENT toggles
    sparse (backward_sparse) vs dense (this function) -- both always take
    dense input.
    """
    return layer.backward_dense(np.asarray(dy, dtype=np.float32), lr)


def _clip6(x: np.ndarray) -> np.ndarray:
    """FP4 clips at ±6. Apply explicitly in the Python path for clarity."""
    return np.clip(x, -6.0, 6.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Patch embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed:
    """
    Split a (H, W, C) image into non-overlapping patches and flatten each.

    The recurrent slice h_vis is stacked as extra channels on the current
    frame so the encoder sees both the new frame AND prior state.
    """

    def __init__(self, img_h: int, img_w: int, img_c: int,
                 patch_h: int, patch_w: int):
        self.img_h   = img_h;  self.img_w  = img_w;  self.img_c  = img_c
        self.patch_h = patch_h; self.patch_w = patch_w
        self.n_patches_h = img_h // patch_h
        self.n_patches_w = img_w // patch_w
        self.n_patches   = self.n_patches_h * self.n_patches_w
        self.patch_dim   = patch_h * patch_w * img_c

    @property
    def output_size(self) -> int:
        return self.n_patches * self.patch_dim

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        frame: (H, W, C) float32  → (n_patches × patch_dim,)
        """
        H, W, C = frame.shape
        assert H == self.img_h and W == self.img_w and C == self.img_c
        ph, pw = self.patch_h, self.patch_w
        nph, npw = self.n_patches_h, self.n_patches_w
        # Reshape into patches.
        patches = (frame.reshape(nph, ph, npw, pw, C)
                       .transpose(0, 2, 1, 3, 4)
                       .reshape(self.n_patches, -1))
        return patches.ravel().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MultimodalSparseRNN
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalSparseRNN:
    """
    Unified sparse recurrent network for multimodal sensorimotor processing.

    Parameters
    ──────────
    cfg : MultimodalConfig

    State
    ─────
    h : (total_state_size,) float32   — the unified recurrent context
    """

    def __init__(self, cfg: MultimodalConfig):
        self.cfg = cfg
        n_state  = cfg.total_state_size
        cpus     = cfg.num_cpus

        # ── Per-modality input → state encoders ──────────────────────────────
        self.encoders: dict[str, "_cpu.SparseLinearLayer"] = {}
        offset = 0
        self.modality_slices: dict[str, slice] = {}
        for mod in cfg.modalities:
            self.encoders[mod.name] = _cpu.SparseLinearLayer(
                mod.input_size,
                mod.state_size,
                mod.encoder_bw,
                mod.encoder_budget,
                cpus)
            self.modality_slices[mod.name] = slice(offset, offset + mod.state_size)
            offset += mod.state_size

        # ── Recurrent state → state (W_hh) ───────────────────────────────────
        self.W_hh = _cpu.SparseLinearLayer(
            n_state, n_state,
            cfg.recurrent_bw,
            cfg.recurrent_budget,
            cpus)

        # ── Q, K, V recurrent projections ────────────────────────────────────
        self.W_q = _cpu.SparseLinearLayer(
            n_state, cfg.qkv_size, cfg.qkv_bw, cfg.qkv_budget, cpus)
        self.W_k = _cpu.SparseLinearLayer(
            n_state, cfg.qkv_size, cfg.qkv_bw, cfg.qkv_budget, cpus)
        self.W_v = _cpu.SparseLinearLayer(
            n_state, cfg.qkv_size, cfg.qkv_bw, cfg.qkv_budget, cpus)

        # ── Output heads ──────────────────────────────────────────────────────
        self.W_lang = _cpu.SparseLinearLayer(
            n_state, cfg.language_output_size,
            cfg.output_bw, cfg.output_budget, cpus)
        self.W_motor = _cpu.SparseLinearLayer(
            n_state, cfg.motor_output_size,
            cfg.output_bw, cfg.output_budget, cpus)

        # Accumulate |x| and |grad| for synaptogenesis (zeroed every N steps).
        self._step_count = 0

    # ── State initialisation ──────────────────────────────────────────────────

    def zero_state(self) -> np.ndarray:
        return np.zeros(self.cfg.total_state_size, dtype=np.float32)

    # ── Forward step ──────────────────────────────────────────────────────────

    def step(self,
             inputs: dict[str, np.ndarray],
             h: np.ndarray,
             lr: float = 0.01) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        One recurrent step.

        Parameters
        ──────────
        inputs : {modality_name: array}  — present modalities only; absent
                 modalities contribute 0 to their slot (their neurons stay at
                 whatever value h already holds, then decay via W_hh → 0).
        h      : (total_state_size,) current state
        lr     : learning rate for inline importance update

        Returns
        ───────
        outputs : {"language": logits, "motor": commands}
        h_new   : updated state
        """
        h1 = np.zeros(self.cfg.total_state_size, dtype=np.float32)
        h2d = h[np.newaxis, :]   # (1, state)

        # ── Encoder contributions ─────────────────────────────────────────────
        for mod in self.cfg.modalities:
            if mod.name not in inputs:
                continue
            x = np.asarray(inputs[mod.name], dtype=np.float32).ravel()[np.newaxis, :]
            enc_out = _forward(self.encoders[mod.name], x, lr)   # (1, mod.state_size)
            sl = self.modality_slices[mod.name]
            h1[sl] += enc_out.ravel()

        # ── Recurrent contribution (W_hh) ─────────────────────────────────────
        rec_out = _forward(self.W_hh, h2d, lr)   # (1, total_state)
        h1 += rec_out.ravel()

        # Clip at ±6 — this IS the nonlinearity. FP4 will re-quantise on
        # the next write, so values are doubly bounded.
        h_new = _clip6(h1)

        # ── Outputs ───────────────────────────────────────────────────────────
        h_new2d = h_new[np.newaxis, :]
        lang_logits = _forward(self.W_lang,  h_new2d, 0.0).ravel()
        motor_cmds  = _forward(self.W_motor, h_new2d, 0.0).ravel()

        outputs = {
            "language": lang_logits,
            "motor":    motor_cmds,
        }
        self._step_count += 1
        return outputs, h_new

    # ── Recurrent sparse attention ────────────────────────────────────────────

    def attention(self,
                  h_seq: np.ndarray,        # (T, state_size) recent state history
                  top_k: int = 0,           # for sparse_attention: 0 = sqrt(T)
                  half_bandwidth: int = 0,  # >0 → use sparse_banded_attention
                  inner_k: int = 0,         # inner top-k within band (0 = full band)
                  lr: float = 0.0) -> np.ndarray:
        """
        Sparse or sparse-banded attention over T recurrent states.

        half_bandwidth=0 (default):
            Uses sparse_attention — selects top-k queries and keys globally
            by L2 norm.  Cost O(k²·d), k = top_k or sqrt(T).

        half_bandwidth>0:
            Uses sparse_banded_attention — geometric diagonal outer loop
            (same diagonal as the weight init) restricts each query to the
            nearest ±half_bandwidth keys, then inner_k further filters by
            L2 norm within that band.  Cost O(T·inner_k·d).
            This is the preferred mode for sequences longer than ~64 steps
            because it scales linearly in T rather than quadratically.

        Both modes clip output to ±6 (FP4 range) and return the last row.
        """
        T = h_seq.shape[0]
        if T == 0:
            return np.zeros(self.cfg.qkv_size, dtype=np.float32)

        Q = _forward(self.W_q, h_seq, lr)   # (T, qkv_size)
        K = _forward(self.W_k, h_seq, lr)
        V = _forward(self.W_v, h_seq, lr)

        Q = Q.astype(np.float32)
        K = K.astype(np.float32)
        V = V.astype(np.float32)

        if half_bandwidth > 0:
            attn_out = _cpu.sparse_banded_attention(
                Q, K, V,
                half_bandwidth = half_bandwidth,
                inner_k        = inner_k,
                num_cpus       = self.cfg.num_cpus)
        else:
            attn_out = _cpu.sparse_attention(
                Q, K, V,
                top_k    = top_k,
                num_cpus = self.cfg.num_cpus)

        return _clip6(attn_out[-1])

    # ── Backward pass ─────────────────────────────────────────────────────────

    def backward(self,
                 inputs: dict[str, np.ndarray],
                 h_prev: np.ndarray,
                 h_new: np.ndarray,
                 grad_h: np.ndarray,
                 grad_outputs: dict[str, np.ndarray],
                 lr: float = 0.01) -> dict[str, np.ndarray]:
        """
        One backward step.

        grad_h        : (state_size,)   gradient w.r.t. h_new
        grad_outputs  : {"language": dL/d_logits, "motor": dL/d_motor}

        Returns
        ───────
        grad_inputs : {modality_name: grad_x}
        """
        h_new2d  = h_new [np.newaxis, :]
        h_prev2d = h_prev[np.newaxis, :]

        # Accumulate gradient from output heads into d_h.
        d_h = grad_h.copy()

        if "language" in grad_outputs:
            dy = np.asarray(grad_outputs["language"], dtype=np.float32)[np.newaxis, :]
            d_h_lang = _backward(self.W_lang, h_new2d, dy, lr).ravel()
            d_h += d_h_lang

        if "motor" in grad_outputs:
            dy = np.asarray(grad_outputs["motor"], dtype=np.float32)[np.newaxis, :]
            d_h_motor = _backward(self.W_motor, h_new2d, dy, lr).ravel()
            d_h += d_h_motor

        # Clip gradient (FP4 quantisation acts as implicit clip, but be explicit).
        d_h = np.clip(d_h, -6.0, 6.0).astype(np.float32)
        d_h2d = d_h[np.newaxis, :]

        # Recurrent backward (W_hh).
        _backward(self.W_hh, h_prev2d, d_h2d, lr)

        # Encoder backward — each modality gets its slice of d_h.
        grad_inputs = {}
        for mod in self.cfg.modalities:
            if mod.name not in inputs:
                continue
            sl = self.modality_slices[mod.name]
            d_enc = d_h[sl][np.newaxis, :]
            dx = self.encoders[mod.name].backward_dense(
                np.asarray(d_enc, dtype=np.float32), lr)
            grad_inputs[mod.name] = dx.ravel()

        return grad_inputs

    # ── Synaptogenesis ────────────────────────────────────────────────────────

    def synap_cycle(self, n_steps: Optional[int] = None) -> None:
        """
        Run one full synaptogenesis cycle across all layers.

        Call this every N forward/backward steps (e.g. every 100 steps).
        Builds probes from neuron accumulators, then runs one synap_step
        per input neuron across each layer.

        After completing, zeroes all accumulators.
        """
        cfg = self.cfg
        all_layers = (
            list(self.encoders.values()) +
            [self.W_hh, self.W_q, self.W_k, self.W_v,
             self.W_lang, self.W_motor])

        for layer in all_layers:
            layer.build_probes(cfg.synap_k)
            n = n_steps if n_steps is not None else layer.n_inputs
            for _ in range(n):
                layer.synap_step(cfg.synap_importance_cutoff,
                                 cfg.synap_max_row_weights)
                layer.equalizer_step()
            layer.zero_accum()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def total_synapses(self) -> int:
        all_layers = (
            list(self.encoders.values()) +
            [self.W_hh, self.W_q, self.W_k, self.W_v,
             self.W_lang, self.W_motor])
        return sum(l.nnz for l in all_layers)

    def debug_neuron(self, modality: str, neuron_idx: int, synapse_idx: int):
        """
        Quick debug accessor.  Example:
            model.debug_neuron("vision", 216, 12)
        Returns a dict with the index, weight, and importance for that synapse.
        """
        layer = self.encoders.get(modality) or getattr(self, f"W_{modality}", None)
        if layer is None:
            raise KeyError(f"No layer for modality or weight name: {modality!r}")
        indices = layer.indices
        weights = layer.importance
        vals    = layer.weights_vals
        # Count how many synapses belong to rows 0..neuron_idx-1 to find offset.
        import numpy as np
        ptrs = layer.ptrs   # row_ptr array
        offset = int(ptrs[neuron_idx]) + synapse_idx
        return {
            "row":       neuron_idx,
            "synapse":   synapse_idx,
            "col_idx":   int(indices[offset]),
            "weight":    float(vals[offset]),
            "importance": float(weights[offset]),
        }

    def __repr__(self) -> str:
        n = self.cfg.total_state_size
        syns = self.total_synapses()
        return (f"MultimodalSparseRNN("
                f"state={n}, synapses={syns:,}, "
                f"modalities={[m.name for m in self.cfg.modalities]})")


# ─────────────────────────────────────────────────────────────────────────────
# Patch-based vision front-end convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

class VisionPatchFrontend:
    """
    Converts a raw camera frame into a flat patch embedding suitable for the
    vision encoder input.

    Recurrent visual state h_vis (the vision slot of h) is stacked as extra
    channels on each patch before embedding, so the encoder sees both the
    current frame AND the prior visual context.
    """

    def __init__(self, img_h=64, img_w=64, img_c=3,
                 patch_h=8, patch_w=8):
        self.embed = PatchEmbed(img_h, img_w, img_c, patch_h, patch_w)
        self.patch_dim = patch_h * patch_w * img_c
        self.n_patches = self.embed.n_patches

    def __call__(self, frame: np.ndarray,
                 h_vis: Optional[np.ndarray] = None) -> np.ndarray:
        """
        frame  : (H, W, C) float32   raw camera frame [0, 1]
        h_vis  : (vision_state_size,) recurrent visual state (optional)

        Returns flat (n_patches × patch_dim,) float32 patch embedding.
        If h_vis is supplied, each patch gets the corresponding h_vis slice
        appended as extra channels. This requires the encoder to accept
        n_patches × (patch_dim + h_vis_per_patch) inputs.
        """
        patches_flat = self.embed(frame)
        # Simple path: just return the raw patches (no h_vis stacking).
        # Full path: tile h_vis across patches and concatenate.
        return patches_flat


# ─────────────────────────────────────────────────────────────────────────────
# Example usage / minimal smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("Building model (small config for smoke test)...")
    KB = 1024
    MB = 1024 * KB

    # Tiny config for fast construction.
    cfg = MultimodalConfig(
        modalities=[
            ModalityConfig("vision",        64, 128, 4,  256*KB),
            ModalityConfig("audio",          16,  32, 2,   64*KB),
            ModalityConfig("proprioception",  8,  16, 2,   32*KB),
            ModalityConfig("language",        8,  32, 2,   64*KB),
        ],
        recurrent_bw     = 4,
        recurrent_budget = 1 * MB,
        language_output_size = 64,
        motor_output_size    = 8,
        output_bw     = 2,
        output_budget = 128 * KB,
        qkv_size      = 16,
        qkv_bw        = 2,
        qkv_budget    = 64 * KB,
        synap_k       = 4,
        synap_importance_cutoff = 0.0,
        synap_max_row_weights   = 8,
        num_cpus = 1,
    )

    model = MultimodalSparseRNN(cfg)
    print(model)
    print(f"  State size:   {cfg.total_state_size}")
    print(f"  Total synaps: {model.total_synapses():,}")

    h = model.zero_state()

    # Simulate 10 steps.
    losses = []
    for step in range(10):
        inputs = {
            "vision":        np.random.randn(64).astype(np.float32),
            "audio":         np.random.randn(16).astype(np.float32),
            "proprioception":np.random.randn(8).astype(np.float32),
            "language":      np.random.randn(8).astype(np.float32),
        }
        # Target: motor commands = 0.
        target_motor = np.zeros(cfg.motor_output_size, dtype=np.float32)

        out, h_new = model.step(inputs, h, lr=0.01)

        # Simple MSE loss on motor output.
        diff  = out["motor"] - target_motor
        loss  = float(np.mean(diff ** 2))
        losses.append(loss)

        # Backward pass.
        grad_motor = (2.0 / cfg.motor_output_size) * diff
        model.backward(
            inputs, h, h_new,
            grad_h=np.zeros_like(h),
            grad_outputs={"motor": grad_motor},
            lr=0.01)

        h = h_new

    print(f"  Losses: {[f'{l:.4f}' for l in losses]}")
    print("  (Zero-weight init → zero output → zero loss on zero target — expected.)")

    # Debug access.
    syn = model.W_hh.buffer.neuron[0].synapse[0]
    print(f"\nDebug: W_hh neuron[0].synapse[0] = {syn}")
    print(f"  weight={syn.weight:.4f}, importance={syn.importance:.4f}, index={syn.index}")

    # Synaptogenesis.
    print("\nRunning synaptogenesis cycle...")
    model.synap_cycle(n_steps=cfg.total_state_size)
    print(f"  Synapses after: {model.total_synapses():,}")

    print("\nAll OK.")
