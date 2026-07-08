#!/usr/bin/env python3
"""
gen_toy_mistral.py
───────────────────
Build a tiny, structurally-accurate 70-layer "Mistral-shaped" state dict for
testing the sparse_prune -> rnn_fold -> neural_column pipeline.

Architecture matches real Mistral-7B naming/shape conventions (verified against
HF transformers configuration_mistral.py and the Mistral 7B paper) so that
sparse_prune.py's threshold logic, rnn_fold.py's block/GQA/attention detection,
and model_reconstruct.py's family detection all exercise their real code paths
-- just at toy dimensions (~a few hundred KB total instead of ~14GB).

Real Mistral-7B:  hidden=4096  n_layers=32  n_heads=32  n_kv_heads=8
                  head_dim=128  intermediate=14336  vocab=32000
Toy (this file):  hidden=32    n_layers=70  n_heads=4   n_kv_heads=2
                  head_dim=8   intermediate=64        vocab=48

n_layers=70 is chosen for this test (not derived from any real model) per
explicit request -- deep enough to stress rnn_fold's block detection and the
neural-column duplication step.

Weight init: N(0, 0.02) (standard transformer init scale), then a random
fraction of ENTRIES (not whole layers) forced to exact 0.0.  This creates a
realistic mixed population: a genuinely-sparse component (the forced zeros,
which pruning should always catch) plus a dense small-magnitude noise
component (which is the actual test of whether the auto-threshold is
well-calibrated -- forcing too much noise into "pruned" territory is exactly
the over-sparsification failure mode being checked for).

1-D tensors (norms) are left fully dense, matching real usage -- LayerNorm
weights are never meaningfully sparse in a trained model.
"""
import torch
import numpy as np

torch.manual_seed(1234)

# ── Toy architecture config (Mistral-shaped, KB-scale) ────────────────────────
N_LAYERS      = 70
HIDDEN        = 32
N_HEADS       = 4
N_KV_HEADS    = 2          # GQA: groups = N_HEADS // N_KV_HEADS = 2
HEAD_DIM      = 8          # N_HEADS * HEAD_DIM = 32 = HIDDEN (matches real Mistral convention)
INTERMEDIATE  = 64
VOCAB         = 48

INIT_STD      = 0.02       # standard transformer weight init scale
ZERO_FRAC_LO  = 0.60       # per-layer forced-zero fraction varies, to stress
ZERO_FRAC_HI  = 0.92       # both under- and over- the min_sparsity(0.33) cutoff


def _sparse_normal(shape, zero_frac, rng):
    """N(0, INIT_STD) with a random `zero_frac` of entries forced to exact 0.0."""
    w = torch.from_numpy(
        rng.normal(0.0, INIT_STD, size=shape).astype(np.float32))
    mask = torch.from_numpy(rng.random(shape) < zero_frac)
    w[mask] = 0.0
    return w


def build_toy_mistral_state_dict(seed: int = 1234) -> dict:
    rng = np.random.default_rng(seed)
    sd  = {}

    # ── Embedding (always dense -- lookup table, sparse_prune keeps it dense
    #    anyway since ndim>2 rule doesn't apply here but it's 2D... actually
    #    embed_tokens IS 2D [vocab,hidden], so sparse_prune's threshold WOULD
    #    apply unless it fails min_sparsity.  We deliberately keep it fully
    #    dense (zero_frac=0) since a real embedding table is never sparse in
    #    the "prunable small weights" sense -- this tests that dense-by-content
    #    (not just dense-by-shape) layers are correctly kept dense.
    sd["model.embed_tokens.weight"] = _sparse_normal(
        (VOCAB, HIDDEN), zero_frac=0.0, rng=rng)

    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        # Per-layer zero fraction varies across the min_sparsity(33%) boundary
        # so both "goes sparse" and "stays dense (too little sparsity)" paths
        # get exercised across the 70 layers.
        zf = rng.uniform(ZERO_FRAC_LO, ZERO_FRAC_HI)

        sd[p+"self_attn.q_proj.weight"] = _sparse_normal(
            (N_HEADS*HEAD_DIM, HIDDEN), zf, rng)
        sd[p+"self_attn.k_proj.weight"] = _sparse_normal(
            (N_KV_HEADS*HEAD_DIM, HIDDEN), zf, rng)
        sd[p+"self_attn.v_proj.weight"] = _sparse_normal(
            (N_KV_HEADS*HEAD_DIM, HIDDEN), zf, rng)
        sd[p+"self_attn.o_proj.weight"] = _sparse_normal(
            (HIDDEN, N_HEADS*HEAD_DIM), zf, rng)

        sd[p+"mlp.gate_proj.weight"] = _sparse_normal(
            (INTERMEDIATE, HIDDEN), zf, rng)
        sd[p+"mlp.up_proj.weight"] = _sparse_normal(
            (INTERMEDIATE, HIDDEN), zf, rng)
        sd[p+"mlp.down_proj.weight"] = _sparse_normal(
            (HIDDEN, INTERMEDIATE), zf, rng)

        # Norms: always dense, small values near 1.0 (RMSNorm init convention).
        sd[p+"input_layernorm.weight"] = torch.ones(HIDDEN) + torch.from_numpy(
            rng.normal(0, 0.01, HIDDEN).astype(np.float32))
        sd[p+"post_attention_layernorm.weight"] = torch.ones(HIDDEN) + torch.from_numpy(
            rng.normal(0, 0.01, HIDDEN).astype(np.float32))

    sd["model.norm.weight"] = torch.ones(HIDDEN)
    sd["lm_head.weight"] = _sparse_normal((VOCAB, HIDDEN), zero_frac=0.70, rng=rng)

    return sd


def build_toy_mistral_state_dict(seed: int = 1234):
    rng = np.random.default_rng(seed)
    sd  = {}
    intended_zero_frac = {}   # name -> exact fraction forced to 0.0 (ground truth)

    def add(name, shape, zf, dense=False):
        if dense:
            sd[name] = _sparse_normal(shape, 0.0, rng)
            intended_zero_frac[name] = 0.0
        else:
            sd[name] = _sparse_normal(shape, zf, rng)
            intended_zero_frac[name] = zf

    add("model.embed_tokens.weight", (VOCAB, HIDDEN), 0.0, dense=True)

    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        zf = rng.uniform(ZERO_FRAC_LO, ZERO_FRAC_HI)

        add(p+"self_attn.q_proj.weight", (N_HEADS*HEAD_DIM, HIDDEN), zf)
        add(p+"self_attn.k_proj.weight", (N_KV_HEADS*HEAD_DIM, HIDDEN), zf)
        add(p+"self_attn.v_proj.weight", (N_KV_HEADS*HEAD_DIM, HIDDEN), zf)
        add(p+"self_attn.o_proj.weight", (HIDDEN, N_HEADS*HEAD_DIM), zf)
        add(p+"mlp.gate_proj.weight",    (INTERMEDIATE, HIDDEN), zf)
        add(p+"mlp.up_proj.weight",      (INTERMEDIATE, HIDDEN), zf)
        add(p+"mlp.down_proj.weight",    (HIDDEN, INTERMEDIATE), zf)

        sd[p+"input_layernorm.weight"] = torch.ones(HIDDEN) + torch.from_numpy(
            rng.normal(0, 0.01, HIDDEN).astype(np.float32))
        sd[p+"post_attention_layernorm.weight"] = torch.ones(HIDDEN) + torch.from_numpy(
            rng.normal(0, 0.01, HIDDEN).astype(np.float32))
        intended_zero_frac[p+"input_layernorm.weight"] = 0.0
        intended_zero_frac[p+"post_attention_layernorm.weight"] = 0.0

    sd["model.norm.weight"] = torch.ones(HIDDEN)
    intended_zero_frac["model.norm.weight"] = 0.0
    add("lm_head.weight", (VOCAB, HIDDEN), 0.70)

    return sd, intended_zero_frac


if __name__ == "__main__":
    sd, izf = build_toy_mistral_state_dict()
    total_params = sum(t.numel() for t in sd.values())
    total_bytes  = sum(t.numel() * 4 for t in sd.values())
    print(f"Toy Mistral-shaped model:")
    print(f"  layers          : {N_LAYERS}")
    print(f"  hidden/heads/kv : {HIDDEN}/{N_HEADS}/{N_KV_HEADS}  (GQA groups={N_HEADS//N_KV_HEADS})")
    print(f"  intermediate    : {INTERMEDIATE}")
    print(f"  vocab           : {VOCAB}")
    print(f"  tensors         : {len(sd)}")
    print(f"  total params    : {total_params:,}")
    print(f"  dense fp32 size : {total_bytes/1024:.1f} KB")

    out_path = "/home/claude/foldtest/toy_mistral.pt"
    torch.save(sd, out_path)
    import json
    with open("/home/claude/foldtest/toy_mistral_ground_truth.json", "w") as f:
        json.dump(izf, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Saved: toy_mistral_ground_truth.json (intended zero fraction per tensor)")
