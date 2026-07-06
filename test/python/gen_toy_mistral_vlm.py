#!/usr/bin/env python3
"""
gen_toy_mistral_vlm.py
──────────────────────
Tiny, structurally-accurate Mistral-Small-3.1 (2503) VLM-shaped state dict:
Pixtral vision tower + Mistral text + multimodal projector, matching the
HF Mistral3ForConditionalGeneration tensor schema.

Supersedes gen_toy_mistral.py for VLM work: vision is a hard requirement for
the target use case (FreeCAD/KiCad agents), and 2503 is the vision base model
(the 2501 base is TEXT-ONLY -- an easy mistag to hit when searching).

Reproduces the two structural quirks the old toy missed:
  1. TWO repeated block families (language + vision) -- block detection must
     find both groups, and per-suffix folding must handle both prefixes.
  2. Text heads*head_dim != hidden (real: 32*128=4096 vs hidden 5120).
     Toy: 4*16=64 vs hidden 32 -> q_proj (64,32), k/v (32,32), o_proj (32,64).
     This exercises the non-square fold path.

Also includes the 4-D patch_conv.weight (conv patch embed) -- CSR is 2-D only,
so pruning must reshape (out, -1) and record the original shape.

Prefix vintages (transformers refactored VLM key layout ~v4.52):
  default        : model.language_model.layers.* / model.vision_tower.* / lm_head.weight
  --legacy-prefix: language_model.model.layers.* / vision_tower.* /
                   language_model.lm_head.weight
See docs/requirements_vlm_streaming_rtac.md section 1.2. Names are
high-confidence from the transformers mistral3/pixtral module structure but
MUST be diffed against the real 2503 model.safetensors.index.json when a
download exists (expected_names() below is built for exactly that diff).
"""
import argparse
import torch
import numpy as np

# ── Toy dims (see requirements doc section 1.1 for real/toy table) ───────────
TXT_HIDDEN = 32; TXT_LAYERS = 12; TXT_HEADS = 4; TXT_KV = 2; TXT_HDIM = 16
TXT_INTER  = 64; VOCAB = 64
VIS_HIDDEN = 16; VIS_LAYERS = 6;  VIS_HEADS = 2; VIS_HDIM = 8
VIS_INTER  = 32; PATCH = 4; IN_CH = 3
MERGE = 2
ZERO_FRAC_LO, ZERO_FRAC_HI = 0.55, 0.85


def _sparse_normal(shape, zero_frac, rng):
    t = torch.from_numpy(rng.normal(0, 0.02, shape).astype(np.float32))
    if zero_frac > 0:
        t = t * torch.from_numpy((rng.random(shape) >= zero_frac))
    return t.float()


def _prefixes(legacy: bool):
    if legacy:
        return dict(lang="language_model.model.layers.",
                    lang_embed="language_model.model.embed_tokens.weight",
                    lang_norm="language_model.model.norm.weight",
                    lm_head="language_model.lm_head.weight",
                    vis="vision_tower.transformer.layers.",
                    vis_conv="vision_tower.patch_conv.weight",
                    vis_ln="vision_tower.ln_pre.weight",
                    proj="multi_modal_projector.")
    return dict(lang="model.language_model.layers.",
                lang_embed="model.language_model.embed_tokens.weight",
                lang_norm="model.language_model.norm.weight",
                lm_head="lm_head.weight",
                vis="model.vision_tower.transformer.layers.",
                vis_conv="model.vision_tower.patch_conv.weight",
                vis_ln="model.vision_tower.ln_pre.weight",
                proj="model.multi_modal_projector.")


def build_toy_mistral_vlm_state_dict(seed: int = 1234, legacy_prefix: bool = False):
    rng = np.random.default_rng(seed)
    P = _prefixes(legacy_prefix)
    sd, izf = {}, {}

    def add(name, shape, zf, dense=False):
        sd[name] = _sparse_normal(shape, 0.0 if dense else zf, rng)
        izf[name] = 0.0 if dense else zf

    def add_norm(name, dim):
        sd[name] = torch.ones(dim) + torch.from_numpy(
            rng.normal(0, 0.01, dim).astype(np.float32))
        izf[name] = 0.0

    # ── language family (heads*head_dim != hidden quirk) ─────────────────────
    add(P["lang_embed"], (VOCAB, TXT_HIDDEN), 0.0, dense=True)
    q_out = TXT_HEADS * TXT_HDIM          # 64 != 32
    kv_out = TXT_KV * TXT_HDIM            # 32
    for i in range(TXT_LAYERS):
        p = f"{P['lang']}{i}."
        zf = rng.uniform(ZERO_FRAC_LO, ZERO_FRAC_HI)
        add(p+"self_attn.q_proj.weight", (q_out, TXT_HIDDEN), zf)
        add(p+"self_attn.k_proj.weight", (kv_out, TXT_HIDDEN), zf)
        add(p+"self_attn.v_proj.weight", (kv_out, TXT_HIDDEN), zf)
        add(p+"self_attn.o_proj.weight", (TXT_HIDDEN, q_out), zf)
        add(p+"mlp.gate_proj.weight",    (TXT_INTER, TXT_HIDDEN), zf)
        add(p+"mlp.up_proj.weight",      (TXT_INTER, TXT_HIDDEN), zf)
        add(p+"mlp.down_proj.weight",    (TXT_HIDDEN, TXT_INTER), zf)
        add_norm(p+"input_layernorm.weight", TXT_HIDDEN)
        add_norm(p+"post_attention_layernorm.weight", TXT_HIDDEN)
    add_norm(P["lang_norm"], TXT_HIDDEN)
    add(P["lm_head"], (VOCAB, TXT_HIDDEN), 0.70)

    # ── vision family (Pixtral block naming) ─────────────────────────────────
    # 4-D conv patch embed: CSR-2D pruning must reshape (out, -1)
    add(P["vis_conv"], (VIS_HIDDEN, IN_CH, PATCH, PATCH), 0.0, dense=True)
    add_norm(P["vis_ln"], VIS_HIDDEN)
    v_out = VIS_HEADS * VIS_HDIM          # 16 == VIS_HIDDEN (square, unlike text)
    for i in range(VIS_LAYERS):
        p = f"{P['vis']}{i}."
        zf = rng.uniform(ZERO_FRAC_LO, ZERO_FRAC_HI)
        add(p+"attention.q_proj.weight", (v_out, VIS_HIDDEN), zf)
        add(p+"attention.k_proj.weight", (v_out, VIS_HIDDEN), zf)
        add(p+"attention.v_proj.weight", (v_out, VIS_HIDDEN), zf)
        add(p+"attention.o_proj.weight", (VIS_HIDDEN, v_out), zf)
        add(p+"feed_forward.gate_proj.weight", (VIS_INTER, VIS_HIDDEN), zf)
        add(p+"feed_forward.up_proj.weight",   (VIS_INTER, VIS_HIDDEN), zf)
        add(p+"feed_forward.down_proj.weight", (VIS_HIDDEN, VIS_INTER), zf)
        add_norm(p+"attention_norm.weight", VIS_HIDDEN)
        add_norm(p+"ffn_norm.weight", VIS_HIDDEN)

    # ── projector: norm -> patch_merger(2x2) -> linear_1 -> GELU -> linear_2 ──
    add_norm(P["proj"]+"norm.weight", VIS_HIDDEN)
    add(P["proj"]+"patch_merger.merging_layer.weight",
        (VIS_HIDDEN, VIS_HIDDEN * MERGE * MERGE), 0.30)
    add(P["proj"]+"linear_1.weight", (TXT_HIDDEN, VIS_HIDDEN), 0.30)
    add(P["proj"]+"linear_2.weight", (TXT_HIDDEN, TXT_HIDDEN), 0.30)

    return sd, izf


def expected_names(legacy_prefix: bool = False):
    """For diffing against a real model.safetensors.index.json weight_map."""
    sd, _ = build_toy_mistral_vlm_state_dict(legacy_prefix=legacy_prefix)
    return sorted(sd.keys())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-prefix", action="store_true")
    ap.add_argument("--out", default="/tmp/toy_mistral_vlm.pt")
    a = ap.parse_args()
    sd, izf = build_toy_mistral_vlm_state_dict(legacy_prefix=a.legacy_prefix)
    n = sum(t.numel() for t in sd.values())
    print(f"toy Mistral 3.1 VLM: {len(sd)} tensors, {n:,} params, "
          f"{n*4/1024:.0f} KB fp32, vintage={'legacy' if a.legacy_prefix else 'new'}")
    torch.save(sd, a.out)
    import json
    with open(a.out.replace('.pt', '_ground_truth.json'), 'w') as f:
        json.dump(izf, f, indent=1)
    print(f"saved: {a.out}")
