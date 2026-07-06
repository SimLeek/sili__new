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

Also includes the 4-D patch_conv.weight (conv patch embed, (1024,3,14,14) in
the real model). Kept DENSE throughout -- not pruned, not reshaped to CSR.
~602K params, small in every dimension; prior sparsification attempts on
vision-tower-scale tensors found poor ratios vs the 21M-scale text matrices
where CSR actually pays off. Sparsifying it is later-todo, not in scope here
(see refactoring_todo.md).

Prefix schema: VERIFIED against the real
mistralai/Mistral-Small-3.1-24B-Base-2503 model.safetensors.index.json
(fetched directly, July 2026). There is NO "model." top-level wrapper --
language_model.*, vision_tower.*, and multi_modal_projector.* are top-level
keys (e.g. language_model.lm_head.weight, vision_tower.patch_conv.weight).
This is the module's default schema. An `--alt-prefix` flag exists for a
"model."-wrapped variant this file previously (and wrongly) defaulted to,
based on an uncited memory of a transformers refactor -- no evidence for that
variant was found against the real download; it is kept only as a cheap
defensive alternate, not because a second real vintage is confirmed to exist.
See _prefixes() docstring below and docs/requirements_vlm_streaming_rtac.md
section 1.2 for the full account of this correction.

All numeric dimensions (hidden sizes, layer counts, head counts, head_dim,
intermediate sizes, spatial_merge_size, vocab_size) are VERIFIED against the
real config.json -- every value in the real/toy table below matched exactly
on first fetch; the toy's dimension choices needed no correction, only the
prefix schema did.
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


def _prefixes(alt_prefix: bool = False):
    """
    default (alt_prefix=False): VERIFIED against the real
    mistralai/Mistral-Small-3.1-24B-Base-2503 model.safetensors.index.json
    (fetched directly, July 2026). No "model." top-level wrapper exists --
    language_model.*, vision_tower.*, and multi_modal_projector.* are all
    top-level keys.

    alt_prefix=True: a "model."-wrapped variant this file previously
    defaulted to, based on a vague, uncited memory of "transformers
    refactored VLM key layout ~v4.52." No evidence for this variant was
    found when checked against the real download -- it may not exist for
    mistral3 at all. Kept only as a defensive/unlikely-but-cheap-to-support
    alternate structure, NOT because there is a known second vintage.
    Remove this branch entirely if nothing ever surfaces using it.
    """
    if alt_prefix:
        return dict(lang="model.language_model.layers.",
                    lang_embed="model.language_model.embed_tokens.weight",
                    lang_norm="model.language_model.norm.weight",
                    lm_head="lm_head.weight",
                    vis="model.vision_tower.transformer.layers.",
                    vis_conv="model.vision_tower.patch_conv.weight",
                    vis_ln="model.vision_tower.ln_pre.weight",
                    proj="model.multi_modal_projector.")
    # VERIFIED (see docstring above): real index.json weight_map keys, e.g.
    #   language_model.lm_head.weight
    #   language_model.model.embed_tokens.weight
    #   language_model.model.layers.{i}.self_attn.q_proj.weight
    #   language_model.model.norm.weight
    #   vision_tower.patch_conv.weight
    #   vision_tower.ln_pre.weight
    #   vision_tower.transformer.layers.{i}.attention.q_proj.weight
    #   multi_modal_projector.linear_1.weight
    return dict(lang="language_model.model.layers.",
                lang_embed="language_model.model.embed_tokens.weight",
                lang_norm="language_model.model.norm.weight",
                lm_head="language_model.lm_head.weight",
                vis="vision_tower.transformer.layers.",
                vis_conv="vision_tower.patch_conv.weight",
                vis_ln="vision_tower.ln_pre.weight",
                proj="multi_modal_projector.")


def build_toy_mistral_vlm_state_dict(seed: int = 1234, alt_prefix: bool = False):
    rng = np.random.default_rng(seed)
    P = _prefixes(alt_prefix)
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


def expected_names(alt_prefix: bool = False):
    """
    For diffing against a real model.safetensors.index.json weight_map.
    Default (alt_prefix=False) matches the VERIFIED schema -- diff against
    this first. Confirmed exact match against mistralai/Mistral-Small-3.1-
    24B-Base-2503's real index.json (July 2026): every language_model.*,
    vision_tower.*, and multi_modal_projector.* key name and layer count
    (40 language layers, 24 vision layers -- toy uses 12/6 as a scaled-down
    stand-in) lines up with the ground truth. Only the layer COUNT and the
    numeric dims differ (toy uses TXT_LAYERS=12 not 40, VIS_LAYERS=6 not 24,
    etc. -- see the real/toy table this module's docstring points to).
    """
    sd, _ = build_toy_mistral_vlm_state_dict(alt_prefix=alt_prefix)
    return sorted(sd.keys())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt-prefix", action="store_true",
                    help="Use the UNVERIFIED 'model.'-wrapped schema instead "
                         "of the verified default (see _prefixes docstring)")
    ap.add_argument("--out", default="/tmp/toy_mistral_vlm.pt")
    a = ap.parse_args()
    sd, izf = build_toy_mistral_vlm_state_dict(alt_prefix=a.alt_prefix)
    n = sum(t.numel() for t in sd.values())
    print(f"toy Mistral 3.1 VLM: {len(sd)} tensors, {n:,} params, "
          f"{n*4/1024:.0f} KB fp32, "
          f"schema={'ALT (unverified)' if a.alt_prefix else 'VERIFIED'}")
    torch.save(sd, a.out)
    import json
    with open(a.out.replace('.pt', '_ground_truth.json'), 'w') as f:
        json.dump(izf, f, indent=1)
    print(f"saved: {a.out}")
