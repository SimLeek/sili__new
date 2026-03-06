#!/usr/bin/env python3
"""
trace_model.py
──────────────
Generate a readable, executable Python trace file from a sparse_prune /
rnn_fold payload.  The generated file:

  • Lists every layer in forward-pass order, annotated with
    [SPARSE X%] / [DENSE: reason] / [!IMPLEMENT ME]
  • Groups transformer blocks and marks which are folded vs individual
  • Provides a runnable forward() skeleton wired to the sparse runtime
  • Has a __main__ section with correctly-sized example inputs that
    runs immediately and prints the first NotImplementedError so you
    know exactly what to implement next

Two tracing strategies are used:
  1. torch.fx symbolic trace  — attempted on the original reconstructed model
     when available; produces an accurate graph with real node ordering
  2. Structural trace         — walks the sparse payload's layer names; always
     works, no forward() required; used as primary or fallback

Usage
─────
  # Generate from a sparse payload (standalone)
  python trace_model.py model_sparse.pt

  # Generate from sparse + original (better node ordering via FX)
  python trace_model.py model_sparse.pt --original Qwen3-VL-7B/

  # Specify output path
  python trace_model.py model_sparse.pt -o my_trace.py

  # sparse_runtime.py calls this automatically after loading
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers shared with rnn_fold / sparse_prune
# ══════════════════════════════════════════════════════════════════════════════

_BLOCK_RE = re.compile(r'^(.*?\.)(\d+)(\..+)$')

def _parse_block_key(name: str) -> Optional[Tuple[str, int, str]]:
    m = _BLOCK_RE.match(name)
    if m is None:
        return None
    return m.group(1), int(m.group(2)), m.group(3)

def _safe(name: str) -> str:
    """Convert a dotted parameter name to a safe Python identifier."""
    return name.replace(".", "_").replace("-", "_")


# ══════════════════════════════════════════════════════════════════════════════
#  Layer descriptor
# ══════════════════════════════════════════════════════════════════════════════

class LayerDesc:
    """Everything known about one tensor in the payload."""
    __slots__ = ("name", "shape", "fmt", "sparsity", "block_idx",
                 "block_prefix", "suffix", "not_impl", "not_impl_reason",
                 "ndim")

    def __init__(self, name: str, shape: tuple, fmt: str, sparsity: float = 0.0):
        self.name     = name
        self.shape    = shape
        self.fmt      = fmt          # "sparse" | "dense" | "folded"
        self.sparsity = sparsity
        self.ndim     = len(shape)

        parsed = _parse_block_key(name)
        if parsed:
            self.block_prefix, self.block_idx, self.suffix = parsed
        else:
            self.block_prefix = None
            self.block_idx    = None
            self.suffix       = name

        # Mark layers that will definitely raise NotImplementedError
        self.not_impl        = False
        self.not_impl_reason = ""
        self._classify_not_impl()

    def _classify_not_impl(self) -> None:
        n = self.name.lower()
        s = self.suffix.lower() if self.suffix else ""

        # Embedding tables — row lookup, not matmul
        if "embed_tokens" in n or "word_embed" in n or "wte" in n:
            self.not_impl        = True
            self.not_impl_reason = "embedding lookup (F.embedding, not matmul)"
            return

        # Patch / position embeddings
        if "patch_embed" in n or "pos_embed" in n or "position_embed" in n:
            self.not_impl        = True
            self.not_impl_reason = "non-linear conv / positional embed"
            return

        # LayerNorm / RMSNorm — no sparse kernel
        if any(t in n for t in ("layernorm", "layer_norm", "rmsnorm",
                                  "input_layernorm", "post_attention_layernorm",
                                  "norm.weight", "norm.bias")):
            self.not_impl        = True
            self.not_impl_reason = "layer normalisation (no sparse mean/var kernel)"
            return

        # LM head — usually large and tied; softmax output is always dense
        if "lm_head" in n:
            self.not_impl        = True
            self.not_impl_reason = "lm_head projection + softmax (output is dense)"
            return

    @property
    def annotation(self) -> str:
        parts = []
        if self.fmt == "sparse":
            parts.append(f"SPARSE {self.sparsity:.1%}")
        elif self.fmt == "folded":
            parts.append(f"FOLDED {self.sparsity:.1%}")
        else:
            parts.append("DENSE")
        if self.not_impl:
            parts.append("!IMPLEMENT ME")
        return "[" + "  ".join(parts) + "]"


# ══════════════════════════════════════════════════════════════════════════════
#  Structural tracer — builds LayerDesc list from payload
# ══════════════════════════════════════════════════════════════════════════════

def structural_trace(payload: dict) -> List[LayerDesc]:
    """
    Walk sparse_state_dict and folded_blocks to produce an ordered list of
    LayerDesc objects.  No forward() execution required.
    """
    ssd = payload.get("sparse_state_dict", payload)
    descs: List[LayerDesc] = []

    def _nat_key(s):
        return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    for name in sorted(ssd.keys(), key=_nat_key):
        entry = ssd[name]
        if isinstance(entry, dict):
            csr   = entry.get("csr")
            dense = entry.get("dense")
            if dense is None:
                dense = entry.get("raw")
            raw_shape = entry.get("shape")
            if raw_shape is not None:
                shape = tuple(raw_shape)
            elif csr is not None:
                shape = tuple(csr.shape)
            elif dense is not None:
                shape = tuple(dense.shape)
            else:
                shape = ()
            if csr is not None:
                nnz      = int(csr.values().numel())
                n_elem   = shape[0] * shape[1] if len(shape) >= 2 else 1
                sparsity = 1.0 - nnz / n_elem if n_elem > 0 else 0.0
                fmt      = "sparse"
            else:
                sparsity = 0.0
                fmt      = "dense"
        elif isinstance(entry, torch.Tensor):
            shape    = tuple(entry.shape)
            sparsity = 0.0
            fmt      = "sparse" if entry.layout == torch.sparse_csr else "dense"
        else:
            continue

        descs.append(LayerDesc(name, shape, fmt, sparsity))

    # Folded blocks — add as a single descriptor per group
    for fd in payload.get("folded_blocks", []):
        prefix      = fd.get("prefix", "")
        n_folds     = fd.get("n_folds", 1)
        block_range = fd.get("block_indices", [])
        for suffix, csr in fd.get("stacked_weights", {}).items():
            shape    = tuple(csr.shape) if isinstance(csr, torch.Tensor) else ()
            nnz      = int(csr.values().numel()) if isinstance(csr, torch.Tensor) else 0
            n_elem   = shape[0] * shape[1] if len(shape) >= 2 else 1
            sparsity = 1.0 - nnz / n_elem if n_elem > 0 else 0.0
            fake_name = f"{prefix}{block_range[0]}{suffix}__folded_{n_folds}"
            d = LayerDesc(fake_name, shape, "folded", sparsity)
            descs.append(d)

    return descs


# ══════════════════════════════════════════════════════════════════════════════
#  FX symbolic tracer (optional, requires original model)
# ══════════════════════════════════════════════════════════════════════════════

def fx_trace(model: nn.Module, example_inputs: dict) -> Optional[object]:
    """
    Attempt torch.fx symbolic trace.  Returns a GraphModule or None on failure.

    Uses a permissive Tracer subclass that treats unknown leaf modules
    (anything that would raise NotImplementedError) as opaque call_module
    nodes rather than failing the trace.
    """
    from torch.fx import Tracer, GraphModule

    class _PermissiveTracer(Tracer):
        """Treat every nn.Module as a leaf so we never enter forward()."""
        def is_leaf_module(self, m: nn.Module, qualname: str) -> bool:
            return True

    tracer = _PermissiveTracer()
    dummy  = example_inputs.get("input_ids", torch.zeros(1, 32, dtype=torch.long))
    graph  = tracer.trace(model, concrete_args={"input_ids": dummy})
    return GraphModule(model, graph)


# ══════════════════════════════════════════════════════════════════════════════
#  Block grouper
# ══════════════════════════════════════════════════════════════════════════════

def group_into_blocks(descs: List[LayerDesc]) -> List[Tuple[Optional[int], List[LayerDesc]]]:
    """
    Group LayerDesc list into (block_index_or_None, [LayerDesc]) pairs.

    Layers with the same block_prefix and block_idx are grouped together.
    Layers outside any block (embeddings, final norm, lm_head) have index None.
    """
    groups: List[Tuple[Optional[int], List[LayerDesc]]] = []
    cur_idx   : Optional[int]  = "UNSET"  # type: ignore
    cur_prefix: Optional[str]  = None
    cur_group : List[LayerDesc] = []

    for d in descs:
        key = (d.block_prefix, d.block_idx)

        if d.block_idx is None:
            # Not in any block — flush current, emit as standalone
            if cur_group:
                groups.append((cur_idx if cur_idx != "UNSET" else None, cur_group))
                cur_group  = []
                cur_idx    = "UNSET"  # type: ignore
                cur_prefix = None
            groups.append((None, [d]))

        elif cur_idx == "UNSET" or key == (cur_prefix, cur_idx):
            cur_idx    = d.block_idx
            cur_prefix = d.block_prefix
            cur_group.append(d)

        else:
            groups.append((cur_idx, cur_group))
            cur_idx    = d.block_idx
            cur_prefix = d.block_prefix
            cur_group  = [d]

    if cur_group:
        groups.append((cur_idx if cur_idx != "UNSET" else None, cur_group))

    return groups


# ══════════════════════════════════════════════════════════════════════════════
#  Example input inference
# ══════════════════════════════════════════════════════════════════════════════

def infer_example_inputs(payload: dict) -> dict:
    """
    Infer reasonable example input shapes from the payload's weight tensors.

    Returns a dict with Python source snippets suitable for the generated file.
    """
    ssd = payload.get("sparse_state_dict", {})

    def _shape(name_fragment: str) -> Optional[tuple]:
        for k, v in ssd.items():
            if name_fragment in k:
                if isinstance(v, dict):
                    return tuple(v.get("shape") or ())
                if isinstance(v, torch.Tensor):
                    return tuple(v.shape)
        return None

    # Vocab size + d_model from embedding
    embed_shape = _shape("embed_tokens.weight") or _shape("wte.weight")
    vocab_size  = embed_shape[0] if embed_shape else 32000
    d_model     = embed_shape[1] if embed_shape and len(embed_shape) > 1 else 4096
    seq_len     = 32

    # Vision inputs?  Look for patch embed conv weight
    patch_shape = _shape("patch_embed.proj.weight")
    has_vision  = patch_shape is not None
    if has_vision and len(patch_shape) >= 2:
        in_chans   = patch_shape[1] if len(patch_shape) >= 4 else 3
        patch_h    = patch_shape[-2] if len(patch_shape) >= 4 else 14
        patch_w    = patch_shape[-1] if len(patch_shape) >= 4 else 14
        img_h      = patch_h * 16
        img_w      = patch_w * 16
    else:
        in_chans, img_h, img_w = 3, 448, 448

    inputs = {
        "vocab_size": vocab_size,
        "d_model":    d_model,
        "seq_len":    seq_len,
        "has_vision": has_vision,
        "in_chans":   in_chans,
        "img_h":      img_h,
        "img_w":      img_w,
    }
    return inputs


# ══════════════════════════════════════════════════════════════════════════════
#  Code generator
# ══════════════════════════════════════════════════════════════════════════════

def _layer_comment(d: LayerDesc, indent: int = 4) -> str:
    pad   = " " * indent
    shape = str(d.shape)
    ann   = d.annotation
    line  = f"{pad}#  {d.name:<60} {shape:<22} {ann}"
    if d.not_impl:
        line += f"\n{pad}#     reason: {d.not_impl_reason}"
    return line


def _detect_block_config(groups):
    """
    Examine the first transformer block's weight dict to detect architecture:
      - n_heads, n_kv_heads, head_dim  (from q/k projection shapes)
      - has_swiglu                     (gate_proj present)
      - norm suffix names              (pre/post attention)

    Returns a dict of config values used when emitting transformer_block().
    """
    attn_sfx = ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
                "query", "key", "value", "in_proj")
    norm_sfx = ("input_layernorm", "post_attention_layernorm",
                "layernorm", "layer_norm", "rmsnorm")

    cfg = {
        "pre_norm":  "input_layernorm.weight",
        "post_norm": "post_attention_layernorm.weight",
        "has_swiglu": False,
        "has_gqa": False,
    }

    for bidx, bdescs in groups:
        if bidx is None:
            continue
        norms = [d.suffix.lstrip(".") for d in bdescs
                 if any(k in d.suffix for k in norm_sfx)]
        if len(norms) >= 1:
            cfg["pre_norm"]  = norms[0]
        if len(norms) >= 2:
            cfg["post_norm"] = norms[1]

        # Detect GQA: q_proj out != k_proj out
        q_shape = next((d.shape for d in bdescs if "q_proj" in d.suffix), None)
        k_shape = next((d.shape for d in bdescs if "k_proj" in d.suffix), None)
        if q_shape and k_shape and len(q_shape) >= 1 and len(k_shape) >= 1:
            cfg["has_gqa"] = q_shape[0] != k_shape[0]

        cfg["has_swiglu"] = any("gate_proj" in d.suffix for d in bdescs)
        break   # only need block 0

    return cfg


def generate_trace_file(
    payload,
    output_path,
    sparse_path    = "model_sparse.pt",
    original_model = None,
    gm             = None,
):
    """
    Write the trace file.

    Generated file layout (four sections, each with a suggested split target):
        ops.py    -- sili imports (the actual leaf ops live in sili)
        blocks.py -- transformer_block() for this architecture
        model.py  -- SiliModel: embed -> block loop -> norm -> lm_head
        run.py    -- weight loading, example inputs, inference harness
    """
    from datetime import datetime

    descs  = structural_trace(payload)
    groups = group_into_blocks(descs)
    inp    = infer_example_inputs(payload)
    cfg    = _detect_block_config(groups)

    block_indices = sorted(
        set(d.block_idx for d in descs if d.block_idx is not None), key=int
    )
    n_blocks   = len(block_indices)
    has_vision = inp["has_vision"]
    vocab      = inp["vocab_size"]
    d_model    = inp["d_model"]
    seq_len    = inp["seq_len"]
    in_chans   = inp["in_chans"]
    img_h      = inp["img_h"]
    img_w      = inp["img_w"]
    has_swiglu = cfg["has_swiglu"]
    has_gqa    = cfg["has_gqa"]
    pre_norm   = cfg["pre_norm"]
    post_norm  = cfg["post_norm"]
    now        = datetime.now().strftime("%Y-%m-%d %H:%M")

    L = []
    e = L.append

    def section(title, split_to, *notes):
        e("")
        e("# " + "=" * 70)
        e(f"# {title}")
        e(f"# Suggested split -> {split_to}")
        for n in notes:
            e(f"# {n}")
        e("# " + "=" * 70)
        e("")

    def lines(*ls):
        for l in ls:
            e(l)

    # =========================================================================
    # File header
    # =========================================================================
    vision_note = (
        f"yes  ({in_chans}ch  {img_h}x{img_w})" if has_vision else "no"
    )
    lines(
        "#!/usr/bin/env python3",
        '"""',
        f"model_trace.py  --  auto-generated by trace_model.py  {now}",
        f"Source      : {sparse_path}",
        f"Blocks      : {n_blocks}",
        f"d_model     : {d_model}",
        f"vocab_size  : {vocab}",
        f"vision      : {vision_note}",
        f"GQA         : {has_gqa}",
        f"activation  : {'swiglu' if has_swiglu else 'gelu'}",
        "",
        "HOW TO USE",
        "  python model_trace.py",
        "  Fails at the first ImportError from sili.",
        "  Implement that function in sili, then re-run.",
        "",
        "FILE SPLIT PLAN",
        "  ops.py    -- sili import list (already this file's Section 1)",
        "  blocks.py -- transformer_block() (Section 2)",
        "  model.py  -- SiliModel class (Section 3)",
        "  run.py    -- weight loading and inference harness (Section 4)",
        '"""',
        "",
    )

    # =========================================================================
    # Imports
    # =========================================================================
    lines(
        "from __future__ import annotations",
        "from typing import Optional",
        "import torch",
        "",
    )

    # =========================================================================
    # SECTION 1 -- OPS  (just imports -- sili provides the implementations)
    # =========================================================================
    section(
        "SECTION 1 -- OPS", "ops.py",
        "All leaf ops are imported from sili.",
        "If something is missing, add it to sili and re-run -- no edits needed here.",
    )

    activation_import = "swiglu" if has_swiglu else "gelu"
    lines(
        "from sili import (",
        "    Module,",
        "    Tensor,",
        "    embed,            # embed(token_ids, weight) -> (B, T, D)",
        "    rmsnorm,          # rmsnorm(x, weight, eps) -> (B, T, D)",
        "    sparse_linear,    # sparse_linear(x, weight_csr, bias) -> (B, T, out)",
        f"    {activation_import},",
    )
    if activation_import == "swiglu":
        e("    #   swiglu(gate, up) = silu(gate) * up")
    else:
        e("    #   gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))")
    lines(
        "    attention,        # attention(q, k, v, mask, n_heads, n_kv_heads)",
        "    rope,             # rope(x, position_ids) -> rotary-position-encoded x",
        ")",
        "",
        "from sparse_runtime import load_sparse_runtime",
        "",
    )

    # =========================================================================
    # SECTION 2 -- BLOCKS  (single transformer_block, not N copies)
    # =========================================================================

    # Collect the first block's weight list for the docstring
    first_block_descs = next(
        (bd for bi, bd in groups if bi is not None), []
    )

    section(
        "SECTION 2 -- BLOCKS", "blocks.py",
        "Single transformer_block() covers all N blocks -- they share the same",
        "architecture; only their weights differ.  SiliModel passes the right",
        "weight dict for each index via self._w(i).",
    )

    lines(
        "",
        "def transformer_block(x, w, mask=None):",
        "    # One Qwen/LLaMA-style transformer block.",
        "    # Args:",
        "    #   x    : (B, T, D)  hidden state",
        "    #   w    : weight dict keyed by suffix after 'layers.N.'",
        "    #   mask : (B, 1, T, T) additive causal mask, or None",
        "    # Weight keys expected in w:",
    )
    for d in first_block_descs:
        key    = d.suffix.lstrip(".")
        layout = "CSR" if d.fmt == "sparse" else "dense"
        e(f"    #   {key:<50} {str(d.shape):<22} [{layout}]")
    lines(
        "",
        f"    # -- Pre-attention norm  (key: '{pre_norm}')",
        f"    x_normed = rmsnorm(x, w['{pre_norm}'])",
        "",
        "    # -- Q / K / V projections  (CSR sparse_linear)",
        "    q = sparse_linear(x_normed, w['self_attn.q_proj.weight'])",
        "    k = sparse_linear(x_normed, w['self_attn.k_proj.weight'])",
        "    v = sparse_linear(x_normed, w['self_attn.v_proj.weight'])",
        "",
        "    # -- Rotary position encoding",
        "    q = rope(q)",
        "    k = rope(k)",
        "",
        "    # -- Scaled dot-product attention",
        "    #    GQA: sili's attention handles n_kv_heads < n_heads automatically",
        "    attn_out = attention(q, k, v, mask=mask)",
        "",
        "    # -- Output projection + residual",
        "    x = x + sparse_linear(attn_out, w['self_attn.o_proj.weight'])",
        "",
        f"    # -- Post-attention norm  (key: '{post_norm}')",
        f"    x_normed2 = rmsnorm(x, w['{post_norm}'])",
        "",
    )

    if has_swiglu:
        lines(
            "    # -- SwiGLU MLP",
            "    gate = sparse_linear(x_normed2, w['mlp.gate_proj.weight'])",
            "    up   = sparse_linear(x_normed2, w['mlp.up_proj.weight'])",
            "    x    = x + sparse_linear(swiglu(gate, up), w['mlp.down_proj.weight'])",
        )
    else:
        lines(
            "    # -- GELU MLP",
            "    hidden = sparse_linear(x_normed2, w['mlp.fc1.weight'])",
            "    x      = x + sparse_linear(gelu(hidden), w['mlp.fc2.weight'])",
        )

    lines(
        "",
        "    return x",
        "",
    )

    # =========================================================================
    # SECTION 3 -- MODEL
    # =========================================================================
    section(
        "SECTION 3 -- MODEL", "model.py",
        "SiliModel orchestrates embed -> block loop -> norm -> lm_head.",
        "self.sparse  : {name: SparseLinear}  CSR weights from sparse_prune",
        "self.dense   : {name: Tensor}         dense weights (norms, embeds)",
    )

    lines(
        "",
        "class SiliModel(Module):",
        "    # Full Qwen/LLaMA forward pass backed by the sparse runtime.",
        "",
        "    def __init__(self, runtime) -> None:",
        "        super().__init__()",
        "        base = getattr(runtime, 'inner', runtime)  # unwrap RNNAllLayer",
        "        self.sparse = base.sparse_layers   # {name: SparseLinear}",
        "        self.dense  = base._dense_buffers  # {name: Tensor}",
        "        self.fl     = base.folded_layers",
        f"        self.n_blocks = {n_blocks}",
        "",
        "    def _w(self, block_idx: int) -> dict:",
        "        # Weight dict for one block, keyed by suffix after layers.N.",
        "        prefix = f'model.layers.{block_idx}.'",
        "        w = {}",
        "        for name, layer in self.sparse.items():",
        "            if name.startswith(prefix):",
        "                w[name[len(prefix):]] = layer.weight_csr",
        "        for name, tensor in self.dense.items():",
        "            if name.startswith(prefix):",
        "                w[name[len(prefix):]] = tensor",
        "        return w",
        "",
    )

    fwd = [
        "    def forward(",
        "        self,",
        "        input_ids: Tensor,",
        "        attention_mask: Optional[Tensor] = None,",
    ]
    if has_vision:
        fwd.append("        pixel_values: Optional[Tensor] = None,")
    fwd.append("    ) -> Tensor:")
    lines(*fwd)
    e("")

    if has_vision:
        lines(
            "        # -- Vision tokens (Qwen3-VL) -----------------------------------",
            "        # patch_embed + ViT blocks + merger should live in ops.py.",
            "        vision_tokens = None",
            "        if pixel_values is not None:",
            "            from sili import vit_encode  # add to sili when ready",
            "            vision_tokens = vit_encode(pixel_values, self.sparse, self.dense)",
            "",
        )

    lines(
        "        # -- Text embedding",
        "        embed_w = self.dense.get('model.embed_tokens.weight')",
        "        if embed_w is None:",
        "            embed_w = next((v for k, v in self.dense.items()",
        "                            if 'embed_tokens' in k or 'wte' in k), None)",
        f"        x = embed(input_ids, embed_w)   # (B, T, {d_model})",
        "",
    )

    if has_vision:
        lines(
            "        if vision_tokens is not None:",
            "            x = torch.cat([vision_tokens, x], dim=1)",
            "            if attention_mask is not None:",
            "                ones = torch.ones(",
            "                    x.shape[0], vision_tokens.shape[1],",
            "                    device=x.device, dtype=attention_mask.dtype,",
            "                )",
            "                attention_mask = torch.cat([ones, attention_mask], dim=1)",
            "",
        )

    lines(
        "        # -- Transformer blocks",
        "        for i in range(self.n_blocks):",
        "            print(f'[forward]  block {i}/{self.n_blocks - 1}  x={x.shape}')",
        "            x = transformer_block(x, self._w(i), mask=attention_mask)",
        "",
        "        # -- Final norm",
        "        norm_w = self.dense.get('model.norm.weight')",
        "        if norm_w is None:",
        "            norm_w = next(",
        "                (v for k, v in self.dense.items()",
        "                 if k.endswith('norm.weight') and 'layers' not in k),",
        "                None,",
        "            )",
        "        x = rmsnorm(x, norm_w)",
        "",
        "        # -- LM head",
        "        lm_w = self.dense.get('lm_head.weight')",
        "        if lm_w is None:",
        "            lm_w = next((v for k, v in self.dense.items()",
        "                         if 'lm_head' in k), None)",
        f"        logits = sparse_linear(x, lm_w)   # (B, T, vocab={vocab})",
        "        return logits",
        "",
    )

    # =========================================================================
    # SECTION 4 -- RUN
    # =========================================================================
    section(
        "SECTION 4 -- RUN", "run.py",
        "Weight loading, example inputs, and inference harness.",
        "For training: add a loss function and call sili's backward() here.",
    )

    lines(
        f'SPARSE_PATH = "{sparse_path}"  # update if you moved the file',
        f"SEQ_LEN    = {seq_len}",
        f"VOCAB_SIZE = {vocab}",
        "",
        "",
        "if __name__ == '__main__':",
        "    import os, sys, traceback as _tb",
        "",
        "    if not os.path.exists(SPARSE_PATH):",
        "        print(f'[run]  sparse payload not found: {SPARSE_PATH}')",
        "        print( '[run]  update SPARSE_PATH at the top of this file')",
        "        sys.exit(1)",
        "",
        "    print(f'[run]  Loading runtime from {SPARSE_PATH} ...')",
        "    runtime = load_sparse_runtime(SPARSE_PATH, generate_trace=False)",
        "    model   = SiliModel(runtime)",
        "",
        "    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))",
    )
    if has_vision:
        lines(
            f"    pixel_values = torch.randn(1, {in_chans}, {img_h}, {img_w})",
            f"    #   (B=1, C={in_chans}, H={img_h}, W={img_w})",
        )
    else:
        e("    pixel_values = None")

    lines(
        "",
        "    print()",
        "    print('─' * 60)",
        "    print('[run]  Forward pass')",
        "    print('─' * 60)",
        "    print()",
        "",
        "    try:",
    )
    if has_vision:
        e("        out = model(input_ids, pixel_values=pixel_values)")
    else:
        e("        out = model(input_ids)")
    lines(
        "        print(f'✓  output shape: {out.shape}')",
        "        print(f'   logits: {out.min():.3f} .. {out.max():.3f}')",
        "",
        "    except ImportError as exc:",
        "        print(f'->  ImportError: {exc}')",
        "        print('[run]  Add the missing function to sili, then re-run.')",
        "",
        "    except Exception:",
        "        _tb.print_exc()",
        "",
    )

    source = "\n".join(L)
    Path(output_path).write_text(source, encoding="utf-8")
    return output_path


def generate_for_runtime(
    payload:       dict,
    sparse_path:   str,
    output_path:   Optional[str] = None,
    original_model: Optional[nn.Module] = None,
) -> str:
    """
    Generate a trace file for a sparse_prune / rnn_fold payload.

    Called automatically by sparse_runtime.load_sparse_runtime().

    Parameters
    ----------
    payload        : the loaded .pt payload dict
    sparse_path    : the path string used to load it (embedded in generated file)
    output_path    : where to write the .py file; defaults to <sparse_path stem>_trace.py
    original_model : optional original nn.Module for FX-assisted ordering

    Returns
    -------
    Path of the generated file (str).
    """
    if output_path is None:
        stem        = Path(sparse_path).stem
        output_path = str(Path(sparse_path).parent / f"{stem}_trace.py")

    gm = None
    if original_model is not None:
        inp = infer_example_inputs(payload)
        gm  = fx_trace(original_model, {"input_ids": torch.zeros(1, inp["seq_len"], dtype=torch.long)})

    generate_trace_file(
        payload=payload,
        output_path=output_path,
        sparse_path=sparse_path,
        original_model=original_model,
        gm=gm,
    )
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a runnable traced forward skeleton from a sparse payload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input",       help="sparse_prune / rnn_fold .pt payload")
    p.add_argument("-o", "--output", default=None,
                   help="Output .py path  (default: <input>_trace.py)")
    p.add_argument("--original",  default=None, metavar="PATH",
                   help="Original model path for FX-assisted node ordering")
    p.add_argument("--run",       action="store_true",
                   help="Execute the generated file immediately after writing")
    args = p.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)

    original_model = None
    if args.original:
        import model_reconstruct as mr
        original_model = mr.reconstruct_model(args.original)

    out = generate_for_runtime(
        payload=payload,
        sparse_path=args.input,
        output_path=args.output,
        original_model=original_model,
    )
    print(f"[trace_model]  Written: {out}")

    if args.run:
        import subprocess
        print(f"[trace_model]  Running {out} ...\n")
        subprocess.run([sys.executable, out])


if __name__ == "__main__":
    main()