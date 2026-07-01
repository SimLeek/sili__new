#!/usr/bin/env python3
"""
model_reconstruct.py
────────────────────
Infer the architecture of a HuggingFace-style model from its weight file
(.bin pickle or .safetensors) and return a runnable nn.Module with those
weights already loaded.

Supported architecture families
─────────────────────────────────
  qwen3vl  — Qwen2-VL, Qwen3-VL  (vision-language, checked FIRST)
             Key signals: visual.patch_embed.proj, visual.blocks.N.attn.qkv
             LM backbone is Qwen3/LLaMA-style with GQA + RMSNorm + SwiGLU
  llama    — LLaMA 1/2/3, Mistral, Mixtral, Phi-3, DeepSeek, Gemma
             Key signals: q_proj / k_proj / v_proj, gate_proj, RMSNorm
  gpt2     — GPT-2, GPT-Neo, CodeGen, SantaCoder
             Key signals: c_attn (fused QKV), c_proj, mlp.c_fc
  bert     — BERT, RoBERTa, DeBERTa, DistilBERT, ELECTRA
             Key signals: attention.self.query, intermediate.dense

For unrecognised weight files the loader still returns a GenericStateModule
that holds the raw tensors and can be probed, but forward() raises
NotImplementedError since we have no architecture graph to run.

Usage
─────
  # Python API
  from model_reconstruct import reconstruct_model
  model = reconstruct_model("llama-7b/model.safetensors")
  model.eval()
  out = model(input_ids)

  # CLI — inspect and optionally save the reconstructed module
  python model_reconstruct.py model.safetensors
  python model_reconstruct.py model.bin --save model_module.pt
  python model_reconstruct.py model.safetensors --show-arch
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  Weight file loading
# ══════════════════════════════════════════════════════════════════════════════

def load_weights(path: str) -> Dict[str, torch.Tensor]:
    """
    Load a flat {name: tensor} state dict from a .bin or .safetensors file.

    Handles:
      .safetensors  — memory-mapped, no pickle, fastest
      .bin / .pt / .pth — torch.load (pickle); uses weights_only=True for safety
      multi-shard   — if path is "model-00001-of-00003.safetensors" the caller
                      is responsible for merging shards; this loads one shard.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError:
            sys.exit("pip install safetensors")
        return dict(load_file(str(p)))

    obj = torch.load(str(p), map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        return obj.get("state_dict", obj)
    if hasattr(obj, "state_dict"):
        return dict(obj.state_dict())
    raise ValueError(f"Cannot interpret object of type {type(obj)}")


def load_weights_sharded(directory: str) -> Dict[str, torch.Tensor]:
    """
    Load and merge all shards found in a directory.

    Looks for files matching:
      model-*-of-*.safetensors
      pytorch_model-*-of-*.bin

    Returns a single merged state dict (all shards in CPU memory).
    """
    d = Path(directory)
    shards: List[Path] = sorted(
        list(d.glob("model-*-of-*.safetensors")) +
        list(d.glob("pytorch_model-*-of-*.bin")) +
        list(d.glob("model.safetensors")) +
        list(d.glob("pytorch_model.bin"))
    )
    if not shards:
        raise FileNotFoundError(f"No model weight files found in {directory}")

    merged: Dict[str, torch.Tensor] = {}
    for shard in shards:
        print(f"  [load shard]  {shard.name}")
        merged.update(load_weights(str(shard)))
    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  Architecture detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_family(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Classify the model family from parameter name patterns.

    Returns one of: 'qwen3vl', 'llama', 'gpt2', 'bert', 'unknown'.

    qwen3vl is checked before llama because Qwen3-VL LM weights are
    llama-style; the presence of 'visual.' keys is the disambiguator.
    """
    names = set(state_dict.keys())

    # Qwen2-VL / Qwen3-VL: vision encoder keys are unambiguous
    qwen_vl_signals = [
        any(n.startswith("visual.patch_embed") for n in names),
        any("visual.blocks" in n for n in names),
        any("visual.merger" in n for n in names),
    ]
    if sum(qwen_vl_signals) >= 2:
        return "qwen3vl"

    # LLaMA family: rotary embeddings, SwiGLU MLP, RMSNorm, separate Q/K/V
    llama_signals = [
        any("q_proj" in n for n in names),
        any("gate_proj" in n for n in names),
        any("input_layernorm" in n or "norm.weight" in n for n in names),
    ]
    if sum(llama_signals) >= 2:
        return "llama"

    # GPT-2 family: fused c_attn, Conv1D-style weights
    gpt2_signals = [
        any("c_attn" in n for n in names),
        any("c_proj" in n for n in names),
        any("mlp.c_fc" in n or "mlp.c_proj" in n for n in names),
    ]
    if sum(gpt2_signals) >= 2:
        return "gpt2"

    # BERT family: separate self-attention query/key/value, intermediate.dense
    bert_signals = [
        any("attention.self.query" in n for n in names),
        any("intermediate.dense" in n for n in names),
        any("attention.output.dense" in n for n in names),
    ]
    if sum(bert_signals) >= 2:
        return "bert"

    return "unknown"


def _count_layers(state_dict: Dict[str, torch.Tensor], prefix_re: str) -> int:
    """Count the number of unique integer indices matching a prefix pattern."""
    indices = set()
    pat = re.compile(prefix_re)
    for name in state_dict:
        m = pat.search(name)
        if m:
            indices.add(int(m.group(1)))
    return len(indices)


# ══════════════════════════════════════════════════════════════════════════════
#  LLaMA family
# ══════════════════════════════════════════════════════════════════════════════

class _LlamaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class _LlamaRotaryEmbedding(nn.Module):
    """RoPE — recomputed on the fly, no stored parameters."""
    def __init__(self, dim: int, max_seq: int = 4096, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq = max_seq

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rotary(q, k, cos, sin):
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class _LlamaAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int],
                 head_dim: Optional[int], max_seq: int):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim   = head_dim or (d_model // n_heads)
        self.groups     = n_heads // self.n_kv_heads   # GQA groups

        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.rope   = _LlamaRotaryEmbedding(self.head_dim, max_seq=max_seq)

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        cos, sin = self.rope(T, x.device)

        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rotary(q, k, cos[:, :, :T, :], sin[:, :, :T, :])

        # Expand KV heads for GQA
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        # Causal mask
        causal = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        scores = scores + causal
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = F.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class _LlamaMLP(nn.Module):
    def __init__(self, d_model: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
        self.up_proj   = nn.Linear(d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _LlamaBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int],
                 head_dim: Optional[int], intermediate: int, max_seq: int,
                 rms_eps: float = 1e-5):
        super().__init__()
        self.input_layernorm       = _LlamaRMSNorm(d_model, rms_eps)
        self.self_attn             = _LlamaAttention(d_model, n_heads, n_kv_heads,
                                                     head_dim, max_seq)
        self.post_attention_layernorm = _LlamaRMSNorm(d_model, rms_eps)
        self.mlp                   = _LlamaMLP(d_model, intermediate)

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaModel(nn.Module):
    """
    Reconstructed LLaMA-family decoder.

    Covers: LLaMA 1/2/3, Mistral 7B, Mixtral (dense path), Qwen2, Phi-3,
    DeepSeek-V2 (dense), Gemma (uses GeGLU but same weight names).
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, n_kv_heads: Optional[int], head_dim: Optional[int],
                 intermediate: int, max_seq: int = 4096, rms_eps: float = 1e-5,
                 tie_embeddings: bool = False):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            _LlamaBlock(d_model, n_heads, n_kv_heads, head_dim,
                        intermediate, max_seq, rms_eps)
            for _ in range(n_layers)
        ])
        self.norm   = _LlamaRMSNorm(d_model, rms_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        return self.lm_head(x)


def _infer_llama_hparams(sd: Dict[str, torch.Tensor]) -> dict:
    """Infer LLaMA hyperparameters from weight shapes."""
    embed = sd.get("model.embed_tokens.weight")
    if embed is None:
        embed = sd.get("embed_tokens.weight")
    vocab_size, d_model = embed.shape

    n_layers = _count_layers(sd, r"layers\.(\d+)\.")

    # n_heads from q_proj output dim
    q_weight = next(v for k, v in sd.items() if "q_proj.weight" in k)
    q_out = q_weight.shape[0]

    # n_kv_heads from k_proj output dim (GQA)
    k_weight = next(v for k, v in sd.items() if "k_proj.weight" in k)
    k_out = k_weight.shape[0]

    # head_dim — often d_model // n_heads but can differ (Mistral)
    # Recover from: n_heads * head_dim = q_out
    # and:          n_kv_heads * head_dim = k_out
    # Try head_dim = d_model // round(q_out / (k_out / round(k_out/d_model)))
    # Simpler: find the GCD pattern
    from math import gcd
    head_dim = gcd(q_out, k_out)
    # Make sure head_dim is sensible (at least 16, divides both evenly)
    for hd in [128, 64, 96, 80, 256, 32, 16]:
        if q_out % hd == 0 and k_out % hd == 0:
            head_dim = hd
            break

    n_heads    = q_out // head_dim
    n_kv_heads = k_out // head_dim

    # intermediate from gate_proj
    gate = next(v for k, v in sd.items() if "gate_proj.weight" in k)
    intermediate = gate.shape[0]

    # rms_eps: not recoverable from weights alone, use LLaMA default
    rms_eps = 1e-5

    # tie_embeddings: check if lm_head.weight is absent (tied)
    tie = not any("lm_head.weight" in k for k in sd)

    return dict(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
        intermediate=intermediate, rms_eps=rms_eps, tie_embeddings=tie,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  GPT-2 family
# ══════════════════════════════════════════════════════════════════════════════

class _GPT2Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # GPT-2 uses a single Conv1D (= Linear transposed) for Q, K, V fused
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.max_seq = max_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)                                       # [B,T,3C]
        q, k, v = qkv.split(C, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        causal = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        scores = scores + causal
        attn   = F.softmax(scores, dim=-1)
        out    = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        return self.c_proj(out)


class _GPT2MLP(nn.Module):
    def __init__(self, d_model: int, intermediate: int):
        super().__init__()
        self.c_fc   = nn.Linear(d_model, intermediate)
        self.c_proj = nn.Linear(intermediate, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class _GPT2Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, intermediate: int, max_seq: int):
        super().__init__()
        self.ln_1  = nn.LayerNorm(d_model)
        self.attn  = _GPT2Attention(d_model, n_heads, max_seq)
        self.ln_2  = nn.LayerNorm(d_model)
        self.mlp   = _GPT2MLP(d_model, intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    """Reconstructed GPT-2 style decoder (GPT-2, GPT-Neo, CodeGen, SantaCoder)."""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, intermediate: int, max_seq: int = 2048):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_seq, d_model)
        self.h   = nn.ModuleList([
            _GPT2Block(d_model, n_heads, intermediate, max_seq)
            for _ in range(n_layers)
        ])
        self.ln_f  = nn.LayerNorm(d_model)
        # GPT-2 ties lm_head to wte
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.wte(input_ids) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def _infer_gpt2_hparams(sd: Dict[str, torch.Tensor]) -> dict:
    embed = sd.get("transformer.wte.weight")
    if embed is None:
        embed = sd.get("wte.weight")
    vocab_size, d_model = embed.shape
    n_layers = _count_layers(sd, r"\.h\.(\d+)\.")
    # c_attn is [d_model, 3*d_model] (Conv1D stores transposed)
    c_attn = next(v for k, v in sd.items() if "c_attn.weight" in k)
    # Conv1D weight shape is [in, out] — opposite of nn.Linear
    if c_attn.shape[0] == d_model:
        three_d = c_attn.shape[1]
    else:
        three_d = c_attn.shape[0]
    # three_d = 3*d_model confirms d_model; n_heads from layernorm or heuristic
    # Usual GPT-2 ratios: 64 dims / head
    n_heads = d_model // 64
    for nh in [16, 12, 8, 4, 1]:
        if d_model % nh == 0:
            n_heads = nh
            break
    # intermediate from mlp
    fc = next((v for k, v in sd.items() if "mlp.c_fc.weight" in k), None)
    intermediate = fc.shape[1] if fc is not None else 4 * d_model
    # pos embedding tells us max_seq
    wpe = sd.get("transformer.wpe.weight")
    if wpe is None:
        wpe = sd.get("wpe.weight")
    max_seq = wpe.shape[0] if wpe is not None else 2048
    return dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                n_heads=n_heads, intermediate=intermediate, max_seq=max_seq)


# ══════════════════════════════════════════════════════════════════════════════
#  BERT family
# ══════════════════════════════════════════════════════════════════════════════

class _BERTAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out   = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        def split(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = F.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class _BERTLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, intermediate: int):
        super().__init__()
        self.attention   = _BERTAttention(d_model, n_heads)
        self.attn_norm   = nn.LayerNorm(d_model)
        self.intermediate = nn.Linear(d_model, intermediate)
        self.output_dense = nn.Linear(intermediate, d_model)
        self.output_norm  = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn_norm(x + self.attention(x, attention_mask))
        x = self.output_norm(x + self.output_dense(F.gelu(self.intermediate(x))))
        return x


class BERTModel(nn.Module):
    """Reconstructed BERT-family encoder (BERT, RoBERTa, DistilBERT, ELECTRA)."""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, intermediate: int, max_seq: int = 512,
                 type_vocab_size: int = 2):
        super().__init__()
        self.word_embeddings     = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq, d_model)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, d_model)
        self.emb_norm   = nn.LayerNorm(d_model)
        self.encoder    = nn.ModuleList([
            _BERTLayer(d_model, n_heads, intermediate) for _ in range(n_layers)
        ])
        self.pooler = nn.Linear(d_model, d_model)

    def forward(self, input_ids: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        tt   = token_type_ids if token_type_ids is not None \
               else torch.zeros_like(input_ids)
        x = (self.word_embeddings(input_ids)
             + self.position_embeddings(pos)
             + self.token_type_embeddings(tt))
        x = self.emb_norm(x)
        # Extend attention mask: [B,1,1,T] → additive bias
        if attention_mask is not None:
            attn_bias = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)
            attn_bias = attn_bias * float("-inf")
        else:
            attn_bias = None
        for layer in self.encoder:
            x = layer(x, attn_bias)
        return x


def _infer_bert_hparams(sd: Dict[str, torch.Tensor]) -> dict:
    # BERT uses bert.embeddings.word_embeddings.weight or just
    # embeddings.word_embeddings.weight
    embed = next(v for k, v in sd.items() if "word_embeddings.weight" in k)
    vocab_size, d_model = embed.shape
    n_layers = _count_layers(sd, r"layer\.(\d+)\.")
    q_weight = next(v for k, v in sd.items() if "attention.self.query.weight" in k)
    n_heads = d_model // (d_model // 12)   # BERT standard: 64 per head
    for nh in [16, 12, 8, 4]:
        if d_model % nh == 0:
            n_heads = nh
            break
    inter = next(v for k, v in sd.items() if "intermediate.dense.weight" in k)
    intermediate = inter.shape[0]
    pos = next((v for k, v in sd.items() if "position_embeddings.weight" in k), None)
    max_seq = pos.shape[0] if pos is not None else 512
    return dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                n_heads=n_heads, intermediate=intermediate, max_seq=max_seq)



# ══════════════════════════════════════════════════════════════════════════════
#  Qwen2-VL / Qwen3-VL family
# ══════════════════════════════════════════════════════════════════════════════
#
#  Architecture overview
#  ─────────────────────
#  ┌─────────────────────────────────────────────────────────┐
#  │  Input image  →  PatchEmbed  →  ViT blocks (window attn)│
#  │                               ↓                         │
#  │  visual.merger MLP  →  vision tokens  (merged to LM dim)│
#  └─────────────────────────────────────────────────────────┘
#                          ↓ interleaved with text tokens
#  ┌─────────────────────────────────────────────────────────┐
#  │  Qwen3 LM decoder (GQA + SwiGLU + RMSNorm + M-RoPE)    │
#  └─────────────────────────────────────────────────────────┘
#
#  Key weight-name prefixes
#  ────────────────────────
#  visual.patch_embed.proj.*          — 2D conv patch embedding
#  visual.blocks.N.norm1/2.*          — ViT LayerNorm
#  visual.blocks.N.attn.qkv.*        — fused QKV (NOT separate like LLaMA)
#  visual.blocks.N.attn.proj.*       — ViT output projection
#  visual.blocks.N.mlp.fc1/fc2.*     — ViT MLP (GELU, NOT SwiGLU)
#  visual.merger.mlp.{0,2,4}.*       — vision→LM dimension merger
#  model.embed_tokens.*               — text token embedding
#  model.layers.N.self_attn.*        — LM attention (same as LLaMA)
#  model.layers.N.mlp.*              — LM MLP (SwiGLU, same as LLaMA)
#  model.norm.*                       — final LM RMSNorm
#  lm_head.*                          — vocabulary projection

class _Qwen3VLPatchEmbed(nn.Module):
    """Project image patches to vision embedding dimension via Conv2d."""
    def __init__(self, in_chans: int, embed_dim: int,
                 patch_h: int = 14, patch_w: int = 14):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=(patch_h, patch_w),
                              stride=(patch_h, patch_w))
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]  →  [B, N_patches, embed_dim]
        x = self.proj(x)                         # [B, D, H/ph, W/pw]
        return x.flatten(2).transpose(1, 2)      # [B, N, D]


class _Qwen3VLViTAttention(nn.Module):
    """Fused QKV self-attention used inside the vision encoder."""
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale    = self.head_dim ** -0.5
        self.qkv      = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj     = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.n_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))          # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v)             # [B, H, N, D]
        out  = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _Qwen3VLViTMLP(nn.Module):
    """Two-layer GELU MLP inside the vision encoder (NOT SwiGLU)."""
    def __init__(self, embed_dim: int, intermediate: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate)
        self.fc2 = nn.Linear(intermediate, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Qwen3VLViTBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, intermediate: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = _Qwen3VLViTAttention(embed_dim, n_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = _Qwen3VLViTMLP(embed_dim, intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _Qwen3VLVisionEncoder(nn.Module):
    """
    ViT-based vision encoder.

    Encodes image patches → sequence of vision tokens.
    Uses standard full-attention (no causal mask, no RoPE — the 2D M-RoPE
    is applied inside the LM decoder when vision + text tokens are merged).
    """
    def __init__(self, in_chans: int, embed_dim: int, depth: int,
                 n_heads: int, intermediate: int,
                 patch_h: int = 14, patch_w: int = 14):
        super().__init__()
        self.patch_embed = _Qwen3VLPatchEmbed(in_chans, embed_dim, patch_h, patch_w)
        self.blocks = nn.ModuleList([
            _Qwen3VLViTBlock(embed_dim, n_heads, intermediate)
            for _ in range(depth)
        ])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(pixel_values)
        for blk in self.blocks:
            x = blk(x)
        return x                 # [B, N_patches, vision_embed_dim]


class _Qwen3VLMerger(nn.Module):
    """
    Project vision tokens from vision_embed_dim to lm_dim.

    Qwen-VL uses a 3-layer MLP (Linear → GELU → Linear → GELU → Linear)
    with indices 0, 2, 4 in the saved weight dict (the odd indices are
    the activation functions, which have no weights).
    """
    def __init__(self, vision_dim: int, lm_dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or lm_dim
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, lm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Qwen3VLModel(nn.Module):
    """
    Reconstructed Qwen2-VL / Qwen3-VL model.

    Holds both the vision encoder and the language decoder.  In the real
    HuggingFace implementation, vision tokens and text tokens are interleaved
    inside the LM forward pass using special image token placeholders.  Here
    we expose the two components separately so they can be called individually
    or combined by the user.

    Parameters
    ----------
    vision_*   : vision encoder hyperparameters
    lm_*       : language model hyperparameters (Qwen3/LLaMA-style)
    merger_*   : vision→LM projection hyperparameters
    """
    def __init__(
        self,
        # Vision encoder
        vision_embed_dim: int,
        vision_depth:     int,
        vision_n_heads:   int,
        vision_inter:     int,
        patch_h:          int,
        patch_w:          int,
        in_chans:         int,
        # Merger
        merger_hidden:    int,
        # LM decoder
        vocab_size:       int,
        d_model:          int,
        n_layers:         int,
        n_heads:          int,
        n_kv_heads:       Optional[int],
        head_dim:         Optional[int],
        intermediate:     int,
        max_seq:          int = 32768,
        rms_eps:          float = 1e-6,
        tie_embeddings:   bool = False,
    ):
        super().__init__()
        self.visual = _Qwen3VLVisionEncoder(
            in_chans=in_chans,
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            n_heads=vision_n_heads,
            intermediate=vision_inter,
            patch_h=patch_h,
            patch_w=patch_w,
        )
        self.visual_merger = _Qwen3VLMerger(
            vision_dim=vision_embed_dim,
            lm_dim=d_model,
            hidden=merger_hidden,
        )
        # Reuse the LLaMA model for the LM decoder — identical architecture
        self.model = LlamaModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            intermediate=intermediate,
            max_seq=max_seq,
            rms_eps=rms_eps,
            tie_embeddings=tie_embeddings,
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode an image into LM-dimension vision tokens.

        pixel_values : [B, C, H, W]
        returns      : [B, N_patches, d_model]  — ready to interleave with text
        """
        vt = self.visual(pixel_values)          # [B, N, vision_embed_dim]
        return self.visual_merger(vt)           # [B, N, d_model]

    def forward(
        self,
        input_ids:       torch.Tensor,
        attention_mask:  Optional[torch.Tensor] = None,
        pixel_values:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Text-only or image+text forward.

        When pixel_values is None: text-only, identical to LlamaModel.forward().

        When pixel_values is provided: vision tokens are prepended to the text
        embedding sequence.  No image-token-placeholder logic is implemented
        here — for that use the HuggingFace Qwen2VLForConditionalGeneration
        class.  This simplified forward() is useful for:
          • Inspecting that weights loaded correctly
          • Sparse-pruning experiments on LM layers only
          • rnn_fold experiments on the LM decoder
        """
        x = self.model.embed_tokens(input_ids)  # [B, T, d_model]

        if pixel_values is not None:
            vision_tokens = self.encode_image(pixel_values)   # [B, N, d_model]
            x = torch.cat([vision_tokens, x], dim=1)
            # Extend attention mask if provided
            if attention_mask is not None:
                vision_ones = torch.ones(
                    attention_mask.shape[0], vision_tokens.shape[1],
                    dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([vision_ones, attention_mask], dim=1)

        for layer in self.model.layers:
            x = layer(x, attention_mask)
        x = self.model.norm(x)
        return self.model.lm_head(x)


def _infer_qwen3vl_hparams(sd: Dict[str, torch.Tensor]) -> dict:
    """Infer Qwen2-VL / Qwen3-VL hyperparameters from weight shapes."""

    # ── Vision encoder ────────────────────────────────────────────────────────
    patch_proj = sd.get("visual.patch_embed.proj.weight")
    if patch_proj is None:
        patch_proj = next(v for k, v in sd.items() if "patch_embed.proj.weight" in k)
    # Conv2d weight: [out_channels, in_channels, kH, kW]
    vision_embed_dim = int(patch_proj.shape[0])
    in_chans         = int(patch_proj.shape[1])
    patch_h          = int(patch_proj.shape[2])
    patch_w          = int(patch_proj.shape[3])

    vision_depth = _count_layers(sd, r"visual\.blocks\.(\d+)\.")

    # QKV weight is [3*embed_dim, embed_dim]
    qkv_w = next(v for k, v in sd.items() if "visual.blocks" in k and "attn.qkv.weight" in k)
    vision_n_heads = max(h for h in [8, 12, 16, 24, 32]
                         if vision_embed_dim % h == 0 and vision_embed_dim // h >= 32)

    # ViT MLP intermediate
    fc1_w = next(v for k, v in sd.items() if "visual.blocks" in k and "mlp.fc1.weight" in k)
    vision_inter = int(fc1_w.shape[0])

    # ── Merger ────────────────────────────────────────────────────────────────
    merger_w0 = next((v for k, v in sd.items()
                      if "visual.merger" in k and ".0.weight" in k), None)
    merger_hidden = int(merger_w0.shape[0]) if merger_w0 is not None else vision_embed_dim * 2

    # ── LM decoder (same as LLaMA) ────────────────────────────────────────────
    embed = sd.get("model.embed_tokens.weight")
    if embed is None:
        embed = next(v for k, v in sd.items() if "embed_tokens.weight" in k)
    vocab_size, d_model = embed.shape

    n_layers = _count_layers(sd, r"model\.layers\.(\d+)\.")

    q_weight = next(v for k, v in sd.items()
                    if "model.layers" in k and "self_attn.q_proj.weight" in k)
    k_weight = next(v for k, v in sd.items()
                    if "model.layers" in k and "self_attn.k_proj.weight" in k)
    q_out, k_out = q_weight.shape[0], k_weight.shape[0]

    head_dim = next(hd for hd in [128, 64, 96, 80, 256, 32]
                    if q_out % hd == 0 and k_out % hd == 0)
    n_heads    = q_out // head_dim
    n_kv_heads = k_out // head_dim

    gate = next(v for k, v in sd.items()
                if "model.layers" in k and "mlp.gate_proj.weight" in k)
    intermediate = int(gate.shape[0])

    tie = not any("lm_head.weight" in k for k in sd)
    # Qwen3 uses eps=1e-6 by default (tighter than LLaMA's 1e-5)
    rms_eps = 1e-6

    return dict(
        vision_embed_dim=vision_embed_dim,
        vision_depth=vision_depth,
        vision_n_heads=vision_n_heads,
        vision_inter=vision_inter,
        patch_h=patch_h,
        patch_w=patch_w,
        in_chans=in_chans,
        merger_hidden=merger_hidden,
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        intermediate=intermediate,
        rms_eps=rms_eps,
        tie_embeddings=tie,
    )


def _remap_qwen3vl_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap HuggingFace Qwen3-VL keys to our module tree.

    HuggingFace Qwen2VLForConditionalGeneration layout:
      model.visual.*        → visual.*
      model.embed_tokens.*  → model.embed_tokens.*
      model.layers.*        → model.layers.*
      model.norm.*          → model.norm.*
      lm_head.*             → model.lm_head.*

    merger is at:
      model.visual.merger.* (HF) → visual_merger.*
    or
      visual.merger.*       (raw) → visual_merger.*

    ViT blocks:
      model.visual.blocks.*  → visual.blocks.*

    merger MLP:
      HF saves merger as visual.merger.mlp.0/2/4 using Sequential indexing.
      We map these to visual_merger.mlp.0/2/4 (our nn.Sequential).
    """
    out = {}
    for k, v in sd.items():
        # Strip outer "model." wrapper that HF adds around the whole thing
        nk = k
        if nk.startswith("model.visual."):
            nk = nk[len("model."):]          # → visual.*
        # visual.merger → visual_merger
        if nk.startswith("visual.merger.mlp."):
            # visual.merger.mlp.0.weight → visual_merger.mlp.0.weight
            nk = "visual_merger.mlp." + nk[len("visual.merger.mlp."):]
        elif nk.startswith("visual.merger."):
            nk = "visual_merger." + nk[len("visual.merger."):]
        # lm_head lives at model.lm_head in our tree
        if nk == "lm_head.weight":
            nk = "model.lm_head.weight"
        out[nk] = v
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Generic fallback
# ══════════════════════════════════════════════════════════════════════════════

class GenericStateModule(nn.Module):
    """
    A plain container for unrecognised weight files.

    Holds the raw state dict as nn.ParameterDict so parameters are registered
    (gradients, .to(device), etc. all work).  forward() raises NotImplementedError
    because we have no architecture graph to execute.

    Inspect the parameters to understand what you have:
        for name, p in model.named_parameters():
            print(name, p.shape)
    """
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        # nn.ParameterDict requires string keys without dots; store as a flat
        # dict via register_buffer so the tensors are tracked without the
        # key-naming restriction
        self._param_names: List[str] = []
        for i, (name, tensor) in enumerate(state_dict.items()):
            self.register_buffer(f"_p{i}", tensor)
            self._param_names.append(name)

    def named_original_params(self):
        for i, name in enumerate(self._param_names):
            yield name, getattr(self, f"_p{i}")

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "GenericStateModule has no architecture — cannot run forward().\n"
            "Use named_original_params() to inspect weights, or open an issue "
            "to request support for this architecture family."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Key remapping
# ══════════════════════════════════════════════════════════════════════════════

def _remap_keys(sd: Dict[str, torch.Tensor], family: str) -> Dict[str, torch.Tensor]:
    """
    Strip common HuggingFace wrapper prefixes so keys match our module layout.

    HuggingFace saves weights as "model.layers.0...." or "transformer.h.0...."
    but our nn.Module trees are rooted directly at the equivalent submodule.
    """
    def _strip(name: str, prefix: str) -> str:
        return name[len(prefix):] if name.startswith(prefix) else name

    if family == "qwen3vl":
        return _remap_qwen3vl_keys(sd)
    if family == "llama":
        return {_strip(k, "model."): v for k, v in sd.items()}
    if family == "gpt2":
        return {_strip(k, "transformer."): v for k, v in sd.items()}
    if family == "bert":
        # Try "bert." prefix first, then "roberta."
        for pfx in ("bert.", "roberta.", "electra.", "deberta."):
            if any(k.startswith(pfx) for k in sd):
                return {_strip(k, pfx): v for k, v in sd.items()}
    return sd


def _load_into(model: nn.Module, sd: Dict[str, torch.Tensor],
               strict: bool = False) -> None:
    """
    Load state dict with informative mismatch reporting.

    strict=False allows partial loads (common when e.g. lm_head is tied and
    not present in the file, or when loading a single shard).
    """
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        print(f"  [warn]  {len(missing)} missing keys  (first 5: {missing[:5]})")
    if unexpected:
        print(f"  [warn]  {len(unexpected)} unexpected keys  (first 5: {unexpected[:5]})")


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_model(
    path: str,
    strict: bool = False,
    device: str = "cpu",
) -> nn.Module:
    """
    Load a .bin or .safetensors weight file and return a runnable nn.Module.

    Parameters
    ----------
    path   : path to a single weight shard, or a directory of shards
    strict : if True, raise on missing/unexpected keys during weight load
    device : 'cpu', 'cuda', 'cuda:0', etc.

    Returns
    -------
    An nn.Module in eval() mode with weights loaded.
    """
    p = Path(path)

    # Directory → sharded load
    if p.is_dir():
        print(f"[reconstruct]  Loading sharded model from {p}")
        sd = load_weights_sharded(str(p))
    else:
        print(f"[reconstruct]  Loading {p.name}")
        sd = load_weights(str(p))

    print(f"[reconstruct]  {len(sd)} tensors loaded")

    family = detect_family(sd)
    print(f"[reconstruct]  Detected family: {family}")

    if family == "qwen3vl":
        hp = _infer_qwen3vl_hparams(sd)
        _print_hparams(hp, family)
        model = Qwen3VLModel(**hp)
        _load_into(model, _remap_keys(sd, "qwen3vl"), strict)

    elif family == "llama":
        hp = _infer_llama_hparams(sd)
        _print_hparams(hp, family)
        model = LlamaModel(**hp)
        _load_into(model, _remap_keys(sd, "llama"), strict)

    elif family == "gpt2":
        hp = _infer_gpt2_hparams(sd)
        _print_hparams(hp, family)
        model = GPT2Model(**hp)
        _load_into(model, _remap_keys(sd, "gpt2"), strict)

    elif family == "bert":
        hp = _infer_bert_hparams(sd)
        _print_hparams(hp, family)
        model = BERTModel(**hp)
        _load_into(model, _remap_keys(sd, "bert"), strict)

    else:
        print("[reconstruct]  Unknown architecture — returning GenericStateModule")
        model = GenericStateModule(sd)

    model = model.to(device)
    model.eval()
    return model


def _print_hparams(hp: dict, family: str) -> None:
    print(f"[reconstruct]  {family} hyperparameters:")
    for k, v in hp.items():
        print(f"               {k:<22} = {v}")


def show_arch(path: str) -> None:
    """Print architecture summary without building the full model."""
    p = Path(path)
    sd = load_weights_sharded(str(p)) if p.is_dir() else load_weights(str(p))
    family = detect_family(sd)
    print(f"\nFamily   : {family}")
    print(f"Tensors  : {len(sd)}")
    print(f"\nParameter summary (name  →  shape):")
    for name, t in list(sd.items())[:40]:
        print(f"  {name:<60}  {tuple(t.shape)}")
    if len(sd) > 40:
        print(f"  … and {len(sd)-40} more")

    if family == "qwen3vl":
        hp = _infer_qwen3vl_hparams(sd)
        print(f"\nInferred hparams:")
        for k, v in hp.items(): print(f"  {k:<24} = {v}")
    elif family == "llama":
        hp = _infer_llama_hparams(sd)
        print(f"\nInferred hparams: {hp}")
    elif family == "gpt2":
        hp = _infer_gpt2_hparams(sd)
        print(f"\nInferred hparams: {hp}")
    elif family == "bert":
        hp = _infer_bert_hparams(sd)
        print(f"\nInferred hparams: {hp}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Reconstruct a runnable nn.Module from a .bin/.safetensors file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input",       help=".bin / .safetensors file or shard directory")
    p.add_argument("--save",      default=None, metavar="PATH",
                   help="Save the reconstructed module with torch.save()")
    p.add_argument("--show-arch", action="store_true",
                   help="Print parameter names and inferred hparams, then exit")
    p.add_argument("--device",    default="cpu",
                   help="Target device  (default: cpu)")
    p.add_argument("--strict",    action="store_true",
                   help="Strict key matching when loading weights")
    args = p.parse_args()

    if args.show_arch:
        show_arch(args.input)
        return

    model = reconstruct_model(args.input, strict=args.strict, device=args.device)
    print(f"\n[reconstruct]  Module type : {type(model).__name__}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[reconstruct]  Parameters  : {n_params:,}")

    if args.save:
        torch.save(model, args.save)
        print(f"[reconstruct]  Saved to    : {args.save}")


if __name__ == "__main__":
    main()
