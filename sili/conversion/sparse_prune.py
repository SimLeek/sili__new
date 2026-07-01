#!/usr/bin/env python3
"""
sparse_prune.py
---------------
Prune a PyTorch model (pickle or safetensors) by zeroing parameters below a
minimum absolute value threshold, storing surviving weights as COO->CSR sparse
tensors, and reporting sparsity / memory savings.

A tensor is only stored as sparse if it passes BOTH tests:
  1. sparsity  >= min_sparsity   (default 33%) -- below this, sparse indices
                                  cost more than they save
  2. sparse_MB < max_sparse_ratio * dense_MB  (default 0.9x) -- belt-and-
                                  suspenders: CSR header overhead test

Vectors (1-D tensors, e.g. LayerNorm weights, bias terms) are ALWAYS kept
dense: they are typically <1% sparse, use trivial memory, and CSR adds
overhead with zero benefit for rank-1 data.

Default threshold = 2 * synapse_diminish_floor(), estimating the weight
magnitude below which Adam can no longer change the weight's stored bucket at
the given quantisation bit width.

Usage
-----
  python sparse_prune.py model.safetensors
  python sparse_prune.py model.safetensors --bits 8 -o model_sparse.pt
  python sparse_prune.py --show-floor --lr 1e-5 --days 30
  python sparse_prune.py model.safetensors --rnn-fold --rnn-fold-skip-outputs
  python sparse_prune.py model.safetensors --rnn-all
  python sparse_prune.py model.safetensors --min-sparsity 0.5 --max-sparse-ratio 0.75
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


# ==============================================================================
#  Lazy rnn_fold import -- only used when --rnn-fold / --rnn-all are requested
# ==============================================================================

def _import_rnn_fold():
    """Import rnn_fold.py from the same directory, or exit with a clear error."""
    try:
        import rnn_fold as _rf
        return _rf
    except ImportError:
        sys.exit(
            "rnn_fold.py not found.  Place it in the same directory as "
            "sparse_prune.py or add it to PYTHONPATH."
        )


# ==============================================================================
#  Quantisation epsilon + synapse floor
# ==============================================================================

def quantization_epsilon(bits: int) -> float:
    """
    Return the effective machine epsilon for a given storage bit width.

    Float formats  : standard IEEE definition  2^{-mantissa_bits}
    Integer formats: symmetric-quant step size 1 / (2^{bits-1} - 1)

      32 -> float32  e ~= 1.19e-7
      16 -> float16  e ~= 9.77e-4
       8 -> int8     e ~= 7.87e-3
       4 -> int4     e ~= 1.43e-1
       2 -> int2     e  = 1.00
    """
    if bits == 32:
        return float(torch.finfo(torch.float32).eps)
    elif bits == 16:
        return float(torch.finfo(torch.float16).eps)
    elif bits in (8, 4, 2):
        return 1.0 / ((1 << (bits - 1)) - 1)
    else:
        raise ValueError(
            f"Unsupported bit width {bits}. "
            "Choose from: 32 (float32), 16 (float16), 8 (int8), 4 (int4), 2 (int2)."
        )


def synapse_diminish_floor(
    learning_rate: float = 1e-5,
    typical_input: float = 1.0,
    training_time_seconds: float = 30 * 24 * 3600,
    steps_per_second: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    bits: int = 32,
) -> float:
    """
    Estimate the weight magnitude floor below which Adam can no longer move
    the weight's stored representation at a given quantisation bit width.

    Formula
    -------
        w_floor = |typical_input| * 2 * eps_bits / lr_eff

    where lr_eff = lr * sqrt(1-b2^t) / (1-b1^t)  (bias-corrected LR).

    Interpretation
    --------------
    The floor is the weight magnitude at which the format's quantisation step
    size exceeds what the effective learning rate can produce as a gradient
    signal from typical inputs.  Above this floor a weight can be updated;
    below it the weight is frozen by the format's resolution.

    At float32 (eps ~= 1.2e-7) and lr=1e-5 the floor is small.
    At float16 (eps ~= 1e-3) or INT8 (eps ~= 8e-3) the floor is meaningfully
    large -- consistent with the research finding that ~80% of transformer
    neurons (not individual synapses) are functionally dead after training.
    At INT4/INT2 the floor dominates and nearly all small weights are frozen.

    Note: Adam's numerical-stability epsilon is intentionally absent.
    What matters is the FORMAT's resolution vs the LR, not Adam's internals.
    """
    n_steps = max(1, int(training_time_seconds * steps_per_second))
    bc1     = 1.0 - beta1 ** n_steps
    bc2     = 1.0 - beta2 ** n_steps
    lr_eff  = learning_rate * math.sqrt(bc2) / bc1

    eps_bits = quantization_epsilon(bits)
    return abs(typical_input) * 2.0 * eps_bits / lr_eff


def default_min_abs_param(
    learning_rate: float = 1e-5,
    typical_input: float = 1.0,
    training_time_seconds: float = 30 * 24 * 3600,
    steps_per_second: float = 1.0,
    bits: int = 32,
) -> float:
    """Return 2 * synapse_diminish_floor() as the default pruning threshold."""
    return 2.0 * synapse_diminish_floor(
        learning_rate=learning_rate,
        typical_input=typical_input,
        training_time_seconds=training_time_seconds,
        steps_per_second=steps_per_second,
        bits=bits,
    )


def calibrate_min_abs_param(
    state_dict: Dict[str, torch.Tensor],
    target_sparsity: float = 0.5,
) -> float:
    """
    Derive min_abs_param directly from the ACTUAL loaded weight-magnitude
    distribution, rather than from an assumed training-dynamics formula
    (synapse_diminish_floor).  THIS IS THE DEFAULT MODE (see --calibrate /
    --no-calibrate on the CLI, and the `calibrate` parameter on
    sparsify_model()).

    Rationale
    ---------
    synapse_diminish_floor() estimates where Adam's update resolution floor
    sits GIVEN a specific (lr, training_time, bits) combination -- it has no
    knowledge of the actual weight scale in the file being pruned.  When the
    assumed training conditions don't match the real checkpoint (different
    LR, different init scale, a randomly-initialized model, etc.) the
    computed threshold can land anywhere relative to the real weight
    distribution, including deep inside its bulk -- causing catastrophic
    over-pruning.  Empirically: on an untrained N(0, 0.02) toy model, the
    formula's default settings pruned 98.9% of weights against an injected
    ground truth of ~76% intended; the SAME threshold value, applied to a
    population where it happened to sit outside the bulk, reproduced ground
    truth within a few points.  The formula isn't wrong, it's just blind to
    what it's actually being pointed at.

    calibrate_min_abs_param() asks a direct, scale-invariant question
    instead: "what |weight| value is the target_sparsity-th percentile of
    THIS model's actual weight magnitudes?" -- guaranteeing the resulting
    sparsity lands close to target_sparsity regardless of the weights'
    scale.  This does not claim to find the "true" dead-weight boundary --
    no conversion-time method can, without knowing which weights actually
    matter to this specific model's function.  It trades that unattainable
    precision for a robust, predictable, hard-to-catastrophically-misfire
    memory target:  sparsification here doesn't need to be good, it just
    needs to get large models running in limited system memory without
    destroying them.  Real training-time synaptogenesis/pruning (importance-
    based growth/pruning during actual training) is expected to do the
    precision work this conversion-time step deliberately does not attempt.

    Parameters
    ----------
    state_dict      : flat {name: tensor} state dict (pre-pruning).
    target_sparsity : fraction of eligible (2-D) weight magnitude to target
                      as "below threshold".  Default 0.5 -- conservative,
                      unlikely to destroy the model; raise for more
                      aggressive memory savings, lower to preserve more
                      signal.  This is the single knob that matters for
                      "does this fit in my RAM budget" -- see also
                      calibrate_for_memory_budget() for a budget-first framing
                      of the same idea.

    Returns
    -------
    min_abs_param : float, the calibrated threshold.
    """
    mags = []
    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor) or t.ndim != 2:
            continue    # only 2-D matrices are threshold-eligible, matching
                        # _keep_dense_reason's own ndim==2 requirement --
                        # vectors and higher-rank tensors are always dense
                        # regardless of threshold, so their magnitudes
                        # shouldn't influence the percentile calculation.
        mags.append(t.detach().float().abs().flatten())

    if not mags:
        return 0.0    # nothing eligible -- threshold is moot either way

    all_mags = torch.cat(mags)
    k = max(1, min(all_mags.numel(),
                   int(round(target_sparsity * all_mags.numel()))))
    # kthvalue is exact and, unlike torch.quantile's interpolation path,
    # doesn't require materializing a second full-precision sorted copy --
    # matters for real multi-billion-parameter checkpoints.
    threshold = torch.kthvalue(all_mags, k).values.item()
    return threshold


def calibrate_for_memory_budget(
    state_dict: Dict[str, torch.Tensor],
    target_bytes: int,
    min_sparsity: float = 0.33,
    max_sparse_ratio: float = 0.9,
    max_iterations: int = 6,
) -> float:
    """
    Budget-first variant of calibrate_min_abs_param(): instead of specifying
    a target sparsity fraction directly, specify how many bytes the result
    must fit in, and this searches for a threshold that gets there.

    Because a higher threshold ALWAYS prunes at least as much as a lower one
    (the mask |w| >= threshold shrinks monotonically as threshold grows),
    stored bytes is a monotonically non-increasing function of the
    percentile-derived threshold -- so a small number of bisection steps on
    the TARGET SPARSITY (not on min_abs_param directly, since the
    sparsity-to-bytes relationship has per-layer CSR overhead and
    min_sparsity/max_sparse_ratio dense-fallback nonlinearities baked in)
    converges quickly.  Each iteration's cost is one pass through
    _keep_dense_reason-equivalent bookkeeping (cheap; no CSR tensors are
    actually constructed during the search, only sizes are estimated).

    Parameters
    ----------
    state_dict       : flat {name: tensor} state dict (pre-pruning).
    target_bytes     : maximum acceptable stored size, in bytes.
    min_sparsity      : same meaning as sparsify_model()'s parameter --
                        included here so the byte ESTIMATE during search
                        matches what sparsify_model() will actually do.
    max_sparse_ratio  : same meaning as sparsify_model()'s parameter.
    max_iterations    : bisection step cap (6 steps narrows target_sparsity
                        to within ~1.6% of its converged value -- plenty
                        given this is a "don't destroy the model" safety
                        target, not a precision instrument).

    Returns
    -------
    min_abs_param : float, threshold estimated to land at-or-under target_bytes.
    """
    def _estimate_stored_bytes(threshold: float) -> int:
        total = 0
        for name, t in state_dict.items():
            if not isinstance(t, torch.Tensor) or t.numel() == 0 or t.ndim == 0:
                total += max(t.numel() if isinstance(t, torch.Tensor) else 1, 1) * 4
                continue
            if t.ndim != 2:
                total += t.numel() * 4
                continue
            n_elem = t.numel()
            n_nz   = int((t.detach().float().abs() >= threshold).sum().item())
            sparsity = (n_elem - n_nz) / n_elem
            dense_bytes = n_elem * 4
            if sparsity < min_sparsity:
                total += dense_bytes
                continue
            csr_est = n_nz * 12 + (t.shape[0] + 1) * 8   # matches _keep_dense_reason's
                                                          # corrected int64-index formula
            if csr_est > max_sparse_ratio * dense_bytes:
                total += dense_bytes
            else:
                total += csr_est
        return total

    lo, hi = 0.0, 0.999   # target_sparsity search range
    threshold = calibrate_min_abs_param(state_dict, target_sparsity=0.5)

    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        threshold = calibrate_min_abs_param(state_dict, target_sparsity=mid)
        est_bytes = _estimate_stored_bytes(threshold)
        if est_bytes > target_bytes:
            lo = mid     # need MORE sparsity -> raise the target_sparsity floor
        else:
            hi = mid     # fits with room to spare -> can afford less sparsity

    # Final threshold at the tightest still-fitting point found (hi side,
    # since that's where the last "fits" observation came from).
    return calibrate_min_abs_param(state_dict, target_sparsity=hi)


# ==============================================================================
#  Model loading
# ==============================================================================

def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """
    Load a flat {name: tensor} state dict from a .bin, .pt, .pth,
    .safetensors file, or a directory of shards.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {path}")

    if p.is_dir():
        merged: Dict[str, torch.Tensor] = {}
        shards = sorted(
            list(p.glob("model-*-of-*.safetensors")) +
            list(p.glob("pytorch_model-*-of-*.bin")) +
            list(p.glob("model.safetensors")) +
            list(p.glob("pytorch_model.bin"))
        )
        if not shards:
            raise FileNotFoundError(f"No weight shards found in {path}")
        for shard in shards:
            print(f"  [load shard]  {shard.name}")
            merged.update(load_state_dict(str(shard)))
        return merged

    if p.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError:
            sys.exit("safetensors not installed.  Run: pip install safetensors")
        return dict(load_file(str(p)))

    obj = torch.load(str(p), map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        return obj.get("state_dict", obj)
    if hasattr(obj, "state_dict"):
        return dict(obj.state_dict())
    raise ValueError(f"Cannot interpret loaded object of type {type(obj)}")


# ==============================================================================
#  Sparse-vs-dense decision
# ==============================================================================

def _keep_dense_reason(
    tensor: torch.Tensor,
    min_sparsity: float,
    max_sparse_ratio: float,
) -> Optional[str]:
    """
    Return a reason string if this tensor should stay dense, or None if it
    should be stored as CSR.

    Rules (applied in order):
      1. Scalar or empty              -> dense
      2. Vector (ndim == 1)           -> dense  (LayerNorm weights, biases,
                                                  positional scales, etc.)
      3. Non-matrix (ndim > 2)        -> dense  (conv kernels, patch embeddings,
                                                  3-D position tensors, etc.)
                                                 The synapse-floor formula is
                                                 derived for linear layers only;
                                                 conv weights have different
                                                 gradient dynamics and are often
                                                 legitimately small by design.
      4. sparsity < min_sparsity      -> dense  (index overhead > savings)
      5. estimated csr > ratio*dense  -> dense  (CSR overhead test)
    """
    if tensor.numel() == 0 or tensor.ndim == 0:
        return "scalar/empty"

    if tensor.ndim == 1:
        return "vector (1-D)"

    if tensor.ndim > 2:
        return f"non-matrix conv/embed ({tensor.ndim}-D)"

    n_elem    = tensor.numel()
    n_nonzero = int((tensor != 0).sum().item())
    sparsity  = (n_elem - n_nonzero) / n_elem

    if sparsity < min_sparsity:
        return f"sparsity {sparsity:.1%} < {min_sparsity:.0%}"

    # Estimate CSR storage: values (f32=4B) + col_indices (int64=8B) + crow (int64=8B).
    # torch.to_sparse_csr() uses int64 indices by default (verified: csr.col_indices().dtype
    # == torch.int64) -- NOT int32 as an earlier version of this comment assumed. That
    # mismatch made this overhead check ~1.5x too optimistic, silently letting some
    # layers go "sparse" that actually end up LARGER than dense once really encoded.
    # crow has (n_rows + 1) entries.
    n_rows      = tensor.shape[0]
    dense_bytes = n_elem * 4
    csr_est     = n_nonzero * 12 + (n_rows + 1) * 8   # (val+col) per nnz, then crow
    ratio       = csr_est / dense_bytes if dense_bytes > 0 else 1.0

    if ratio > max_sparse_ratio:
        return f"csr/dense {ratio:.2f} > {max_sparse_ratio:.1f}"

    return None   # go sparse


# ==============================================================================
#  CSR helpers (also imported by rnn_fold.py and sparse_runtime.py)
# ==============================================================================

def to_sparse_csr(
    tensor: torch.Tensor,
    min_abs: float,
) -> Tuple[torch.Tensor, torch.Size]:
    """
    Zero elements below min_abs, then convert to 2-D CSR sparse format.
    Returns (csr_tensor, original_shape).
    """
    original_shape = tensor.shape
    t = tensor.detach().float()

    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim > 2:
        t = t.reshape(t.shape[0], -1)

    pruned = t * (t.abs() >= min_abs)
    csr    = pruned.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
    return csr, original_shape


def csr_bytes(csr: torch.Tensor) -> int:
    """Storage footprint of a CSR tensor in bytes."""
    return (
        csr.values().nbytes
        + csr.col_indices().nbytes
        + csr.crow_indices().nbytes
    )


# ==============================================================================
#  Main pipeline
# ==============================================================================

def sparsify_model(
    input_path: str,
    output_path: Optional[str] = None,
    min_abs_param: Optional[float] = None,
    # Calibration -- data-driven, DEFAULT mode (see calibrate_min_abs_param).
    calibrate: bool = True,
    target_sparsity: float = 0.5,
    memory_budget_bytes: Optional[int] = None,
    # Threshold-estimation knobs (legacy mode -- used only if calibrate=False
    # and min_abs_param is not given; kept for anyone who wants the old
    # training-dynamics-based estimate specifically).
    learning_rate: float = 1e-5,
    typical_input: float = 1.0,
    training_days: float = 30.0,
    steps_per_second: float = 1.0,
    bits: int = 32,
    # Sparse-vs-dense thresholds
    min_sparsity: float = 0.33,
    max_sparse_ratio: float = 0.9,
    # RNN fold options
    rnn_fold: bool = False,
    rnn_fold_min_group: int = 2,
    rnn_fold_band_width: Optional[int] = None,
    rnn_fold_skip_outputs: bool = False,
    rnn_all: bool = False,
) -> None:
    """
    Prune + sparsify a model, outputting a mixed sparse/dense payload.

    Threshold selection priority (highest to lowest):
      1. min_abs_param, if explicitly given -- always wins, no calibration run.
      2. calibrate=True (DEFAULT) -- data-driven percentile threshold from
         calibrate_min_abs_param(), or calibrate_for_memory_budget() if
         memory_budget_bytes is given.  See calibrate_min_abs_param()'s
         docstring for why this replaced the training-dynamics formula as
         the default: the old formula doesn't look at the actual weights
         being pruned and can silently catastrophically over-prune when its
         assumed training conditions don't match the real checkpoint.
      3. calibrate=False -- legacy synapse_diminish_floor()-based estimate
         (requires learning_rate/typical_input/training_days/steps_per_second
         to actually reflect the model's real training history to be
         meaningful).

    Regardless of which mode picked the threshold, a post-hoc check warns
    loudly if the resulting overall sparsity exceeds 95% -- a strong signal
    the threshold landed somewhere it shouldn't have, whichever mode chose it.

    Each entry in sparse_state_dict is one of:
      {"csr": <CSR tensor>, "shape": <original torch.Size>}  -- sparse layer
      {"raw": <f32 tensor>, "shape": <torch.Size>}           -- dense layer

    Dense entries are used for: vectors (LayerNorm, biases), low-sparsity
    matrices, and matrices where CSR overhead exceeds savings.

    NOTE: the dense-entry key is "raw", not "dense" -- this matches the key
    BOTH rnn_fold.py and sparse_runtime.py already expect (verified: 9 call
    sites across the two files consistently read entry.get("raw"); an
    earlier version of this function used "dense", which meant every
    LayerNorm weight, embedding table, and bias was silently dropped the
    moment a payload crossed from here into either downstream file).
    """
    # -- Load FIRST: calibration needs to see the actual weights ---------------
    print(f"\n[load]  {input_path}")
    state_dict = load_state_dict(input_path)
    print(f"        {len(state_dict)} tensor(s) found")

    # -- Determine pruning threshold ---------------------------------------------
    if min_abs_param is not None:
        print(f"[threshold]  mode             = user-supplied")
        print(f"[threshold]  min_abs_param    = {min_abs_param:.6e}")
    elif calibrate:
        if memory_budget_bytes is not None:
            min_abs_param = calibrate_for_memory_budget(
                state_dict, memory_budget_bytes,
                min_sparsity=min_sparsity, max_sparse_ratio=max_sparse_ratio)
            print(f"[threshold]  mode             = calibrated, memory-budget-targeted (default)")
            print(f"[threshold]  memory_budget    = {memory_budget_bytes/1e6:.2f} MB")
        else:
            min_abs_param = calibrate_min_abs_param(state_dict, target_sparsity)
            print(f"[threshold]  mode             = calibrated, data-driven (default)")
            print(f"[threshold]  target_sparsity  = {target_sparsity:.0%}")
        print(f"[threshold]  min_abs_param    = {min_abs_param:.6e}  "
              f"(percentile of |weight| across this model's own eligible tensors)")
    else:
        training_seconds = training_days * 24 * 3600
        floor = synapse_diminish_floor(
            learning_rate=learning_rate,
            typical_input=typical_input,
            training_time_seconds=training_seconds,
            steps_per_second=steps_per_second,
            bits=bits,
        )
        min_abs_param = 2.0 * floor
        eps_b      = quantization_epsilon(bits)
        bits_label = {32:"float32", 16:"float16", 8:"int8", 4:"int4", 2:"int2"}[bits]
        print(f"[threshold]  mode             = training-dynamics formula (legacy, --no-calibrate)")
        print(f"[threshold]  quant format     = {bits_label}  (eps = {eps_b:.4e})")
        print(f"[threshold]  synapse floor    = {floor:.6e}")
        print(f"[threshold]  min_abs_param    = {min_abs_param:.6e}  (2 x floor)")

    print(f"[threshold]  min_sparsity     = {min_sparsity:.0%}  (below -> keep dense)")
    print(f"[threshold]  max_sparse_ratio = {max_sparse_ratio:.1f}x  (above -> keep dense)")
    print(f"[threshold]  vectors (1-D)    = always dense")

    # -- Process each tensor ---------------------------------------------------
    sparse_state: Dict[str, dict] = {}

    total_dense_bytes  = 0
    total_stored_bytes = 0
    total_elements     = 0
    total_pruned       = 0
    n_sparse           = 0
    n_dense            = 0

    CW, CS = 56, 20
    hdr  = (f"  {'Layer':<{CW}} {'Shape':<{CS}}"
            f" {'Sparsity':>9} {'Dense MB':>9} {'Stored MB':>9} {'Fmt':>7}")
    sep  = "-" * len(hdr)
    print(f"\n{hdr}\n{sep}")

    for name, param in state_dict.items():
        n_elem      = param.numel()
        dense_bytes = max(n_elem, 1) * 4

        # Scalar / empty -- keep dense, nothing to sparsify
        if n_elem == 0 or param.ndim == 0:
            sparse_state[name] = {"raw": param.detach(), "shape": param.shape}
            total_dense_bytes  += dense_bytes
            total_stored_bytes += dense_bytes
            n_dense            += 1
            continue

        t_f32 = param.detach().float()

        # Apply threshold to 2-D weight matrices only.
        # Vectors (1-D) and higher-rank tensors (conv kernels, patch embeddings,
        # position tensors with ndim > 2) are never zeroed: the synapse-floor
        # formula is derived for linear layers and produces meaningless results
        # for convolutional weights that are legitimately small by design.
        if t_f32.ndim != 2:
            pruned = t_f32
        else:
            mask   = t_f32.abs() >= min_abs_param
            pruned = t_f32 * mask

        n_nz     = int((pruned != 0).sum().item())
        sparsity = (n_elem - n_nz) / n_elem
        # n_pruned reflects ACTUAL final zero count (n_elem - n_nz), not just
        # "entries newly zeroed by this threshold comparison" -- these diverge
        # when the input already contains exact zeros (e.g. calibrated
        # threshold=0.0 on data with pre-existing zeros: mask is all-True
        # since |w|>=0 always holds, so (~mask).sum() would read 0 even
        # though the tensor -- and the correctly-saved output CSR -- may
        # already be mostly zero). Using n_elem - n_nz keeps the summary
        # accurate regardless of what the input already looked like.
        n_pruned = n_elem - n_nz

        total_elements += n_elem
        total_pruned   += n_pruned

        # Decide format
        reason = _keep_dense_reason(pruned, min_sparsity, max_sparse_ratio)

        if reason is not None:
            stored_bytes = dense_bytes
            sparse_state[name] = {"raw": pruned, "shape": param.shape}
            fmt = "dense"
            n_dense += 1
        else:
            t2d = pruned.reshape(pruned.shape[0], -1) if pruned.ndim > 2 else pruned
            csr          = t2d.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
            stored_bytes = csr_bytes(csr)
            sparse_state[name] = {"csr": csr, "shape": param.shape}
            fmt = "sparse"
            n_sparse += 1

        total_dense_bytes  += dense_bytes
        total_stored_bytes += stored_bytes

        note = f"  [{reason}]" if reason else ""
        print(
            f"  {name:<{CW}} {str(tuple(param.shape)):<{CS}}"
            f" {sparsity:>8.1%}"
            f" {dense_bytes/1e6:>8.3f}"
            f"  {stored_bytes/1e6:>8.3f}"
            f"  {fmt:>7}{note}"
        )

    # -- Summary ---------------------------------------------------------------
    overall_sp = total_pruned / total_elements if total_elements else 0.0
    ratio      = total_dense_bytes / total_stored_bytes if total_stored_bytes else float("inf")

    print(sep)
    print(f"\n[summary]")
    print(f"  total parameters   : {total_elements:>14,}")
    print(f"  pruned (|w|<thr)   : {total_pruned:>14,}  ({overall_sp:.2%})")
    print(f"  layers -> sparse   : {n_sparse:>14,}")
    print(f"  layers -> dense    : {n_dense:>14,}  (vectors / low-sparsity / high-overhead)")
    print(f"  dense  model size  : {total_dense_bytes/1e6:>10.2f} MB")
    print(f"  stored model size  : {total_stored_bytes/1e6:>10.2f} MB")
    print(f"  compression ratio  : {ratio:>10.2f}x")

    # -- Safety net: warn loudly regardless of WHICH mode picked the threshold --
    # Catches: a poorly-chosen --min-abs override, --no-calibrate with
    # mismatched training assumptions, or (in principle) a calibration bug --
    # this check doesn't care how the threshold was chosen, only what happened.
    CATASTROPHE_THRESHOLD = 0.95
    if overall_sp > CATASTROPHE_THRESHOLD:
        print()
        print("!" * 70)
        print(f"!  WARNING: {overall_sp:.1%} of weights were pruned -- this is unusually high")
        print(f"!  and may indicate the threshold is miscalibrated for this specific")
        print(f"!  checkpoint's actual weight scale, not a genuinely well-trained,")
        print(f"!  highly-sparse model. A high compression ratio here can mean")
        print(f"!  catastrophic data loss rather than efficient compression.")
        print(f"!  Consider: lower --target-sparsity, or pass --min-abs explicitly")
        print(f"!  after inspecting this model's real |weight| distribution.")
        print("!" * 70)

    # -- Output path -----------------------------------------------------------
    if output_path is None:
        base   = Path(input_path)
        stem   = base.name if base.is_dir() else base.stem
        parent = base if base.is_dir() else base.parent
        output_path = str(parent / f"{stem}_sparse.pt")

    # -- Save base payload -----------------------------------------------------
    payload = {
        "sparse_state_dict": sparse_state,
        "min_abs_param":     min_abs_param,
        "min_sparsity":      min_sparsity,
        "max_sparse_ratio":  max_sparse_ratio,
        "meta": {
            "source_file":      str(input_path),
            "total_elements":   total_elements,
            "total_pruned":     total_pruned,
            "overall_sparsity": overall_sp,
            "dense_bytes":      total_dense_bytes,
            "stored_bytes":     total_stored_bytes,
            "compression":      ratio,
            "n_sparse_layers":  n_sparse,
            "n_dense_layers":   n_dense,
        },
    }
    torch.save(payload, output_path)
    print(f"\n[saved]  {output_path}  ({os.path.getsize(output_path)/1e6:.2f} MB)")

    # -- Optional RNN fold / rnn-all -------------------------------------------
    if rnn_fold or rnn_all:
        rf      = _import_rnn_fold()
        working = payload
        parent  = Path(output_path).parent
        sbase   = Path(output_path).stem.replace("_sparse", "")

        if rnn_fold:
            print("\n" + "=" * 60)
            print("[rnn_fold]  Starting fold stage ...")
            working = rf.fold_sparse_payload(
                working,
                min_group_size=rnn_fold_min_group,
                band_half_width_override=rnn_fold_band_width,
                skip_connection_outputs=rnn_fold_skip_outputs,
            )
            fp = str(parent / f"{sbase}_sparse_folded.pt")
            torch.save(working, fp)
            print(f"[saved]  {fp}  ({os.path.getsize(fp)/1e6:.2f} MB)")

        if rnn_all:
            print("\n" + "=" * 60)
            print("[rnn_all]  Starting per-layer recurrent extension ...")
            working = rf.apply_rnn_all_to_payload(working)
            sfx     = "_sparse_folded_rnnall.pt" if rnn_fold else "_sparse_rnnall.pt"
            rp      = str(parent / f"{sbase}{sfx}")
            torch.save(working, rp)
            print(f"[saved]  {rp}  ({os.path.getsize(rp)/1e6:.2f} MB)")


# ==============================================================================
#  CLI
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prune & sparsify a PyTorch model to mixed sparse/dense format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", nargs="?",
                   help="Model file (.pt/.safetensors) or shard directory")
    p.add_argument("-o", "--output", default=None,
                   help="Output .pt path  (default: <input>_sparse.pt)")
    p.add_argument("--min-abs", type=float, default=None, metavar="F",
                   help="Hard-code pruning threshold (skips calibration entirely)")

    c = p.add_argument_group("calibration (DEFAULT mode -- data-driven, ignored if --min-abs is set)")
    c.add_argument("--calibrate", dest="calibrate", action="store_true", default=True,
                   help="Derive threshold from this model's own weight distribution "
                        "(default: on).  See calibrate_min_abs_param() docstring.")
    c.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                   help="Disable calibration; fall back to the legacy "
                        "training-dynamics threshold formula (see --lr etc. below). "
                        "Only meaningful if those flags actually reflect this "
                        "checkpoint's real training history.")
    c.add_argument("--target-sparsity", type=float, default=0.5, metavar="F",
                   help="Calibration target: fraction of eligible weight magnitude "
                        "to prune (default: 0.5). Conservative default -- raise for "
                        "more memory savings, lower to preserve more signal.")
    c.add_argument("--memory-budget-mb", type=float, default=None, metavar="F",
                   help="If set, calibrate directly against a target stored size "
                        "(MB) instead of --target-sparsity -- answers 'does this "
                        "fit in my RAM budget' directly.")
    c.add_argument("--show-calibration", action="store_true",
                   help="Load the model, print the calibrated threshold and "
                        "resulting sparsity estimate, then exit without saving.")

    g = p.add_argument_group("legacy threshold estimation (only used with --no-calibrate)")
    g.add_argument("--lr",            type=float, default=1e-5,  metavar="F",
                   help="Adam learning rate              (default: 1e-5)")
    g.add_argument("--input-scale",   type=float, default=1.0,   metavar="F",
                   help="Typical activation magnitude    (default: 1.0)")
    g.add_argument("--days",          type=float, default=30.0,  metavar="F",
                   help="Training duration in days       (default: 30)")
    g.add_argument("--steps-per-sec", type=float, default=1.0,   metavar="F",
                   help="Optimizer steps / second        (default: 1.0)")
    g.add_argument("--bits",          type=int,   default=32,
                   choices=[32, 16, 8, 4, 2], metavar="N",
                   help="Weight bit width 32/16/8/4/2    (default: 32)")

    d = p.add_argument_group("sparse-vs-dense thresholds")
    d.add_argument("--min-sparsity",    type=float, default=0.33, metavar="F",
                   help="Min fraction of zeros to store sparse (default: 0.33). "
                        "Below this, keep dense.")
    d.add_argument("--max-sparse-ratio", type=float, default=0.9, metavar="F",
                   help="Max csr/dense byte ratio before fallback to dense "
                        "(default: 0.9).  Vectors (1-D) are always dense.")

    p.add_argument("--show-floor", action="store_true",
                   help="Print synapse floors for all bit widths and exit")

    r = p.add_argument_group("rnn fold (applied after sparsification)")
    r.add_argument("--rnn-fold",              action="store_true",
                   help="Fold repeated blocks into stacked CSR matrices")
    r.add_argument("--rnn-fold-skip-outputs", action="store_true",
                   help="Average per-fold-step outputs (pyramidal pooling)")
    r.add_argument("--rnn-fold-min-group",    type=int, default=2, metavar="N",
                   help="Min identical consecutive blocks to fold (default: 2)")
    r.add_argument("--rnn-fold-band-width",   type=int, default=None, metavar="N",
                   help="Override attention band half-width (default: infer)")
    r.add_argument("--rnn-all",               action="store_true",
                   help="Extend every eligible layer to a per-call RNN cell")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.show_floor:
        training_seconds = args.days * 24 * 3600
        print(f"Training profile: lr={args.lr:.0e}  input={args.input_scale}  "
              f"days={args.days:.1f}  steps/sec={args.steps_per_sec}")
        print()
        rows = [(32,"float32"), (16,"float16"), (8,"int8   "), (4,"int4   "), (2,"int2   ")]
        hdr  = f"  {'Format':<10} {'eps_format':>12} {'floor':>14} {'threshold(2x)':>16}  notes"
        print(hdr)
        print("-" * len(hdr))
        for b, label in rows:
            eps_b = quantization_epsilon(b)
            floor = synapse_diminish_floor(
                learning_rate=args.lr,
                typical_input=args.input_scale,
                training_time_seconds=training_seconds,
                steps_per_second=args.steps_per_sec,
                bits=b,
            )
            thr  = 2.0 * floor
            note = ("  <- most weights ~= 0" if b == 2
                    else "  <- very high floor"  if b <= 4
                    else "  <- fp16 floor ~ INT8" if b == 16
                    else "")
            print(f"  {label:<10} {eps_b:>12.4e} {floor:>14.4e} {thr:>16.4e}{note}")
        print()
        print("Weights below 'floor' are in a region where the format's quantisation")
        print("step is too coarse relative to lr_eff to change the weight's bucket.")
        return

    if args.input is None:
        build_parser().print_help()
        sys.exit(1)

    if args.show_calibration:
        print(f"[load]  {args.input}")
        state_dict = load_state_dict(args.input)
        print(f"        {len(state_dict)} tensor(s) found\n")
        if args.memory_budget_mb:
            budget_bytes = int(args.memory_budget_mb * 1e6)
            thr = calibrate_for_memory_budget(state_dict, budget_bytes)
            print(f"memory_budget_mb : {args.memory_budget_mb}")
        else:
            thr = calibrate_min_abs_param(state_dict, args.target_sparsity)
            print(f"target_sparsity  : {args.target_sparsity:.0%}")
        print(f"min_abs_param    : {thr:.6e}")
        n_elig = sum(t.numel() for t in state_dict.values()
                    if isinstance(t, torch.Tensor) and t.ndim == 2)
        n_below = sum(int((t.detach().float().abs() < thr).sum().item())
                     for t in state_dict.values()
                     if isinstance(t, torch.Tensor) and t.ndim == 2)
        print(f"eligible params  : {n_elig:,}")
        print(f"below threshold  : {n_below:,}  ({n_below/n_elig:.2%})" if n_elig else "")
        return

    sparsify_model(
        input_path=args.input,
        output_path=args.output,
        min_abs_param=args.min_abs,
        calibrate=args.calibrate,
        target_sparsity=args.target_sparsity,
        memory_budget_bytes=(int(args.memory_budget_mb * 1e6)
                             if args.memory_budget_mb else None),
        learning_rate=args.lr,
        typical_input=args.input_scale,
        training_days=args.days,
        steps_per_second=args.steps_per_sec,
        bits=args.bits,
        min_sparsity=args.min_sparsity,
        max_sparse_ratio=args.max_sparse_ratio,
        rnn_fold=args.rnn_fold,
        rnn_fold_min_group=args.rnn_fold_min_group,
        rnn_fold_band_width=args.rnn_fold_band_width,
        rnn_fold_skip_outputs=args.rnn_fold_skip_outputs,
        rnn_all=args.rnn_all,
    )


if __name__ == "__main__":
    main()