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

    # Estimate CSR storage: values (f32=4B) + col_indices (i32=4B) + crow (i32=4B)
    # crow has (n_rows + 1) entries
    n_rows      = tensor.shape[0]
    dense_bytes = n_elem * 4
    csr_est     = n_nonzero * 8 + (n_rows + 1) * 4   # vals+cols, then crow
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
    # Threshold-estimation knobs
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

    Each entry in sparse_state_dict is one of:
      {"csr":   <CSR tensor>, "shape": <original torch.Size>}  -- sparse layer
      {"dense": <f32 tensor>, "shape": <torch.Size>}           -- dense layer

    Dense entries are used for: vectors (LayerNorm, biases), low-sparsity
    matrices, and matrices where CSR overhead exceeds savings.
    """
    training_seconds = training_days * 24 * 3600

    # -- Determine pruning threshold -------------------------------------------
    if min_abs_param is None:
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
        print(f"[threshold]  quant format     = {bits_label}  (eps = {eps_b:.4e})")
        print(f"[threshold]  synapse floor    = {floor:.6e}")
        print(f"[threshold]  min_abs_param    = {min_abs_param:.6e}  (2 x floor)")
    else:
        print(f"[threshold]  min_abs_param    = {min_abs_param:.6e}  (user-supplied)")

    print(f"[threshold]  min_sparsity     = {min_sparsity:.0%}  (below -> keep dense)")
    print(f"[threshold]  max_sparse_ratio = {max_sparse_ratio:.1f}x  (above -> keep dense)")
    print(f"[threshold]  vectors (1-D)    = always dense")

    # -- Load ------------------------------------------------------------------
    print(f"\n[load]  {input_path}")
    state_dict = load_state_dict(input_path)
    print(f"        {len(state_dict)} tensor(s) found")

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
            sparse_state[name] = {"dense": param.detach(), "shape": param.shape}
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
            pruned   = t_f32
            n_pruned = 0
        else:
            mask     = t_f32.abs() >= min_abs_param
            pruned   = t_f32 * mask
            n_pruned = int((~mask).sum().item())

        n_nz     = int((pruned != 0).sum().item())
        sparsity = (n_elem - n_nz) / n_elem

        total_elements += n_elem
        total_pruned   += n_pruned

        # Decide format
        reason = _keep_dense_reason(pruned, min_sparsity, max_sparse_ratio)

        if reason is not None:
            stored_bytes = dense_bytes
            sparse_state[name] = {"dense": pruned, "shape": param.shape}
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
                   help="Hard-code pruning threshold (skips auto-calculation)")

    g = p.add_argument_group("threshold estimation (ignored if --min-abs is set)")
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

    sparsify_model(
        input_path=args.input,
        output_path=args.output,
        min_abs_param=args.min_abs,
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