"""Layer-by-layer model conversion for RAM-limited boxes: two-phase
streaming (per-tensor sparsify, then per-suffix fold) so peak memory
never holds a full state dict. See docs/research/streaming_prune.rst
for the design rationale (streaming_prune.two_phase_architecture), the
resume-fsck design (streaming_prune.resume_fsck_rationale), and why
conv-kernel tensors stay dense (streaming_prune.conv_kernel_dense_rationale).
"""

from __future__ import annotations

import json
import os
import re

import torch

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _tensor_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, "tensors", _SAFE_NAME_RE.sub("_", name) + ".pt")


def _tensor_file_ok(path: str) -> bool:
    """fsck for a single tensor file -- existence alone is not enough.
    See docs/research/streaming_prune.rst:streaming_prune.resume_fsck_rationale.
    """
    if not os.path.exists(path):
        return False
    try:
        torch.load(path, weights_only=False)
        return True
    except Exception:
        return False


def _iter_shards(model_dir: str):
    """Yield (shard_path, [tensor_names]) for single-file or sharded models."""
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        by_shard: dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            by_shard.setdefault(shard, []).append(name)
        for shard, names in sorted(by_shard.items()):
            yield os.path.join(model_dir, shard), sorted(names)
    else:
        single = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(single):
            cands = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
            if len(cands) != 1:
                raise FileNotFoundError(f"No index.json and no unique .safetensors in {model_dir}")
            single = os.path.join(model_dir, cands[0])
        yield single, None  # None -> enumerate keys from the file itself


def streaming_sparsify(model_dir: str, out_dir: str, threshold: float | None = None, verbose: bool = True) -> dict:
    """
    Phase 1: per-tensor prune -> CSR -> disk. Never holds more than one
    tensor in memory. Resumable: names already in manifest.json are skipped.

    Per-tensor handling:
      ndim >= 2 : abs-threshold prune, reshape (shape[0], -1) if ndim > 2
                  (CSR is 2-D only; orig_shape recorded for reconstruction),
                  bf16/fp16 -> fp32, saved as sparse-CSR .pt
      ndim <= 1 : saved raw dense (norms are never meaningfully sparse)

    Returns the manifest dict:
      {name: {orig_shape, csr_shape|None, nnz, numel, layout, dtype}}
    """
    from safetensors import safe_open

    if threshold is None:
        from sili.conversion.sparse_prune import default_min_abs_param

        threshold = default_min_abs_param()

    os.makedirs(os.path.join(out_dir, "tensors"), exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest: dict = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if verbose:
            print(f"[streaming]  resuming: {len(manifest)} tensors already done")

    def flush_manifest():
        tmp = manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f)
        os.replace(tmp, manifest_path)  # atomic on POSIX

    n_done = 0
    for shard_path, names in _iter_shards(model_dir):
        with safe_open(shard_path, framework="pt") as f:
            keys = names if names is not None else sorted(f.keys())
            for name in keys:
                if name in manifest and _tensor_file_ok(_tensor_path(out_dir, name)):
                    continue
                t = f.get_tensor(name)  # ONE tensor in RAM
                orig_shape = list(t.shape)
                entry = {"orig_shape": orig_shape, "numel": t.numel(), "dtype": str(t.dtype)}
                if t.ndim == 2:
                    t = t.float()
                    t = t * (t.abs() >= threshold)
                    csr = t.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
                    entry.update(layout="csr", csr_shape=list(csr.shape), nnz=int(csr.values().numel()))
                    torch.save(csr, _tensor_path(out_dir, name))
                    del csr
                else:
                    # ndim <= 1 (norms, biases) or ndim > 2 (conv kernels): kept
                    # DENSE, not pruned. See docs/research/streaming_prune.rst:
                    # streaming_prune.conv_kernel_dense_rationale.
                    entry.update(layout="dense", csr_shape=None, nnz=int((t != 0).sum().item()))
                    torch.save(t.float(), _tensor_path(out_dir, name))
                del t
                manifest[name] = entry
                n_done += 1
                if n_done % 25 == 0:
                    flush_manifest()
                    if verbose:
                        print(f"[streaming]  {len(manifest)} tensors done (last: {name}, nnz={entry['nnz']})")
    flush_manifest()
    manifest["_meta"] = manifest.get("_meta", {})
    manifest["_meta"].update(threshold=threshold, model_dir=model_dir)
    flush_manifest()
    if verbose:
        total_nnz = sum(e["nnz"] for k, e in manifest.items() if k != "_meta")
        print(
            f"[streaming]  phase 1 complete: {len(manifest) - 1} tensors, "
            f"total nnz={total_nnz:,}, threshold={threshold:.5f}"
        )
    return manifest


def estimate_suffix_bytes(manifest: dict, prefix: str, suffix: str) -> int:
    """Predicted stacked-CSR footprint (fp32 values + int32 cols + ptrs)."""
    pat = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix) + r"$")
    total = 0
    rows = 0
    for name, e in manifest.items():
        if name == "_meta" or not pat.match(name):
            continue
        total += e["nnz"] * 8  # value fp32 + col int32
        rows += (e["csr_shape"] or e["orig_shape"])[0]
    return total + (rows + 1) * 8  # crow ptrs


def streaming_fold_suffix(
    out_dir: str, prefix: str, suffix: str, n_layers: int, mem_budget_gb: float = 8.0, verbose: bool = True
):
    """
    Phase 2: sequentially load one suffix across layers and stack into a
    FoldedBlockDescriptor. If the manifest-predicted footprint exceeds
    mem_budget_gb, returns a LIST of per-layer descriptors (n_folds=1 each)
    instead of one stacked descriptor -- degraded but functional (--no-stack).
    """
    from sili.conversion.rnn_fold import FoldedBlockDescriptor, stack_csr_vertical

    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)

    est = estimate_suffix_bytes(manifest, prefix, suffix)
    over_budget = est > mem_budget_gb * (1024**3)
    if verbose:
        print(
            f"[streaming]  fold {prefix}*{suffix}: predicted "
            f"{est / 1e9:.2f} GB ({'PER-LAYER fallback' if over_budget else 'stacked'})"
        )

    def load(i):
        name = f"{prefix}{i}{suffix}"
        t = torch.load(_tensor_path(out_dir, name), weights_only=False)
        if not t.is_sparse_csr:
            t = t.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
        return t

    def make_desc(csr_list, indices):
        stacked = stack_csr_vertical(csr_list) if len(csr_list) > 1 else csr_list[0]
        return FoldedBlockDescriptor(
            n_folds=len(csr_list),
            block_indices=indices,
            stacked_weights={suffix: stacked},
            out_dims={suffix: int(csr_list[0].shape[0])},
            band_half_widths={suffix: None},
            prefix=prefix,
        )

    if over_budget:
        return [make_desc([load(i)], [i]) for i in range(n_layers)]

    csr_list = [load(i) for i in range(n_layers)]
    return make_desc(csr_list, list(range(n_layers)))
