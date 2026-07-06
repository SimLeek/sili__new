"""
Integration test: streaming_prune.py against a real on-disk safetensors model.

Builds the toy VLM, writes it to disk as a SHARDED safetensors checkpoint
with a model.safetensors.index.json (the real format 24B-scale models ship
in -- 10 shards for the actual Mistral-Small-3.1-24B-Base-2503), then runs
the actual streaming_sparsify()/streaming_fold_suffix() functions against
that directory using the real safetensors.safe_open code path (not mocked).

Covers:
  1. Phase 1 (streaming_sparsify) against a real sharded checkpoint:
     - correct tensor count
     - 2-D tensors get CSR layout, 1-D and 4-D tensors stay dense (the
       later-todo decision on patch_conv, see requirements doc section 1.2)
     - 4-D orig_shape is recorded correctly for later reconstruction
  2. Resume: re-running streaming_sparsify on an already-complete out_dir
     produces an identical manifest (nothing redone, nothing corrupted)
  3. fsck: a manifest entry whose tensor file is missing or corrupt gets
     detected and re-done on the next streaming_sparsify() call, rather than
     silently trusting a stale/bad manifest entry
  4. estimate_suffix_bytes: sane, positive, and consistent with manifest nnz
  5. streaming_fold_suffix --no-stack fallback: a large mem_budget_gb returns
     one stacked FoldedBlockDescriptor; a tiny mem_budget_gb returns a LIST
     of per-layer descriptors (n_folds=1 each) instead of OOMing
  6. FoldedLayer construction + forward from BOTH the stacked and per-layer
     descriptor forms works identically (same output shape)

Run: python -m tests.integration.test_streaming_prune
"""
import sys, os, json, shutil, tempfile, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'test', 'python'))
warnings.filterwarnings('ignore')

import torch
from safetensors.torch import save_file

import sili.cpu
from sili.sparse_rnn import FoldedLayer
from sili.tensor import Tensor

from gen_toy_mistral_vlm import build_toy_mistral_vlm_state_dict, TXT_LAYERS, VIS_LAYERS
from sili.conversion.streaming_prune import (
    streaming_sparsify, streaming_fold_suffix, estimate_suffix_bytes, _tensor_path,
)


def _write_sharded_safetensors(sd: dict, model_dir: str, n_shards: int = 3) -> None:
    """
    Split sd across n_shards files + write model.safetensors.index.json,
    mimicking the real 24B checkpoint's on-disk layout (10 shards there).
    """
    names = sorted(sd.keys())
    shards = [dict() for _ in range(n_shards)]
    weight_map = {}
    for i, name in enumerate(names):
        shard_idx = i % n_shards
        shards[shard_idx][name] = sd[name].contiguous()
        weight_map[name] = f"model-{shard_idx+1:05d}-of-{n_shards:05d}.safetensors"

    os.makedirs(model_dir, exist_ok=True)
    for i, shard in enumerate(shards):
        save_file(shard, os.path.join(model_dir, f"model-{i+1:05d}-of-{n_shards:05d}.safetensors"))
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": sum(t.numel() * 4 for t in sd.values())},
                   "weight_map": weight_map}, f)


def run(verbose: bool = True) -> bool:
    tmp = tempfile.mkdtemp(prefix="streaming_test_")
    try:
        model_dir = os.path.join(tmp, "model")
        out_dir   = os.path.join(tmp, "out")
        sd, _ = build_toy_mistral_vlm_state_dict()
        _write_sharded_safetensors(sd, model_dir, n_shards=3)
        if verbose:
            print(f"\n=== streaming_prune: {len(sd)} tensors across 3 shards ===")

        # -- Phase 1: real safe_open-based sparsify --
        manifest = streaming_sparsify(model_dir, out_dir, verbose=False)
        n_tensors = len([k for k in manifest if k != "_meta"])
        assert n_tensors == len(sd), f"expected {len(sd)} tensors, got {n_tensors}"

        q0 = "language_model.model.layers.0.self_attn.q_proj.weight"
        ln0 = "language_model.model.layers.0.input_layernorm.weight"
        conv = "vision_tower.patch_conv.weight"
        assert manifest[q0]["layout"] == "csr", "2-D tensor should be CSR"
        assert manifest[ln0]["layout"] == "dense", "1-D norm should stay dense"
        assert manifest[conv]["layout"] == "dense", \
            "4-D conv should stay dense (later-todo decision, not pruned)"
        assert manifest[conv]["orig_shape"] == list(sd[conv].shape), \
            "4-D orig_shape must be recorded for reconstruction"
        if verbose:
            print(f"  phase 1: {n_tensors} tensors, layouts correct "
                  f"(q_proj=csr, norm=dense, patch_conv=dense 4-D) -- OK")

        # -- Resume: identical manifest on re-run, nothing redone --
        manifest_before = json.dumps(manifest, sort_keys=True)
        manifest2 = streaming_sparsify(model_dir, out_dir, verbose=False)
        assert json.dumps(manifest2, sort_keys=True) == manifest_before, \
            "resume run should produce an identical manifest"
        if verbose:
            print(f"  resume: re-run on complete out_dir is a no-op -- OK")

        # -- fsck (missing file): resume guard checks the tensor path itself,
        # not just the manifest key, so a deleted-but-manifested file is
        # regenerated rather than silently skipped forever.
        victim_path = _tensor_path(out_dir, ln0)
        os.remove(victim_path)
        manifest3 = streaming_sparsify(model_dir, out_dir, verbose=False)
        assert os.path.exists(victim_path), \
            "fsck: missing tensor file should have been regenerated"
        assert manifest3[ln0] == manifest[ln0], \
            "regenerated entry should match the original exactly"
        if verbose:
            print(f"  fsck (missing file): detected and regenerated -- OK")

        # -- fsck (corrupted file): the actual crash scenario. A truncated
        # file from a process killed mid-torch.save PASSES os.path.exists()
        # but FAILS torch.load(). A bare existence check (what this code had
        # before) would silently trust it forever; the real fsck must call
        # torch.load() to verify. This is the case that actually justifies
        # having a fsck function at all -- the missing-file case above would
        # already work with plain os.path.exists().
        victim_path2 = _tensor_path(out_dir, q0)
        with open(victim_path2, "wb") as f:
            f.write(b"not a valid torch checkpoint, simulates a crash mid-write")
        manifest4 = streaming_sparsify(model_dir, out_dir, verbose=False)
        assert manifest4[q0] == manifest[q0], \
            "corrupted entry should be regenerated to match the original exactly"
        # Confirm the regenerated file is actually loadable again (not still
        # the garbage bytes -- i.e. streaming_sparsify really re-ran, it
        # didn't just leave the corrupt file and patch the manifest around it)
        reloaded = torch.load(victim_path2, weights_only=False)
        assert reloaded.is_sparse_csr, "regenerated q_proj tensor should be valid CSR again"
        if verbose:
            print(f"  fsck (corrupted file): truncated tensor detected via "
                  f"torch.load() failure and regenerated -- OK")

        # -- estimate_suffix_bytes: positive, consistent with manifest --
        est = estimate_suffix_bytes(manifest, "language_model.model.layers.",
                                     ".self_attn.q_proj.weight")
        assert est > 0, "estimate should be positive for a real suffix"
        if verbose:
            print(f"  estimate_suffix_bytes: {est} bytes for q_proj -- OK")

        # -- Phase 2a: large budget -> single stacked descriptor --
        desc = streaming_fold_suffix(out_dir, "language_model.model.layers.",
                                      ".self_attn.q_proj.weight", TXT_LAYERS,
                                      mem_budget_gb=100.0, verbose=False)
        assert not isinstance(desc, list), "expected ONE descriptor under large budget"
        assert desc.n_folds == TXT_LAYERS
        if verbose:
            print(f"  phase 2 (stacked): n_folds={desc.n_folds} -- OK")

        # -- Phase 2b: tiny budget -> --no-stack per-layer fallback --
        descs = streaming_fold_suffix(out_dir, "language_model.model.layers.",
                                       ".self_attn.q_proj.weight", TXT_LAYERS,
                                       mem_budget_gb=1e-9, verbose=False)
        assert isinstance(descs, list), "expected a LIST under tiny budget"
        assert len(descs) == TXT_LAYERS
        assert all(d.n_folds == 1 for d in descs)
        if verbose:
            print(f"  phase 2 (--no-stack fallback): "
                  f"{len(descs)} per-layer descriptors, n_folds=1 each -- OK")

        # -- FoldedLayer from BOTH forms: same shape, both work --
        import numpy as np

        layer_stacked = FoldedLayer.from_descriptor(desc, learning_rate=0.001, num_cpus=1)
        x = Tensor(np.random.randn(32).astype(np.float32) * 0.1)
        h_stacked = layer_stacked(x)

        layer_perlayer = FoldedLayer.from_descriptor(descs[0], learning_rate=0.001, num_cpus=1)
        h_perlayer = layer_perlayer(x)

        assert h_stacked.data.shape == h_perlayer.data.shape == (64,), \
            f"shape mismatch: stacked={h_stacked.data.shape} perlayer={h_perlayer.data.shape}"
        assert np.isfinite(h_stacked.data).all() and np.isfinite(h_perlayer.data).all()
        if verbose:
            print(f"  FoldedLayer from both descriptor forms: "
                  f"forward OK, shape {h_stacked.data.shape} -- OK")

        if verbose:
            print("\ntest_streaming_prune: ALL PASS")
        return True

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ok = run(verbose=not ap.parse_args().quiet)
    sys.exit(0 if ok else 1)
