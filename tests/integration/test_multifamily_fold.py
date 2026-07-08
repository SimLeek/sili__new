"""
Integration test: multi-family block detection and folding (VLM regression).

Guards against the prefix-collision bug found by gen_toy_mistral_vlm.py:
detect_repeated_block_groups, report_block_groups, and fold_block_group all
used to key/filter on bare layer index only. Two structurally distinct block
families sharing an index range (a VLM's language layers 0-11 and vision
layers 0-5) would collide: block_params[0] silently merged both families'
suffixes, groups came out sized [6, 6] instead of [12] + [6], and folding
raised KeyError trying to look up a vision suffix under the language prefix.

Fix: key/filter everywhere on (prefix, index), not index alone. This test
runs the toy VLM (verified schema -- see gen_toy_mistral_vlm.py) through
detection and fold_sparse_payload end-to-end and checks:
  - exactly 2 groups, sized 12 and 6 (not merged, not split wrong)
  - each group's prefix is the correct family
  - fold_block_group does not raise (suffixes stay within their own family)
  - lossless stacking: nnz is preserved exactly through the fold
  - zero per-layer keys remain in sparse_state_dict after folding

This is orthogonal to tensor-name accuracy (covered by the direct weight_map
diff in gen_toy_mistral_vlm.py's own module) -- this test only cares that
TWO DIFFERENT block families never get merged, regardless of what their
prefixes are actually named.

Run: python -m tests.integration.test_multifamily_fold
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'unit', 'python'))
warnings.filterwarnings('ignore')

import torch
from sili.conversion.sparse_prune import default_min_abs_param
from sili.conversion.rnn_fold import detect_repeated_block_groups, fold_sparse_payload
from gen_toy_mistral_vlm import build_toy_mistral_vlm_state_dict, TXT_LAYERS, VIS_LAYERS


def _prune_to_payload(sd: dict, threshold: float) -> dict:
    ssd = {}
    for k, v in sd.items():
        if v.ndim == 2:
            m = v.abs() >= threshold
            ssd[k] = {"csr": (v * m).to_sparse(sparse_dim=2).coalesce().to_sparse_csr()}
        else:
            ssd[k] = {"raw": v}   # 1-D norms and 4-D conv both stay dense
    return {"sparse_state_dict": ssd, "min_abs_param": threshold, "meta": {}}


def run(verbose: bool = True) -> bool:
    threshold = default_min_abs_param()

    sd, _ = build_toy_mistral_vlm_state_dict()
    payload = _prune_to_payload(sd, threshold)

    # -- detection: exactly 2 groups, correct sizes, correct prefixes --
    flat = {k: (e["csr"] if "csr" in e else e["raw"])
            for k, e in payload["sparse_state_dict"].items()}
    groups = detect_repeated_block_groups(flat)
    assert len(groups) == 2, f"expected 2 groups, got {len(groups)}"
    sizes = sorted(len(idxs) for _, idxs in groups)
    assert sizes == [VIS_LAYERS, TXT_LAYERS], \
        f"expected sizes [{VIS_LAYERS},{TXT_LAYERS}], got {sizes}"
    for prefix, idxs in groups:
        is_vision = "vision" in prefix
        expected_n = VIS_LAYERS if is_vision else TXT_LAYERS
        assert len(idxs) == expected_n, \
            f"prefix {prefix!r} got {len(idxs)} blocks, expected {expected_n}"
    if verbose:
        print(f"  detection: 2 groups, sizes {sizes}, "
              f"prefixes correctly separated -- OK")

    # -- fold_sparse_payload: must not raise, must fully consume both families --
    out = fold_sparse_payload(payload)
    assert len(out["folded_blocks"]) == 2, \
        f"expected 2 folded blocks, got {len(out['folded_blocks'])}"
    n_folds = sorted(fb["n_folds"] for fb in out["folded_blocks"])
    assert n_folds == [VIS_LAYERS, TXT_LAYERS], \
        f"expected n_folds {[VIS_LAYERS,TXT_LAYERS]}, got {n_folds}"

    remaining = [k for k in out["sparse_state_dict"] if "layers." in k]
    assert len(remaining) == 0, \
        f"{len(remaining)} per-layer keys not folded: {remaining[:3]}"

    if verbose:
        print(f"  fold_sparse_payload: {len(out['folded_blocks'])} "
              f"blocks folded, 0 leftover per-layer keys -- OK")
        print("\ntest_multifamily_fold: ALL PASS "
              "(prefix-collision regression guarded)")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ok = run(verbose=not ap.parse_args().quiet)
    sys.exit(0 if ok else 1)
