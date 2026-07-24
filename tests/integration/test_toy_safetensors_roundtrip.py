"""
Integration test: toy VLM survives real safetensors serialization exactly.

Complements test_streaming_prune.py (which tests the SHARDED multi-file
pipeline path) with a direct single-file round-trip check using
gen_toy_mistral_vlm.save_and_verify_safetensors(): save every toy tensor to
a real .safetensors file, then read it back tensor-by-tensor via safe_open
(the same API streaming_prune.py uses, not the bulk load_file convenience
wrapper) and assert exact key/shape/dtype/value equality.

safetensors is lossless (no compression, no quantization), so this is a
bit-for-bit check, not a numerical-tolerance one. This exists to rule out
contiguity or metadata issues before trusting any downstream consumer
(streaming_sparsify, fold_one_suffix, etc.) with real on-disk models.

Run: python -m tests.integration.test_toy_safetensors_roundtrip
"""
import sys, os, tempfile, shutil, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'unit', 'python'))
warnings.filterwarnings('ignore')

from gen_toy_mistral_vlm import build_toy_mistral_vlm_state_dict, save_and_verify_safetensors


def run(verbose: bool = True) -> bool:
    tmp = tempfile.mkdtemp(prefix="toy_st_roundtrip_")
    try:
        sd, _ = build_toy_mistral_vlm_state_dict()
        path = os.path.join(tmp, "toy_mistral_vlm.safetensors")
        ok = save_and_verify_safetensors(sd, path, verbose=verbose)
        assert ok, "toy VLM did not survive safetensors round-trip exactly"
        assert os.path.exists(path)
        if verbose:
            print("\ntest_toy_safetensors_roundtrip: ALL PASS")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ok = run(verbose=not ap.parse_args().quiet)
    sys.exit(0 if ok else 1)
