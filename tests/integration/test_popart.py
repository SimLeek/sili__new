"""
Integration test: PopArt rescale-invariance (fast, no I/O, no torch).

The property that makes PopArt correct rather than just "a normalizer": when
update_and_rescale() changes mean/std, the PREDICTION IN ORIGINAL SPACE for
any given input must be UNCHANGED by that renormalization event alone -- only
subsequent gradient steps should move it. This is exactly the property a
naive normalizer (rescale the target, leave the weights alone) gets wrong,
and it is the entire point of the "Pop" half of PopArt.

Test: for a fixed (h, action) pair, compute the ORIGINAL-SPACE prediction
before and after calling update_and_rescale() with a new observation --
WITHOUT any intervening gradient step -- and assert they match.

Run: python -m tests.integration.test_popart
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from sili.rl_utils import PopArt


def _predict(h, action_onehot, W, B, n_act):
    """v(h, a) = h @ W + B[a]   (B rows are the per-action bias)"""
    return (h @ W + action_onehot @ B).item()


def run(verbose: bool = True) -> bool:
    rng = np.random.default_rng(0)
    d_h, n_act = 8, 4

    W = (rng.standard_normal((d_h, 1)) * 0.1).astype(np.float64)
    B = (rng.standard_normal((n_act, 1)) * 0.1).astype(np.float64)
    pa = PopArt(beta=0.05, start_pop=2)  # small start_pop, fast test

    h = rng.standard_normal(d_h)
    onehot = np.zeros(n_act); onehot[1] = 1.0

    # Warm up past start_pop with small, consistent-scale targets so the
    # first real rescale event (below) has stable statistics to work with.
    for raw_target in (1.0, 1.1, 0.9):
        pa.update_and_rescale(raw_target, weight_arrays=[W], bias_arrays=[B])

    n_checks = 0
    for raw_target in (5.0, -3.0, 100.0, 0.001, -50.0):
        # Snapshot BEFORE this update: normalized-space prediction + stats.
        pred_norm_before = _predict(h, onehot, W, B, n_act)
        mean_before, std_before = pa.mean, pa.std
        orig_before = pred_norm_before * std_before + mean_before

        pa.update_and_rescale(raw_target, weight_arrays=[W], bias_arrays=[B])

        # AFTER: W/B have been rescaled in place, mean/std have moved too.
        # The property PopArt guarantees is that the DENORMALIZED prediction
        # (original units) is unchanged by the stats update alone -- the
        # normalized-space number itself is EXPECTED to change (that's the
        # whole point of rescaling the weights to compensate for new stats).
        pred_norm_after = _predict(h, onehot, W, B, n_act)
        orig_after = pred_norm_after * pa.std + pa.mean

        assert np.isclose(orig_before, orig_after, atol=1e-6), (
            f"PopArt rescale broke original-space invariance: "
            f"before={orig_before:.6f} after={orig_after:.6f} "
            f"(raw_target={raw_target})"
        )
        n_checks += 1

    if verbose:
        print(f"\n=== PopArt rescale-invariance ({n_checks} updates) ===")
        print(f"  every update_and_rescale() call preserved the DENORMALIZED "
              f"(original-space) prediction to 1e-6 -- OK")

    # Sanity: normalize()/unnormalize() should round-trip
    raw = 42.0
    norm = pa.normalize(raw)
    back = pa.unnormalize(norm)
    assert np.isclose(raw, back, atol=1e-6), f"normalize/unnormalize round-trip: {raw} -> {back}"
    if verbose:
        print(f"  normalize/unnormalize round-trip: {raw} -> {norm:.4f} -> {back:.4f} -- OK")
        print("\ntest_popart: ALL PASS")

    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ok = run(verbose=not ap.parse_args().quiet)
    sys.exit(0 if ok else 1)
