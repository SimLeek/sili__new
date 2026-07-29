#!/usr/bin/env python3
"""Aggregates and diffs repeated bench_block4_layer.py JSON reports from two
directories (baseline vs this branch), one file per repeat, produced by an
INTERLEAVED run (see compare_block4_venvs.sh). Speed is treated as a
statistical measurement: reports the per-repeat PAIRED speedup (repeat i's
baseline time / repeat i's new time -- paired because interleaving puts both
runs of a repeat close together in wall-clock time, canceling a lot of the
drift a single before/after comparison can't), not a single before/after
number. nnz/quality/block4-population are checked for consistency across
repeats (should be identical every time given the seeded RNGs) rather than
averaged."""
import glob
import json
import os
import statistics
import sys


def load_repeats(dir_path):
    files = sorted(glob.glob(os.path.join(dir_path, "*.json")),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not files:
        raise SystemExit(f"no JSON reports found in {dir_path}")
    return [json.load(open(f)) for f in files]


def get(d, path):
    cur = d
    for p in path.split("."):
        cur = cur[p]
    return cur


TIMING_ROWS = [
    ("forward (pre-growth)",  "forward_before_growth.median_s"),
    ("backward (pre-growth)", "backward_before_growth.median_s"),
    ("forward (post-growth)", "forward_after_growth.median_s"),
    ("backward (post-growth)","backward_after_growth.median_s"),
    ("growth phase total",    "growth_phase_s"),
]

CONSISTENCY_ROWS = [
    ("nnz before growth",         "nnz_before_growth"),
    ("nnz after growth",          "nnz_after_growth"),
    ("growth phase throws",       "growth_phase_throws"),
    ("quality err (pre-growth)",  "quality_before_growth_max_abs_err"),
    ("quality err (post-growth)", "quality_after_growth_max_abs_err"),
]


def fmt_us(seconds):
    return f"{seconds * 1e6:.1f}us"


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} baseline_dir new_dir", file=sys.stderr)
        return 1
    base_runs = load_repeats(sys.argv[1])
    new_runs = load_repeats(sys.argv[2])
    n = min(len(base_runs), len(new_runs))
    if len(base_runs) != len(new_runs):
        print(f"WARNING: {len(base_runs)} baseline repeats vs {len(new_runs)} "
              f"new repeats -- using the first {n} of each, paired by index.",
              file=sys.stderr)
    base_runs, new_runs = base_runs[:n], new_runs[:n]

    print(f"Aggregating {n} interleaved repeats.\n")

    # ── Block4 population (context: what does this density actually do) ──
    tiles_after = [get(r, "block4_tiles_after_growth") for r in new_runs]
    syn_after = [get(r, "block4_synapses_after_growth") for r in new_runs]
    nnz_after = [get(r, "nnz_after_growth") for r in new_runs]
    print("Block4 population (this branch, after growth):")
    if len(set(tiles_after)) == 1 and len(set(syn_after)) == 1:
        t, s, z = tiles_after[0], syn_after[0], nnz_after[0]
        pct = (100.0 * s / z) if z else 0.0
        print(f"  {t} tiles, {s} synapses out of {z} total nnz ({pct:.2f}%)")
    else:
        print(f"  INCONSISTENT across repeats -- tiles={tiles_after} synapses={syn_after}"
              " (unexpected given seeded RNGs; investigate before trusting the run)")
    print()

    # ── Paired per-repeat speedups ─────────────────────────────────────────
    print(f"{'':30s} {'median':>10s} {'min':>10s} {'max':>10s}  (speedup = baseline/new, "
          f">1x = this branch faster)")
    print("-" * 80)
    for label, path in TIMING_ROWS:
        speedups = []
        for b, n_ in zip(base_runs, new_runs):
            bt, nt = get(b, path), get(n_, path)
            if nt > 0:
                speedups.append(bt / nt)
        if not speedups:
            print(f"{label:30s} (no data)")
            continue
        med = statistics.median(speedups)
        print(f"{label:30s} {med:9.2f}x {min(speedups):9.2f}x {max(speedups):9.2f}x")

    print()
    print(f"{'':30s} {'baseline':>14} {'this branch':>14} {'consistent':>12}")
    print("-" * 80)
    mismatches = []
    for label, path in CONSISTENCY_ROWS:
        b_vals = [get(r, path) for r in base_runs]
        n_vals = [get(r, path) for r in new_runs]
        b_consistent = len(set(b_vals)) == 1
        n_consistent = len(set(n_vals)) == 1
        if "quality" in path:
            match = b_consistent and n_consistent and abs(b_vals[0] - n_vals[0]) < 1e-3
        else:
            match = b_consistent and n_consistent and b_vals[0] == n_vals[0]
        if not match:
            mismatches.append(label)
        b_disp = b_vals[0] if b_consistent else f"VARIES {b_vals}"
        n_disp = n_vals[0] if n_consistent else f"VARIES {n_vals}"
        print(f"{label:30s} {b_disp!s:>14} {n_disp!s:>14} {'OK' if match else 'DIFFERS':>12}")

    print()
    if mismatches:
        print(f"MISMATCH on: {', '.join(mismatches)} -- growth is driven by "
              "accumulated activity stats + probe selection, which is "
              "identical logic on both branches under a seeded RNG, so "
              "nnz/quality should match exactly and be constant across "
              "repeats. Investigate before trusting the speed numbers above.")
        return 1
    print("nnz and quality match exactly between baseline and this branch, and are "
          "stable across all repeats, as expected (block4 only changes HOW live "
          "synapses are stored, never WHICH ones exist or what they compute) -- "
          "the speedup numbers above are a clean, statistically-repeated comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
