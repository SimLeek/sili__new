#!/usr/bin/env python3
"""Side-by-side diff of two bench_block4_layer.py JSON reports (baseline vs
this branch). Speed rows show speedup as new/baseline (>1x = faster on
this branch); quality/nnz rows are flagged only if they actually diverge,
since those should match exactly regardless of internal representation."""
import json
import sys


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

CHECK_ROWS = [
    ("nnz before growth",           "nnz_before_growth"),
    ("nnz after growth",            "nnz_after_growth"),
    ("growth phase throws",         "growth_phase_throws"),
    ("quality err (pre-growth)",    "quality_before_growth_max_abs_err"),
    ("quality err (post-growth)",   "quality_after_growth_max_abs_err"),
]


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} baseline.json new.json", file=sys.stderr)
        return 1
    with open(sys.argv[1]) as f:
        base = json.load(f)
    with open(sys.argv[2]) as f:
        new = json.load(f)

    print(f"{'':30s} {'baseline':>14s} {'this branch':>14s} {'speedup':>10s}")
    print("-" * 70)
    for label, path in TIMING_ROWS:
        b = get(base, path)
        n = get(new, path)
        speedup = (b / n) if n > 0 else float("nan")
        print(f"{label:30s} {b*1e6:11.1f}us {n*1e6:11.1f}us {speedup:9.2f}x")

    print()
    print(f"{'':30s} {'baseline':>14} {'this branch':>14} {'match':>10}")
    print("-" * 70)
    mismatches = []
    for label, path in CHECK_ROWS:
        b = get(base, path)
        n = get(new, path)
        if "quality" in path:
            ok = abs(b - n) < 1e-3  # both should sit at the FP4 rounding noise floor
        else:
            ok = (b == n)
        if not ok:
            mismatches.append(label)
        print(f"{label:30s} {b!s:>14} {n!s:>14} {'OK' if ok else 'DIFFERS':>10}")

    print()
    if mismatches:
        print(f"MISMATCH on: {', '.join(mismatches)} -- growth is driven by "
              "accumulated activity stats + probe selection, which is "
              "identical logic on both branches, so nnz/quality should "
              "match exactly. Investigate before trusting the speed numbers.")
        return 1
    print("nnz and quality match between baseline and this branch, as expected "
          "(block4 only changes HOW live synapses are stored, never WHICH "
          "ones exist or what they compute) -- the timing numbers above are "
          "a clean comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
