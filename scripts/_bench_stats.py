"""Shared repeat-and-compare statistics for the sili-vs-torch benchmark
scripts (bench_sili_vs_torch.py, bench_sili_vs_torch_matrix.py).

Each measurement is repeated `--repeats` independent times (each repeat
itself an averaged loop of many calls), and the mean/std ACROSS repeats is
reported -- this captures run-to-run system noise (thermal, background
load, scheduler jitter), not just intra-loop variance. A saved baseline
(JSON, default under /tmp -- these numbers are machine- and moment-
specific, not something to commit) lets a later run report whether each
cell improved, regressed, or moved within noise, via a Welch-style
two-sample z-test on the means: z = diff / sqrt(std1^2/n1 + std2^2/n2).
|z| > 2 is flagged as a real change; anything less is "noise" and the
percentage is still printed so the reader can judge magnitude themselves.
"""

import hashlib
import json
import platform
import re
import statistics
import time


def machine_info():
    """CPU model, relevant SIMD flags, and a stable per-machine hash --
    absolute benchmark numbers are only comparable across runs on the SAME
    machine (half-rate vs full-rate AVX2 laptops/desktops give wildly
    different numbers for identical code), so baselines are tagged with
    this and compare() warns on mismatch rather than silently comparing
    apples to oranges."""
    model = "unknown"
    flags = []
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
        m = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        if m:
            model = m.group(1).strip()
        m = re.search(r"^flags\s*:\s*(.+)$", text, re.MULTILINE)
        if m:
            wanted = {"avx", "avx2", "avx512f", "fma"}
            flags = sorted(set(m.group(1).split()) & wanted)
    except OSError:
        model = platform.processor() or platform.machine()

    uid_src = None
    try:
        with open("/etc/machine-id") as f:
            uid_src = f.read().strip()
    except OSError:
        uid_src = platform.node() + model
    machine_uid = hashlib.sha256(uid_src.encode()).hexdigest()[:12]

    return {"cpu_model": model, "simd_flags": flags, "machine_uid": machine_uid}


def bench_repeated(fn, n_calls, repeats, warmup=5):
    """Run `fn` `n_calls` times per repeat, `repeats` times total. Returns
    (mean_ms, std_ms) of the per-call time ACROSS repeats."""
    for _ in range(warmup):
        fn()
    per_repeat_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            fn()
        per_repeat_ms.append((time.perf_counter() - t0) / n_calls * 1000.0)
    mean_ms = statistics.fmean(per_repeat_ms)
    std_ms = statistics.stdev(per_repeat_ms) if repeats > 1 else 0.0
    return mean_ms, std_ms


def load_baseline(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_baseline(path, results):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def compare(new_mean, new_std, new_n, old_mean, old_std, old_n):
    """Returns (pct_change, verdict) where verdict is one of
    'IMPROVED', 'REGRESSED', 'noise', or 'n/a' (not enough repeats either
    side to run the z-test)."""
    if old_mean == 0:
        return float("inf"), "n/a"
    pct_change = (new_mean - old_mean) / old_mean * 100.0
    if new_n < 2 or old_n < 2:
        return pct_change, "n/a"
    se = ((new_std**2) / new_n + (old_std**2) / old_n) ** 0.5
    if se == 0:
        verdict = "noise" if new_mean == old_mean else ("REGRESSED" if new_mean > old_mean else "IMPROVED")
        return pct_change, verdict
    z = (new_mean - old_mean) / se
    if z > 2:
        return pct_change, "REGRESSED"
    if z < -2:
        return pct_change, "IMPROVED"
    return pct_change, "noise"
