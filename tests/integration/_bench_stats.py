"""Shared repeat-and-compare statistics for the sili-vs-torch perf
integration tests (test_perf_disldo_vs_torch.py, test_perf_matrix_vs_torch.py,
test_perf_didldo_sidldo_kernels.py).

Each measurement is repeated `--repeats` independent times (each repeat
itself an averaged loop of many calls), and the mean/std ACROSS repeats is
reported -- this captures run-to-run system noise (thermal, background
load, scheduler jitter), not just intra-loop variance. `gate()` compares
against a COMMITTED per-machine baseline (perf_baselines/<machine_uid>.json,
absolute times are only comparable on the SAME machine -- see
machine_info()'s own docstring) via a Welch-style two-sample z-test on the
means: z = diff / sqrt(std1^2/n1 + std2^2/n2). |z| > 2 AND a minimum
practical-magnitude change (both required, see compare()'s own docstring)
is a real regression -- these gate CI (see .github/workflows/ci.yml).
Accepting a genuine regression means re-running with --save-baseline and
committing the regenerated JSON: a normal, visible PR file change, not a
hidden override.
"""

import hashlib
import json
import os
import platform
import re
import statistics
import time

DEFAULT_PERF_BASELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf_baselines")


def machine_info():
    """CPU model, relevant SIMD flags, and a stable per-machine hash --
    absolute benchmark numbers are only comparable across runs on the SAME
    machine (half-rate vs full-rate AVX2 laptops/desktops give wildly
    different numbers for identical code), so baselines are tagged with
    this and compare() warns on mismatch rather than silently comparing
    apples to oranges.

    Under GitHub Actions (GITHUB_ACTIONS=true), /etc/machine-id is a FRESH
    random value every single job run (ephemeral VM) -- using it as-is
    would make every CI run look like "first run, no baseline yet" and
    never actually gate anything. RUNNER_NAME's own numeric suffix is
    similarly per-instance, not per runner-generation. Use a fixed
    synthetic id keyed by the runner image label instead
    (RUNNER_ENVIRONMENT/ImageOS, e.g. "github-ubuntu-latest") so the
    COMMITTED baseline for that runner generation persists meaningfully
    across job runs; only changes when GitHub rotates the image itself
    (a real, if infrequent, cause of genuine cross-run drift)."""
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

    if os.environ.get("GITHUB_ACTIONS") == "true":
        # ImageOS is GitHub Actions' OWN env var name (case-sensitive,
        # exact match required) -- not a name this file gets to choose.
        uid_src = "github-" + os.environ.get("ImageOS", os.environ.get("RUNNER_OS", "unknown-image"))  # noqa: SIM112
    else:
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


def compare(new_mean, new_std, new_n, old_mean, old_std, old_n, min_pct=5.0):
    """Returns (pct_change, verdict) where verdict is one of
    'IMPROVED', 'REGRESSED', 'noise', or 'n/a' (not enough repeats either
    side to run the z-test).

    Flagging on the z-test alone is too sensitive in practice: the
    within-run repeat variance this z-test is built from is much tighter
    than the TRUE across-process/across-time noise floor (thermal drift,
    scheduler decisions, other load on the box between the baseline run
    and this one) -- confirmed empirically even on quiet dedicated
    hardware, where sub-1% wobbles were statistically "significant" by
    the z-test alone but not remotely a real regression. Requiring BOTH
    z-significance AND a minimum practical magnitude (default 5%) filters
    that out while still catching real, sizeable changes.
    """
    if old_mean == 0:
        return float("inf"), "n/a"
    pct_change = (new_mean - old_mean) / old_mean * 100.0
    if new_n < 2 or old_n < 2 or abs(pct_change) < min_pct:
        return pct_change, "n/a" if (new_n < 2 or old_n < 2) else "noise"

    se = ((new_std**2) / new_n + (old_std**2) / old_n) ** 0.5
    z = float("inf") if se == 0 else (new_mean - old_mean) / se
    if z > 2:
        verdict = "REGRESSED"
    elif z < -2:
        verdict = "IMPROVED"
    else:
        verdict = "noise"
    return pct_change, verdict


def baseline_path_for_machine(meta, baseline_dir=None):
    """Committed per-machine baseline path -- each machine gets its own
    file rather than one shared baseline different runners would fight
    over (or silently invalidate) with incomparable absolute numbers."""
    baseline_dir = baseline_dir or DEFAULT_PERF_BASELINE_DIR
    return os.path.join(baseline_dir, f"{meta['machine_uid']}.json")


def gate(flat_results, meta, baseline_dir=None, min_pct=5.0, quiet=False):
    """Compares flat_results (key -> {mean,std,n}) against this machine's
    committed baseline. No baseline yet: writes one and returns (True, [])
    -- establishing a baseline is not a failure, so a brand-new runner's
    first CI run passes. Baseline exists: compares every matching key,
    returns (ok, regressions) where ok is False iff any key came back
    REGRESSED. Callers gate CI by checking `ok` (see the perf integration
    tests' own main()) -- this function itself never exits the process.
    """
    path = baseline_path_for_machine(meta, baseline_dir)
    baseline_doc = load_baseline(path)
    if baseline_doc is None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_baseline(path, {"_meta": meta, "results": flat_results})
        if not quiet:
            print(f"(no baseline yet at {path} -- wrote one from this run)")
        return True, []

    baseline = baseline_doc.get("results", baseline_doc)  # tolerate old flat-only format
    old_meta = baseline_doc.get("_meta") or {}
    if old_meta.get("machine_uid") not in (None, meta["machine_uid"]):
        # Should be structurally impossible (path is keyed by machine_uid),
        # but a hand-edited/copied baseline file could still hit this --
        # fail loudly rather than silently comparing across machines.
        raise RuntimeError(
            f"baseline at {path} is tagged machine_uid={old_meta.get('machine_uid')}, "
            f"this run is {meta['machine_uid']} -- refusing to compare"
        )
    if old_meta.get("num_cpus") not in (None, meta.get("num_cpus")):
        print(
            f"WARNING: baseline was recorded with num_cpus={old_meta.get('num_cpus')}, this "
            f"run used num_cpus={meta.get('num_cpus')} -- a thread-count change will show up "
            f"as fake REGRESSED/IMPROVED verdicts below, not a real code change."
        )

    regressions = []
    if not quiet:
        print(f"=== Compared against baseline: {path} ===")
        print(f"{'key':>40} | {'baseline':>10} {'now':>10} | {'change':>8} {'verdict':>10}")
    for key in sorted(flat_results):
        if key not in baseline:
            continue
        new, old = flat_results[key], baseline[key]
        pct, verdict = compare(new["mean"], new["std"], new["n"], old["mean"], old["std"], old["n"], min_pct=min_pct)
        if verdict == "REGRESSED":
            regressions.append((key, old["mean"], new["mean"], pct))
        if not quiet and verdict in ("REGRESSED", "IMPROVED"):
            print(f"{key:>40} | {old['mean']:>9.4f}ms {new['mean']:>9.4f}ms | {pct:>+7.1f}% {verdict:>10}")
    ok = len(regressions) == 0
    if not quiet:
        print("(no statistically significant regressions)" if ok else f"{len(regressions)} REGRESSED")
    return ok, regressions


def update_baseline(flat_results, meta, baseline_dir=None):
    """Explicit override for a genuine, accepted regression -- re-writes
    this machine's committed baseline. The resulting file change is a
    normal, visible part of the PR diff, per direct instruction: overriding
    the gate must be clear that it's being done, not a hidden bypass."""
    path = baseline_path_for_machine(meta, baseline_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_baseline(path, {"_meta": meta, "results": flat_results})
    return path
