#!/usr/bin/env python3
"""Perf-gates DIDLDO/SIDLDO's raw kernels (bench_didldo_kernel.cpp,
bench_sidldo_forward.cpp, bench_sidldo_backward.cpp) against a committed
per-machine baseline -- same gate() mechanism as test_perf_disldo_vs_torch.py/
test_perf_matrix_vs_torch.py (_bench_stats.py), just wrapping raw C++
binaries instead of calling sili from Python directly.

Self-contained: builds the three .cpp files itself via a direct g++
invocation (same mkl/mkl-include pip-package convention setup.py's
_find_mkl() uses) rather than depending on a separate tests/unit/
build_tests step -- keeps this runnable standalone
(`python -m tests.integration.test_perf_didldo_sidldo_kernels`) like every
other tests/integration/ file, and avoids a cross-CI-job build dependency.
No MKL available: skips gracefully (exit 0, not a failure -- matches the
graceful compile-out convention DIDLDOLayerV/SIDLDOLayerV already use
everywhere else in this codebase).
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bench_stats import gate, machine_info, update_baseline

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HEADERS_DIR = os.path.join(REPO_ROOT, "sili", "lib", "headers")

# name -> (source file, [(row-label-prefix, column names in printed order)])
BENCHES = {
    "didldo_kernel": ("bench_didldo_kernel.cpp", ["batch", "fwd_ms", "bwd0_ms", "bwdX_ms"]),
    "sidldo_forward": ("bench_sidldo_forward.cpp", ["density", "batch", "fwd_ms"]),
    "sidldo_backward": ("bench_sidldo_backward.cpp", ["density", "batch", "bwd0_ms", "bwdX_ms"]),
}


def find_mkl():
    """Same sys.prefix pip-package convention as setup.py's _find_mkl()."""
    lib_dir = os.path.join(sys.prefix, "lib")
    include_dir = os.path.join(sys.prefix, "include")
    if os.path.exists(os.path.join(lib_dir, "libmkl_gnu_thread.so.3")) and os.path.exists(
        os.path.join(include_dir, "mkl_cblas.h")
    ):
        return lib_dir, include_dir
    return None, None


def build(name, src, mkl_lib_dir, mkl_include_dir, build_dir):
    out_path = os.path.join(build_dir, name)
    cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        "-fopenmp",
        "-DSILI_HAVE_MKL",
        "-I",
        HEADERS_DIR,
        "-I",
        mkl_include_dir,
        os.path.join(os.path.dirname(__file__), src),
        "-L",
        mkl_lib_dir,
        f"-Wl,-rpath,{mkl_lib_dir}",
        "-l:libmkl_intel_lp64.so.3",
        "-l:libmkl_gnu_thread.so.3",
        "-l:libmkl_core.so.3",
        "-lgomp",
        "-o",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603 -- our own fixed g++ invocation
    return out_path


# Matches the printed table rows: whitespace-separated floats/ints, no
# header/comment lines (those start with a letter or '#').
_ROW_RE = re.compile(r"^\s*[\d.]+(?:\s+[\d.]+)+\s*$")


def parse_table(stdout, columns):
    rows = []
    for line in stdout.splitlines():
        if _ROW_RE.match(line):
            parts = line.split()
            if len(parts) == len(columns):
                rows.append(dict(zip(columns, (float(p) for p in parts), strict=False)))
    return rows


def run_bench(binary_path):
    result = subprocess.run(  # noqa: S603 -- our own just-built binary, fixed path
        [binary_path], check=True, capture_output=True, text=True, timeout=300
    )
    return result.stdout


def main():  # noqa: C901, PLR0912 -- CLI plumbing (argparse + gate/save branches), reads clearer flat
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir", default=None, help="defaults to tests/integration/perf_baselines/")
    ap.add_argument(
        "--save-baseline",
        action="store_true",
        help="overwrite this machine's committed baseline with this run -- use to accept "
        "a real, intentional regression; commit the resulting file change",
    )
    ap.add_argument("--no-gate", action="store_true", help="skip comparing against the committed baseline")
    ap.add_argument(
        "--min-regression-pct",
        type=float,
        default=5.0,
        help="practical-magnitude floor for a REGRESSED verdict -- CI raises this on "
        "shared/noisier GH-hosted runners; a quiet dedicated machine can use the default",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="independent process invocations per binary -- each binary already "
        "averages many internal calls per row, this is the OUTER repeat "
        "bench_repeated()'s own docstring describes, needed for gate()'s z-test "
        "(n=1 never regresses, compare() treats it as 'n/a')",
    )
    ap.add_argument("--quiet", action="store_true", help="pass/fail-only output")
    args = ap.parse_args()

    mkl_lib_dir, mkl_include_dir = find_mkl()
    if mkl_lib_dir is None:
        if not args.quiet:
            print("(no mkl/mkl-include pip package found -- DIDLDO/SIDLDO don't exist, skipping)")
        print("PASS (skipped, no MKL)" if args.quiet else "")
        return

    # key -> list of per-repeat ms values, collected across args.repeats
    # independent process invocations, then reduced to mean/std below.
    samples = {}
    with tempfile.TemporaryDirectory() as build_dir:
        for bench_name, (src, columns) in BENCHES.items():
            binary = build(bench_name, src, mkl_lib_dir, mkl_include_dir, build_dir)
            for rep in range(args.repeats):
                stdout = run_bench(binary)
                rows = parse_table(stdout, columns)
                if not args.quiet and rep == 0:
                    print(f"--- {bench_name} (repeat 1/{args.repeats} shown) ---")
                    print(stdout)
                for row in rows:
                    # bench_didldo_kernel.cpp/bench_sidldo_backward.cpp print
                    # multiple _ms columns per row (fwd_ms/bwd0_ms/bwdX_ms) --
                    # one flat key per metric, not just one per row.
                    for c in columns:
                        if c.endswith("_ms"):
                            tag = "_".join(f"{cc}{row[cc]:g}" for cc in columns if not cc.endswith("_ms"))
                            samples.setdefault(f"{bench_name}_{c[:-3]}_{tag}", []).append(row[c])

    flat = {
        key: {"mean": statistics.fmean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}
        for key, vals in samples.items()
    }

    meta = machine_info()

    if args.save_baseline:
        path = update_baseline(flat, meta, args.baseline_dir)
        if not args.quiet:
            print(f"\nSaved baseline to {path}")
        return

    if not args.no_gate:
        ok, regressions = gate(flat, meta, args.baseline_dir, min_pct=args.min_regression_pct, quiet=args.quiet)
        if not ok:
            if args.quiet:
                print(f"FAIL: {len(regressions)} statistically significant regression(s)")
            sys.exit(1)
    if args.quiet:
        print("PASS")


if __name__ == "__main__":
    main()
