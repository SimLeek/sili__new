#!/usr/bin/env python3
"""Benchmark + correctness probe for sili._cpu.SparseLinearLayer.

Runs unmodified against ANY installed `sili` build -- the pre-block4
baseline or this branch's block4-enabled version -- since block4 is purely
an internal representation change with no new Python API surface. Intended
to be run once per venv (see compare_block4_venvs.sh), which diffs the two
JSON reports.

Reports: forward/backward timing before and after a growth (synaptogenesis)
phase, nnz before/after growth, and a quality check (max abs error of
forward_dense(learning_rate=0) against a dense-matmul reference
reconstructed from get_ptrs/get_indices/weights_vals) at both points --
proving output correctness is unaffected by whatever representation
(scattered CSR vs, on this branch, block4 tiles) currently holds a given
synapse.
"""
import argparse
import json
import statistics
import sys
import time

import numpy as np

import sili._cpu as _cpu


def dense_reference(layer, n_in, n_out, ptrs, indices, weights):
    # true_w = stored_w * value_scale[row] * output_scale[col] -- see
    # linear_disldo.hpp's disldo_forward. weights_vals returns STORED
    # values; once any training with learning_rate != 0 has moved
    # value_scale/output_scale away from their 1.0 default, comparing
    # stored values directly against forward_dense's actual output would
    # show a real-looking but spurious "error" that's really just this
    # scale factor, not a correctness problem.
    dense = np.zeros((n_in, n_out), dtype=np.float32)
    row_scale = np.array([layer.get_value_scale(r) for r in range(n_in)], dtype=np.float32)
    col_scale = np.array([layer.get_output_scale(c) for c in range(n_out)], dtype=np.float32)
    for r in range(n_in):
        for k in range(ptrs[r], ptrs[r + 1]):
            c = indices[k]
            dense[r, c] = weights[k] * row_scale[r] * col_scale[c]
    return dense


def time_calls(fn, n_calls, warmup):
    # Warmup discarded before timing: first calls can carry one-time costs
    # (allocator growth, cache cold-start) unrelated to steady-state speed.
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-in", type=int, default=512)
    ap.add_argument("--n-out", type=int, default=512)
    ap.add_argument("--density", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-cpus", type=int, default=1,
                     help="1 (default) for a fully deterministic nnz/quality "
                     "comparison -- backward_dense's OpenMP reduction has "
                     "genuine thread-scheduling-order floating-point "
                     "nondeterminism (pre-existing, confirmed present on the "
                     "pre-block4 baseline too: two same-branch runs at "
                     "num_cpus>1 can already disagree by a few grown/pruned "
                     "synapses purely from reduction order, nothing to do "
                     "with block4). Pass a higher value to measure realistic "
                     "multi-threaded timing instead; expect nnz_after_growth "
                     "to then legitimately differ run-to-run.")
    ap.add_argument("--n-calls", type=int, default=50)
    ap.add_argument("--warmup-calls", type=int, default=5)
    ap.add_argument("--growth-cycles", type=int, default=200,
                     help="number of synap_step calls in the growth phase")
    ap.add_argument("--probe-k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Seed FP4's stochastic rounding RNG too -- disldo_backward's
    # gradient-driven weight/importance updates go through
    # fp4_quantize_stochastic() (unbiased-on-average by design, so it's
    # genuinely, intentionally randomized), not the deterministic
    # fp4_quantize(). Leaving it unseeded makes nnz_after_growth /
    # quality_after_growth differ run-to-run even for two runs of the
    # exact same build -- confirmed by direct comparison; nothing to do
    # with block4 or threading. Seed it for a reproducible comparison.
    _cpu.seed_fp4_stochastic_rng(args.seed)
    rng = np.random.default_rng(args.seed)
    n_in, n_out = args.n_in, args.n_out

    # Build an initial random sparse layer at the requested density.
    ptrs = [0]
    idx = []
    vals = []
    nnz_per_row = max(1, int(n_out * args.density))
    for r in range(n_in):
        cols = rng.choice(n_out, size=nnz_per_row, replace=False)
        cols.sort()
        idx.extend(int(c) for c in cols)
        vals.extend(float(v) for v in rng.uniform(-2.0, 2.0, size=nnz_per_row))
        ptrs.append(len(idx))
    ptrs = np.array(ptrs, dtype=np.int32)
    idx = np.array(idx, dtype=np.int32)
    vals = np.array(vals, dtype=np.float32)

    max_weights = int(len(idx) * 4)  # generous headroom for growth
    layer = _cpu.SparseLinearLayer(n_in, n_out, max_weights, args.num_cpus)
    layer.load_weights(ptrs, idx, vals)
    # One-time construction call: gives every row real per-row headroom for
    # the growth phase below, rather than relying on delta_csr_from_absolute's
    # small default blank_fraction (proportional to that row's OWN size --
    # tiny for a lightly-connected row).
    layer.equalize_to_capacity(nnz_per_row * 3)

    x = rng.uniform(-1.0, 1.0, size=(args.batch, n_in)).astype(np.float32)
    dy = rng.uniform(-1.0, 1.0, size=(args.batch, n_out)).astype(np.float32)

    report = {"sili_version": getattr(__import__("sili"), "__version__", "unknown"),
              "args": vars(args)}

    report["nnz_before_growth"] = int(layer.nnz)
    report["block4_tiles_before_growth"] = int(getattr(layer, 'block4_tiles', 0))
    report["block4_synapses_before_growth"] = int(getattr(layer, 'block4_synapses', 0))

    # Quality check BEFORE growth.
    dense0 = dense_reference(layer, n_in, n_out, layer.ptrs, layer.indices, layer.weights_vals)
    y0 = layer.forward_dense(x, learning_rate=0.0)
    ref0 = x @ dense0
    report["quality_before_growth_max_abs_err"] = float(np.max(np.abs(y0 - ref0)))

    report["forward_before_growth"] = time_calls(
        lambda: layer.forward_dense(x, learning_rate=0.0), args.n_calls, args.warmup_calls)

    y_for_bwd = layer.forward_dense(x, learning_rate=0.0)
    report["backward_before_growth"] = time_calls(
        lambda: layer.backward_dense(dy, learning_rate=0.01), args.n_calls, args.warmup_calls)

    # Growth phase: build probes once, then repeatedly synap_step to insert
    # them row by row (this is what triggers block4 promotion on branches
    # that have it -- a no-op-equivalent structural change on baselines
    # that don't).
    t_growth0 = time.perf_counter()
    layer.build_probes(args.probe_k, True)  # per_row=True: candidates for every row
    n_throws = 0
    for _ in range(args.growth_cycles):
        try:
            layer.synap_step(0.0, nnz_per_row + 4)
        except RuntimeError:
            n_throws += 1
            layer.equalizer_step()
    report["growth_phase_s"] = time.perf_counter() - t_growth0
    report["growth_phase_throws"] = n_throws

    report["nnz_after_growth"] = int(layer.nnz)
    report["block4_tiles_after_growth"] = int(getattr(layer, 'block4_tiles', 0))
    report["block4_synapses_after_growth"] = int(getattr(layer, 'block4_synapses', 0))

    # Quality check AFTER growth -- must still match a dense reference
    # reconstructed from the (block4-aware, see delta_csr_combined_to_absolute)
    # combined ptrs/indices/weights_vals.
    dense1 = dense_reference(layer, n_in, n_out, layer.ptrs, layer.indices, layer.weights_vals)
    y1 = layer.forward_dense(x, learning_rate=0.0)
    ref1 = x @ dense1
    report["quality_after_growth_max_abs_err"] = float(np.max(np.abs(y1 - ref1)))

    report["forward_after_growth"] = time_calls(
        lambda: layer.forward_dense(x, learning_rate=0.0), args.n_calls, args.warmup_calls)
    report["backward_after_growth"] = time_calls(
        lambda: layer.backward_dense(dy, learning_rate=0.01), args.n_calls, args.warmup_calls)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
