#!/usr/bin/env python3
"""Coverage matrix: sili (disldo dense + sisldo sparse) vs torch, across
op x batch x density.

Answers three separate questions that a single fixed-shape benchmark
conflates:

1. Which op (fwd / bwd-compute-only / bwd+RMSprop-update) is furthest
   behind torch, and at which batch size -- see bench_sili_vs_torch.py for
   the deeper single-shape version of this.
2. Does sili's OWN sparse engine (sisldo) actually run faster than its own
   dense engine (disldo) at a given density, and at what density does that
   crossover happen (batch-size-dependent -- see docs/research/sisldo_ops.rst).
3. The one torch can't answer for itself: does sparsity ever let sili beat
   torch's dense BLAS matmul outright, not just beat sili's own dense path?
   torch has no fast general sparse-CSR matmul to compare against, so the
   only fair "does sparsity help in absolute terms" baseline is torch's
   plain dense compute at the same logical shape.

Both disldo and sisldo share the SAME underlying weights (load_dense_values
onto one DISLDOLayerV instance's storage) for a true apples-to-apples
comparison -- not separately-initialized layers. torch's backward always
computes dx (x.requires_grad_(True)) for the same reason
bench_sili_vs_torch.py does: sili's disldo_backward/sisldo backward_sparse
always compute dx regardless of learning_rate, so a fair comparison must
not let torch skip it.

Every measurement is repeated --repeats times (see _bench_stats.py) so
mean/std are real, not single-sample noise. Pass --save-baseline to record
this run as the new reference point (JSON, default under /tmp -- these
numbers are machine-specific, not meant to be committed); subsequent runs
against the same --baseline-path report IMPROVED/REGRESSED/noise per cell
via a two-sample z-test, not just a raw percentage.
"""

import argparse
import os
import tempfile

import numpy as np
import torch
from _bench_stats import bench_repeated, compare, load_baseline, machine_info, save_baseline

from sili import _cpu

DEFAULT_BASELINE_PATH = os.path.join(tempfile.gettempdir(), "sili_bench_matrix_baseline.json")


def n_calls_for(batch, calls_base):
    return max(20, calls_base // batch)


def dense_masked(batch, n, density, rng):
    """Dense [batch, n] array with round(density*n) nonzeros per row."""
    k = max(1, round(density * n))
    out = np.zeros((batch, n), dtype=np.float32)
    for b in range(batch):
        idx = rng.choice(n, size=k, replace=False)
        out[b, idx] = rng.standard_normal(k).astype(np.float32)
    return out


def to_csr(dense):
    batch, n = dense.shape
    ptrs = [0]
    indices, values = [], []
    for b in range(batch):
        nz = np.nonzero(dense[b])[0]
        indices.extend(nz.tolist())
        values.extend(dense[b, nz].tolist())
        ptrs.append(len(indices))
    return (
        np.array(ptrs, dtype=np.int32),
        np.array(indices, dtype=np.int32),
        np.array(values, dtype=np.float32),
    )


def make_layer(n_in, n_out, num_cpus, wvals, ivals):
    layer = _cpu.DISLDOLayerV(n_in, n_out, n_in * n_out, num_cpus)
    layer.load_dense_values(wvals, ivals)
    return layer


def run_cell(n_in, n_out, num_cpus, batch, density, calls_base, repeats, wvals, ivals, rng):
    n_calls = n_calls_for(batch, calls_base)
    x = dense_masked(batch, n_in, density, rng)
    dy = dense_masked(batch, n_out, density, rng)
    x_ptrs, x_idx, x_val = to_csr(x)
    dy_ptrs, dy_idx, dy_val = to_csr(dy)

    disldo_fwd = make_layer(n_in, n_out, num_cpus, wvals, ivals)
    fwd_disldo = bench_repeated(lambda: disldo_fwd.forward(x), n_calls, repeats)
    sisldo_fwd = make_layer(n_in, n_out, num_cpus, wvals, ivals)
    fwd_sisldo = bench_repeated(lambda: sisldo_fwd.forward_sparse(x_ptrs, x_idx, x_val, batch), n_calls, repeats)

    torch.manual_seed(0)
    tl = torch.nn.Linear(n_in, n_out, bias=False)
    xt = torch.from_numpy(x)
    xt.requires_grad_(True)
    dyt = torch.from_numpy(dy)
    with torch.no_grad():
        fwd_torch = bench_repeated(lambda: tl(xt), n_calls, repeats)

    disldo_bwd0 = make_layer(n_in, n_out, num_cpus, wvals, ivals)
    bwd0_disldo = bench_repeated(lambda: disldo_bwd0.backward(x, dy, 0.0, lr_per_row_nnz=True), n_calls, repeats)
    sisldo_bwd0 = make_layer(n_in, n_out, num_cpus, wvals, ivals)
    bwd0_sisldo = bench_repeated(
        lambda: sisldo_bwd0.backward_sparse(x, dy_ptrs, dy_idx, dy_val, batch, 0.0, lr_per_row_nnz=True),
        n_calls,
        repeats,
    )

    def torch_bwd0():
        if xt.grad is not None:
            xt.grad = None
        y = tl(xt)
        y.backward(dyt)

    bwd0_torch = bench_repeated(torch_bwd0, n_calls, repeats)

    disldo_bwdX = make_layer(n_in, n_out, num_cpus, wvals, ivals)
    bwdX_disldo = bench_repeated(lambda: disldo_bwdX.backward(x, dy, 1e-3, lr_per_row_nnz=True), n_calls, repeats)
    sisldo_bwdX = make_layer(n_in, n_out, num_cpus, wvals, ivals)
    bwdX_sisldo = bench_repeated(
        lambda: sisldo_bwdX.backward_sparse(x, dy_ptrs, dy_idx, dy_val, batch, 1e-3, lr_per_row_nnz=True),
        n_calls,
        repeats,
    )
    opt = torch.optim.RMSprop(tl.parameters(), lr=1e-3)

    def torch_bwdX():
        if xt.grad is not None:
            xt.grad = None
        opt.zero_grad()
        y = tl(xt)
        y.backward(dyt)
        opt.step()

    bwdX_torch = bench_repeated(torch_bwdX, n_calls, repeats)

    return {
        "fwd": {"disldo": fwd_disldo, "sisldo": fwd_sisldo, "torch": fwd_torch},
        "bwd0": {"disldo": bwd0_disldo, "sisldo": bwd0_sisldo, "torch": bwd0_torch},
        "bwdX": {"disldo": bwdX_disldo, "sisldo": bwdX_sisldo, "torch": bwdX_torch},
    }


def cell_key(op, batch, density, engine):
    return f"{op}|{batch}|{density:.3f}|{engine}"


def fmt(mean_std):
    mean, std = mean_std
    return f"{mean:>7.4f}±{std:<6.4f}"


def print_raw_tables(results, batches, densities, n_in):
    for density in densities:
        print(f"\n=== density={density:.3f} ({round(density * n_in)}/{n_in} nonzero per row) ===")
        print(
            f"{'batch':>6} | {'fwd disldo':>16} {'fwd sisldo':>16} {'fwd torch':>16} || "
            f"{'bwd0 disldo':>16} {'bwd0 sisldo':>16} {'bwd0 torch':>16} || "
            f"{'bwdX disldo':>16} {'bwdX sisldo':>16} {'bwdX torch':>16}"
        )
        for batch in batches:
            r = results[(batch, density)]
            print(
                f"{batch:>6} | {fmt(r['fwd']['disldo'])} {fmt(r['fwd']['sisldo'])} "
                f"{fmt(r['fwd']['torch'])} || "
                f"{fmt(r['bwd0']['disldo'])} {fmt(r['bwd0']['sisldo'])} {fmt(r['bwd0']['torch'])} || "
                f"{fmt(r['bwdX']['disldo'])} {fmt(r['bwdX']['sisldo'])} {fmt(r['bwdX']['torch'])}"
            )


def print_summary_table(results, batches, densities):
    print("\n=== Summary: where does sili need the most attention? ===")
    print(
        f"{'op':>5} {'batch':>6} {'density':>8} | {'best sili':>16} {'engine':>7} "
        f"{'torch':>16} | {'ratio':>7} {'sili wins?':>10}"
    )
    for op in ("fwd", "bwd0", "bwdX"):
        for batch in batches:
            for density in densities:
                cell = results[(batch, density)][op]
                d, s = cell["disldo"], cell["sisldo"]
                best, engine = (d, "disldo") if d[0] <= s[0] else (s, "sisldo")
                t = cell["torch"]
                ratio = best[0] / t[0]
                win = "YES" if ratio <= 1.0 else "no"
                print(
                    f"{op:>5} {batch:>6} {density:>8.3f} | {fmt(best):>16} {engine:>7} "
                    f"{fmt(t):>16} | {ratio:>6.2f}x {win:>10}"
                )


def flatten_results(results, repeats):
    flat = {}
    for (batch, density), ops in results.items():
        for op, engines in ops.items():
            for engine, (mean, std) in engines.items():
                flat[cell_key(op, batch, density, engine)] = {"mean": mean, "std": std, "n": repeats}
    return flat


def print_baseline_comparison(flat, baseline_path, meta):
    baseline_doc = load_baseline(baseline_path)
    if baseline_doc is None:
        print(f"(no baseline at {baseline_path} -- nothing to compare against)")
        return
    baseline = baseline_doc.get("results", baseline_doc)  # tolerate old flat-only format
    old_meta = baseline_doc.get("_meta")
    if old_meta and old_meta.get("machine_uid") != meta["machine_uid"]:
        print(
            f"WARNING: baseline was recorded on a different machine "
            f"({old_meta.get('cpu_model')}, uid={old_meta.get('machine_uid')}) -- "
            f"absolute times are not comparable across machines. Verdicts below are "
            f"unreliable; re-run with --save-baseline on THIS machine instead."
        )
    print(f"\n=== Compared against baseline: {baseline_path} ===")
    print(f"{'key':>32} | {'baseline':>10} {'now':>10} | {'change':>8} {'verdict':>10}")
    any_regressed = False
    for key in sorted(flat):
        if key not in baseline:
            continue
        new, old = flat[key], baseline[key]
        pct, verdict = compare(new["mean"], new["std"], new["n"], old["mean"], old["std"], old["n"])
        if verdict == "REGRESSED":
            any_regressed = True
        if verdict in ("REGRESSED", "IMPROVED"):
            print(f"{key:>32} | {old['mean']:>9.4f}ms {new['mean']:>9.4f}ms | {pct:>+7.1f}% {verdict:>10}")
    if not any_regressed:
        print("(no statistically significant regressions; unlisted cells were noise or n/a)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-in", type=int, default=288)
    ap.add_argument("--n-out", type=int, default=288)
    ap.add_argument("--num-cpus", type=int, default=4)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 256], help="low, high, ...")
    ap.add_argument("--densities", type=float, nargs="+", default=[1.0, 0.05], help="1.0=dense, low=sparse")
    ap.add_argument("--calls-base", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=5, help="independent timed loops per cell")
    ap.add_argument("--baseline-path", default=DEFAULT_BASELINE_PATH)
    ap.add_argument("--save-baseline", action="store_true", help="write this run as the new baseline")
    ap.add_argument("--no-compare", action="store_true", help="skip comparing against an existing baseline")
    args = ap.parse_args()

    rng_master = np.random.default_rng(0)
    wvals = rng_master.standard_normal(args.n_in * args.n_out).astype(np.float32) * 0.1
    ivals = np.zeros(args.n_in * args.n_out, dtype=np.float32)

    results = {}  # (batch, density) -> {op: {engine: (mean, std)}}
    for density in args.densities:
        for batch in args.batches:
            rng = np.random.default_rng(batch * 1000 + int(density * 100))
            results[(batch, density)] = run_cell(
                args.n_in,
                args.n_out,
                args.num_cpus,
                batch,
                density,
                args.calls_base,
                args.repeats,
                wvals,
                ivals,
                rng,
            )

    print_raw_tables(results, args.batches, args.densities, args.n_in)
    print_summary_table(results, args.batches, args.densities)

    flat = flatten_results(results, args.repeats)
    meta = machine_info()
    print(f"\nMachine: {meta['cpu_model']} [{','.join(meta['simd_flags'])}] uid={meta['machine_uid']}")

    if not args.no_compare:
        print_baseline_comparison(flat, args.baseline_path, meta)

    if args.save_baseline:
        save_baseline(args.baseline_path, {"_meta": meta, "results": flat})
        print(f"\nSaved baseline to {args.baseline_path}")


if __name__ == "__main__":
    main()
