#!/usr/bin/env python3
"""Fair sili-vs-torch backward+optimizer timing sweep, dense fp32 block4.

The naive "torch backward()" baseline is NOT fair to compare against sili's
backward: torch's benchmark input `x` has no `requires_grad`, so autograd
never computes dL/dx, only dL/dW -- while sili's disldo_backward ALWAYS
computes dx (needed for BPTT-style recurrent chains) regardless of
`learning_rate`. Comparing sili's dx+dW+update cost against torch's dW-only
cost overstates the gap. This script forces `x.requires_grad_(True)` on the
torch side so both frameworks compute the same three things: dx, dW, and
(when `mode="plus_step"`) one optimizer step -- torch's own
`torch.optim.RMSprop`, since sili's backward fuses an RMSprop-family update
in place rather than exposing a separate step().

This is the DEFAULT/gating comparison for this project going forward --
superseding any earlier sweep that only measured torch's dW-only backward.
Gates CI on sili's OWN measurements regressing (see .github/workflows/
ci.yml's perf-gate job) -- torch's numbers are recorded and printed for
context but never gate (a torch version bump making torch itself slower on
the runner isn't a sili code regression). Run standalone:
`python -m tests.integration.test_perf_disldo_vs_torch` (add --quiet for
pass/fail-only output).
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _bench_stats import bench_repeated, gate, machine_info, update_baseline

from sili.sparse_rnn import DISLDOLayer32


def n_calls_for(batch, calls_base):
    return max(50, calls_base // batch)


def bench_sili_fwd(n_in, n_out, num_cpus, batch, calls_base, repeats):
    n_calls = n_calls_for(batch, calls_base)
    rng = np.random.default_rng(0)
    layer = DISLDOLayer32(n_in, n_out, n_in * n_out, num_cpus, rng=rng, dense=True)
    x = rng.standard_normal((batch, n_in)).astype(np.float32)
    return bench_repeated(lambda: layer._c.forward(x), n_calls, repeats)


def bench_sili_bwd(n_in, n_out, num_cpus, batch, calls_base, lr, repeats):
    n_calls = n_calls_for(batch, calls_base)
    rng = np.random.default_rng(0)
    layer = DISLDOLayer32(n_in, n_out, n_in * n_out, num_cpus, rng=rng, dense=True)
    x = rng.standard_normal((batch, n_in)).astype(np.float32)
    dy = rng.standard_normal((batch, n_out)).astype(np.float32)
    layer._c.forward(x)
    return bench_repeated(
        lambda: layer._c.backward(x, dy, lr, lr_per_row_nnz=True, damp_by_importance=True),
        n_calls,
        repeats,
    )


def bench_torch_fwd(n_in, n_out, batch, calls_base, repeats):
    n_calls = n_calls_for(batch, calls_base)
    torch.manual_seed(0)
    layer = torch.nn.Linear(n_in, n_out, bias=False)
    x = torch.randn(batch, n_in)
    with torch.no_grad():
        return bench_repeated(lambda: layer(x), n_calls, repeats)


def bench_torch_bwd(n_in, n_out, batch, calls_base, mode, repeats):
    """mode: 'backward_only' or 'backward_plus_step'. Always computes dx AND
    dW (x.requires_grad_(True)) -- see module docstring."""
    n_calls = n_calls_for(batch, calls_base)
    torch.manual_seed(0)
    layer = torch.nn.Linear(n_in, n_out, bias=False)
    opt = torch.optim.RMSprop(layer.parameters(), lr=1e-3)
    x = torch.randn(batch, n_in)
    x.requires_grad_(True)

    def one_call():
        opt.zero_grad()
        if x.grad is not None:
            x.grad = None
        y = layer(x)
        y.sum().backward()
        if mode == "backward_plus_step":
            opt.step()

    return bench_repeated(one_call, n_calls, repeats)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-in", type=int, default=288)
    ap.add_argument("--n-out", type=int, default=288)
    ap.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="defaults to torch.get_num_threads() -- torch's own thread count is fixed by "
        "its runtime regardless of this flag, so giving sili fewer threads than torch "
        "actually uses silently biases the comparison in torch's favor",
    )
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    ap.add_argument("--calls-base", type=int, default=2000, help="n_calls = max(50, calls_base // batch)")
    ap.add_argument("--repeats", type=int, default=5, help="independent timed loops per cell")
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
        help="practical-magnitude floor for a REGRESSED verdict (see compare() in "
        "_bench_stats.py) -- CI raises this on shared/noisier GH-hosted runners; a "
        "quiet dedicated machine can use the default",
    )
    ap.add_argument("--quiet", action="store_true", help="pass/fail-only output")
    args = ap.parse_args()
    if args.num_cpus is None:
        args.num_cpus = torch.get_num_threads()

    if not args.quiet:
        print(
            f"{args.n_in}x{args.n_out} dense fp32, NUM_CPUS={args.num_cpus} "
            f"(matches torch.get_num_threads(); torch backward includes dx: "
            f"x.requires_grad_(True))\n"
        )
        print(
            f"{'batch':>6} | {'sili fwd':>10} {'torch fwd':>10} {'ratio':>7} || "
            f"{'sili bwd lr0':>13} {'torch bwd(dx+dW)':>17} {'ratio':>7} || "
            f"{'sili bwd lrX':>13} {'torch bwd+step':>15} {'ratio':>7}"
        )
    flat = {}
    for b in args.batches:
        sf, sf_std = bench_sili_fwd(args.n_in, args.n_out, args.num_cpus, b, args.calls_base, args.repeats)
        tf, _ = bench_torch_fwd(args.n_in, args.n_out, b, args.calls_base, args.repeats)
        s0, s0_std = bench_sili_bwd(args.n_in, args.n_out, args.num_cpus, b, args.calls_base, 0.0, args.repeats)
        t0, _ = bench_torch_bwd(args.n_in, args.n_out, b, args.calls_base, "backward_only", args.repeats)
        sX, sX_std = bench_sili_bwd(args.n_in, args.n_out, args.num_cpus, b, args.calls_base, 1e-3, args.repeats)
        tX, _ = bench_torch_bwd(args.n_in, args.n_out, b, args.calls_base, "backward_plus_step", args.repeats)
        if not args.quiet:
            print(
                f"{b:>6} | {sf:>9.4f}ms {tf:>9.4f}ms {sf / tf:>6.2f}x || "
                f"{s0:>12.4f}ms {t0:>16.4f}ms {s0 / t0:>6.2f}x || "
                f"{sX:>12.4f}ms {tX:>14.4f}ms {sX / tX:>6.2f}x"
            )
        # Only sili_* keys gate CI -- torch_* is recorded/printed for
        # context but never fails the build (see module docstring).
        flat[f"sili_fwd_batch{b}"] = {"mean": sf, "std": sf_std, "n": args.repeats}
        flat[f"sili_bwd0_batch{b}"] = {"mean": s0, "std": s0_std, "n": args.repeats}
        flat[f"sili_bwdX_batch{b}"] = {"mean": sX, "std": sX_std, "n": args.repeats}
        flat[f"torch_fwd_batch{b}"] = {"mean": tf, "std": 0.0, "n": args.repeats}
        flat[f"torch_bwd0_batch{b}"] = {"mean": t0, "std": 0.0, "n": args.repeats}
        flat[f"torch_bwdX_batch{b}"] = {"mean": tX, "std": 0.0, "n": args.repeats}

    meta = machine_info()
    meta["num_cpus"] = args.num_cpus

    if args.save_baseline:
        path = update_baseline(flat, meta, args.baseline_dir)
        if not args.quiet:
            print(f"\nSaved baseline to {path}")
        return

    if not args.no_gate:
        _, regressions = gate(flat, meta, args.baseline_dir, min_pct=args.min_regression_pct, quiet=args.quiet)
        sili_regressions = [r for r in regressions if r[0].startswith("sili_")]
        if sili_regressions:
            if args.quiet:
                print(f"FAIL: {len(sili_regressions)} statistically significant sili regression(s)")
            else:
                print(f"\n{len(sili_regressions)} sili regression(s) (torch-only changes don't gate):")
                for key, old, new, pct in sili_regressions:
                    print(f"  {key}: {old:.4f}ms -> {new:.4f}ms ({pct:+.1f}%)")
            sys.exit(1)
    if args.quiet:
        print("PASS")


if __name__ == "__main__":
    main()
