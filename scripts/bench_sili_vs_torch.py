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
"""

import argparse
import time

import numpy as np
import torch

from sili.sparse_rnn import DISLDOLayer32


def n_calls_for(batch, calls_base):
    return max(50, calls_base // batch)


def bench_sili_fwd(n_in, n_out, num_cpus, batch, calls_base):
    n_calls = n_calls_for(batch, calls_base)
    rng = np.random.default_rng(0)
    layer = DISLDOLayer32(n_in, n_out, n_in * n_out, num_cpus, rng=rng, dense=True)
    x = rng.standard_normal((batch, n_in)).astype(np.float32)
    for _ in range(10):
        layer._c.forward(x)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        layer._c.forward(x)
    t1 = time.perf_counter()
    return (t1 - t0) / n_calls


def bench_sili_bwd(n_in, n_out, num_cpus, batch, calls_base, lr):
    n_calls = n_calls_for(batch, calls_base)
    rng = np.random.default_rng(0)
    layer = DISLDOLayer32(n_in, n_out, n_in * n_out, num_cpus, rng=rng, dense=True)
    x = rng.standard_normal((batch, n_in)).astype(np.float32)
    dy = rng.standard_normal((batch, n_out)).astype(np.float32)
    for _ in range(10):
        layer._c.forward(x)
        layer._c.backward(x, dy, lr, lr_per_row_nnz=True, damp_by_importance=True)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        layer._c.backward(x, dy, lr, lr_per_row_nnz=True, damp_by_importance=True)
    t1 = time.perf_counter()
    return (t1 - t0) / n_calls


def bench_torch_fwd(n_in, n_out, batch, calls_base):
    n_calls = n_calls_for(batch, calls_base)
    torch.manual_seed(0)
    layer = torch.nn.Linear(n_in, n_out, bias=False)
    x = torch.randn(batch, n_in)
    with torch.no_grad():
        for _ in range(10):
            layer(x)
        t0 = time.perf_counter()
        for _ in range(n_calls):
            layer(x)
        t1 = time.perf_counter()
    return (t1 - t0) / n_calls


def bench_torch_bwd(n_in, n_out, batch, calls_base, mode):
    """mode: 'backward_only' or 'backward_plus_step'. Always computes dx AND
    dW (x.requires_grad_(True)) -- see module docstring."""
    n_calls = n_calls_for(batch, calls_base)
    torch.manual_seed(0)
    layer = torch.nn.Linear(n_in, n_out, bias=False)
    opt = torch.optim.RMSprop(layer.parameters(), lr=1e-3)
    x = torch.randn(batch, n_in)
    x.requires_grad_(True)
    for _ in range(10):
        opt.zero_grad()
        if x.grad is not None:
            x.grad = None
        y = layer(x)
        y.sum().backward()
        if mode == "backward_plus_step":
            opt.step()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        opt.zero_grad()
        if x.grad is not None:
            x.grad = None
        y = layer(x)
        y.sum().backward()
        if mode == "backward_plus_step":
            opt.step()
    t1 = time.perf_counter()
    return (t1 - t0) / n_calls


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-in", type=int, default=288)
    ap.add_argument("--n-out", type=int, default=288)
    ap.add_argument("--num-cpus", type=int, default=4)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    ap.add_argument("--calls-base", type=int, default=2000, help="n_calls = max(50, calls_base // batch)")
    args = ap.parse_args()

    print(
        f"{args.n_in}x{args.n_out} dense fp32, NUM_CPUS={args.num_cpus} "
        f"(torch backward includes dx: x.requires_grad_(True))\n"
    )
    print(
        f"{'batch':>6} | {'sili fwd':>10} {'torch fwd':>10} {'ratio':>7} || "
        f"{'sili bwd lr0':>13} {'torch bwd(dx+dW)':>17} {'ratio':>7} || "
        f"{'sili bwd lrX':>13} {'torch bwd+step':>15} {'ratio':>7}"
    )
    for b in args.batches:
        sf = bench_sili_fwd(args.n_in, args.n_out, args.num_cpus, b, args.calls_base)
        tf = bench_torch_fwd(args.n_in, args.n_out, b, args.calls_base)
        s0 = bench_sili_bwd(args.n_in, args.n_out, args.num_cpus, b, args.calls_base, 0.0)
        t0 = bench_torch_bwd(args.n_in, args.n_out, b, args.calls_base, "backward_only")
        sX = bench_sili_bwd(args.n_in, args.n_out, args.num_cpus, b, args.calls_base, 1e-3)
        tX = bench_torch_bwd(args.n_in, args.n_out, b, args.calls_base, "backward_plus_step")
        print(
            f"{b:>6} | {sf * 1000:>9.4f}ms {tf * 1000:>9.4f}ms {sf / tf:>6.2f}x || "
            f"{s0 * 1000:>12.4f}ms {t0 * 1000:>16.4f}ms {s0 / t0:>6.2f}x || "
            f"{sX * 1000:>12.4f}ms {tX * 1000:>14.4f}ms {sX / tX:>6.2f}x"
        )


if __name__ == "__main__":
    main()
