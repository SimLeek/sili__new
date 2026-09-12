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
"""

import argparse
import time

import numpy as np
import torch

from sili import _cpu


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


def bench(fn, n_calls, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        fn()
    return (time.perf_counter() - t0) / n_calls * 1000.0


def run_cell(n_in, n_out, num_cpus, batch, density, calls_base, wvals, ivals, rng):
    n_calls = n_calls_for(batch, calls_base)
    x = dense_masked(batch, n_in, density, rng)
    dy = dense_masked(batch, n_out, density, rng)
    x_ptrs, x_idx, x_val = to_csr(x)
    dy_ptrs, dy_idx, dy_val = to_csr(dy)

    disldo_fwd = make_layer(n_in, n_out, num_cpus, wvals, ivals)
    t_disldo_fwd = bench(lambda: disldo_fwd.forward(x), n_calls)
    sisldo_fwd = make_layer(n_in, n_out, num_cpus, wvals, ivals)
    t_sisldo_fwd = bench(lambda: sisldo_fwd.forward_sparse(x_ptrs, x_idx, x_val, batch), n_calls)

    torch.manual_seed(0)
    tl = torch.nn.Linear(n_in, n_out, bias=False)
    xt = torch.from_numpy(x)
    xt.requires_grad_(True)
    dyt = torch.from_numpy(dy)
    with torch.no_grad():
        t_torch_fwd = bench(lambda: tl(xt), n_calls)

    disldo_bwd0 = make_layer(n_in, n_out, num_cpus, wvals, ivals)
    t_disldo_bwd0 = bench(lambda: disldo_bwd0.backward(x, dy, 0.0, lr_per_row_nnz=True), n_calls)
    sisldo_bwd0 = make_layer(n_in, n_out, num_cpus, wvals, ivals)
    t_sisldo_bwd0 = bench(
        lambda: sisldo_bwd0.backward_sparse(x, dy_ptrs, dy_idx, dy_val, batch, 0.0, lr_per_row_nnz=True),
        n_calls,
    )

    def torch_bwd0():
        if xt.grad is not None:
            xt.grad = None
        y = tl(xt)
        y.backward(dyt)

    t_torch_bwd0 = bench(torch_bwd0, n_calls)

    disldo_bwdX = make_layer(n_in, n_out, num_cpus, wvals, ivals)
    t_disldo_bwdX = bench(lambda: disldo_bwdX.backward(x, dy, 1e-3, lr_per_row_nnz=True), n_calls)
    sisldo_bwdX = make_layer(n_in, n_out, num_cpus, wvals, ivals)
    t_sisldo_bwdX = bench(
        lambda: sisldo_bwdX.backward_sparse(x, dy_ptrs, dy_idx, dy_val, batch, 1e-3, lr_per_row_nnz=True),
        n_calls,
    )
    opt = torch.optim.RMSprop(tl.parameters(), lr=1e-3)

    def torch_bwdX():
        if xt.grad is not None:
            xt.grad = None
        opt.zero_grad()
        y = tl(xt)
        y.backward(dyt)
        opt.step()

    t_torch_bwdX = bench(torch_bwdX, n_calls)

    return {
        "fwd": (t_disldo_fwd, t_sisldo_fwd, t_torch_fwd),
        "bwd0": (t_disldo_bwd0, t_sisldo_bwd0, t_torch_bwd0),
        "bwdX": (t_disldo_bwdX, t_sisldo_bwdX, t_torch_bwdX),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-in", type=int, default=288)
    ap.add_argument("--n-out", type=int, default=288)
    ap.add_argument("--num-cpus", type=int, default=4)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 256], help="low, high, ...")
    ap.add_argument("--densities", type=float, nargs="+", default=[1.0, 0.05], help="1.0=dense, low=sparse")
    ap.add_argument("--calls-base", type=int, default=1000)
    args = ap.parse_args()

    rng_master = np.random.default_rng(0)
    wvals = rng_master.standard_normal(args.n_in * args.n_out).astype(np.float32) * 0.1
    ivals = np.zeros(args.n_in * args.n_out, dtype=np.float32)

    results = {}  # (batch, density) -> dict
    for density in args.densities:
        print(f"\n=== density={density:.3f} ({round(density * args.n_in)}/{args.n_in} nonzero per row) ===")
        print(
            f"{'batch':>6} | {'fwd disldo':>10} {'fwd sisldo':>10} {'fwd torch':>10} || "
            f"{'bwd0 disldo':>11} {'bwd0 sisldo':>11} {'bwd0 torch':>10} || "
            f"{'bwdX disldo':>11} {'bwdX sisldo':>11} {'bwdX torch':>10}"
        )
        for batch in args.batches:
            rng = np.random.default_rng(batch * 1000 + int(density * 100))
            r = run_cell(args.n_in, args.n_out, args.num_cpus, batch, density, args.calls_base, wvals, ivals, rng)
            results[(batch, density)] = r
            fd, fs, ft = r["fwd"]
            b0d, b0s, b0t = r["bwd0"]
            bXd, bXs, bXt = r["bwdX"]
            print(
                f"{batch:>6} | {fd:>9.4f}ms {fs:>9.4f}ms {ft:>9.4f}ms || "
                f"{b0d:>10.4f}ms {b0s:>10.4f}ms {b0t:>9.4f}ms || "
                f"{bXd:>10.4f}ms {bXs:>10.4f}ms {bXt:>9.4f}ms"
            )

    # Condensed attention table: for each (op, batch), best sili engine vs torch,
    # and whether going sparse ever beats torch outright (not just beats disldo).
    print("\n=== Summary: where does sili need the most attention? ===")
    print(
        f"{'op':>5} {'batch':>6} {'density':>8} | {'best sili':>9} {'engine':>7} "
        f"{'torch':>9} | {'ratio':>7} {'sili wins?':>10}"
    )
    for op, label in [("fwd", "fwd"), ("bwd0", "bwd0"), ("bwdX", "bwdX")]:
        for batch in args.batches:
            for density in args.densities:
                d, s, t = results[(batch, density)][op]
                best, engine = (d, "disldo") if d <= s else (s, "sisldo")
                ratio = best / t
                win = "YES" if ratio <= 1.0 else "no"
                print(
                    f"{label:>5} {batch:>6} {density:>8.3f} | {best:>8.4f}ms {engine:>7} "
                    f"{t:>8.4f}ms | {ratio:>6.2f}x {win:>10}"
                )


if __name__ == "__main__":
    main()
