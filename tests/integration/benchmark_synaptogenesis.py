"""
Benchmark: synaptogenesis cost vs. regular backprop, and why growth needs a
signal that rotates which positions look "most active" over time.

Answers a real question directly rather than estimating: is synaptogenesis
(build_probes + synap_step) hundreds of times slower than a regular
backward_dense() call, as informally recalled? At the CORRECT small-k scale
(k=4, after an earlier severe overcorrection that scaled k by row count was
found and reverted -- see git history), NO: confirmed ~14x at a realistic
size (n_in=1027, n_out=256), not ~500x. The earlier 500x-range estimate was
almost certainly describing the broken k_factor*n_in state (which made
build_probes evaluate k^2 candidates with k already scaled by row count --
a severe, unintended blowup), not the corrected small-k design.

SEPARATE, arguably more important finding this benchmark also demonstrates:
growing a layer via synaptogenesis driven by a RANDOM, undifferentiated
gradient signal stalls hard well below any target (confirmed: nnz plateaus
at 32 after 30 full synaptogenesis calls against a target of ~8216, and
does not move further with more calls). build_probes selects top-k inputs
by accumulated ACTIVITY and top-k outputs by accumulated GRADIENT -- if
those accumulators don't naturally shift which positions rank highest over
time, the same few positions get selected every single call and the rest
of the layer never gets a turn. This is exactly why the energy dynamics'
homeostatic firing (which guarantees every neuron eventually fires and
thus accumulates activity) matters for reliable full-layer growth in
practice, not just as a nice-to-have: a real task signal is not random, but
it still needs SOME rotating structure, and this benchmark shows what
happens when that's absent.

Run: python -m tests.integration.benchmark_synaptogenesis
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings('ignore')

import numpy as np
from sili import _cpu
import sili.cpu


def make_layer(n_in, n_out, max_weights=20000, target_per_row=8, lr=0.01):
    layer = _cpu.SparseLinearLayer(n_in, n_out, max_weights, 1)
    bits = max(1, n_out - 1).bit_length()
    layer.equalize_to_capacity(target_per_row, target_per_row * ((bits + 6) // 7) + 8)
    FP4_MAX = 6.0
    for r in range(n_in):
        layer.set_value_scale_raw(r, lr / FP4_MAX)
    return layer


def one_synaptogenesis_call(layer, k=4, cutoff=0.0, target_per_row=8):
    """Matches sili.sparse_rnn.FoldedLayer.synaptogenesis() exactly:
    build_probes ONCE, then synap_step once PER ROW (n_inputs calls) --
    NOT once total. Benchmarking synap_step alone without this loop
    understates the real per-call cost by roughly n_inputs-fold."""
    layer.build_probes(k)
    for _ in range(layer.n_inputs):
        layer.synap_step(cutoff, target_per_row)


def run(n_in=1027, n_out=256, target_per_row=8, k=4, lr=0.01,
       n_time=50, verbose=True):
    layer = make_layer(n_in, n_out, target_per_row=target_per_row, lr=lr)
    x = np.random.randn(1, n_in).astype(np.float32)

    if verbose:
        print(f"\n=== synaptogenesis benchmark: n_in={n_in} n_out={n_out} "
              f"target/row={target_per_row} k={k} ===")

    # -- Part 1: growth under a RANDOM (undifferentiated) gradient signal --
    # Demonstrates why growth needs a rotating signal, not just any signal.
    for i in range(30):
        _ = layer.forward_dense(x, lr)
        dy = np.random.randn(1, n_out).astype(np.float32)
        layer.backward_dense(dy, lr, True)
        one_synaptogenesis_call(layer, k, 0.0, target_per_row)
        if verbose and (i + 1) % 10 == 0:
            print(f"  after {i+1} synaptogenesis calls (random gradient): "
                  f"nnz={layer.nnz}")
    nnz_random = layer.nnz
    target_total = n_in * target_per_row
    if verbose:
        print(f"  final: nnz={nnz_random} vs target~{target_total} "
              f"({'STALLED -- ' if nnz_random < target_total * 0.5 else ''}"
              f"random gradient does not naturally rotate which positions "
              f"rank highest in the activity/gradient accumulators, so the "
              f"same few keep winning top-k selection every call)")

    # -- Part 2: timing comparison, regular backprop vs one full
    # synaptogenesis call, at the layer's CURRENT (realistic, partially
    # grown) nnz level --
    t0 = time.perf_counter()
    for _ in range(n_time):
        _ = layer.forward_dense(x, lr)
        dy = np.random.randn(1, n_out).astype(np.float32)
        layer.backward_dense(dy, lr, True)
    t_backprop = (time.perf_counter() - t0) / n_time

    t0 = time.perf_counter()
    for _ in range(n_time):
        one_synaptogenesis_call(layer, k, 0.0, target_per_row)
    t_synap = (time.perf_counter() - t0) / n_time

    ratio = t_synap / t_backprop
    if verbose:
        print(f"\n  regular backprop (forward_dense+backward_dense): "
              f"{t_backprop*1e6:.1f} us/call")
        print(f"  one synaptogenesis call (build_probes once + "
              f"synap_step x n_inputs={layer.n_inputs}): "
              f"{t_synap*1e6:.1f} us/call")
        print(f"  ratio: {ratio:.1f}x")
        print(f"  (an earlier, since-reverted k_factor*n_in scaling bug "
              f"made k scale by row count, and build_probes(k) evaluates "
              f"k_in*k_out candidates -- k^2, not k -- so that bug caused "
              f"roughly a further n_in^2-fold blowup on top of this ratio. "
              f"That is almost certainly what an earlier informal ~500x "
              f"estimate was describing, not this corrected small-k design.)")

    return dict(nnz_after_random_growth=nnz_random, target_total=target_total,
               backprop_us=t_backprop * 1e6, synap_us=t_synap * 1e6, ratio=ratio)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-in', type=int, default=1027)
    ap.add_argument('--n-out', type=int, default=256)
    ap.add_argument('--target-per-row', type=int, default=8)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--lr', type=float, default=0.01)
    a = ap.parse_args()
    run(a.n_in, a.n_out, a.target_per_row, a.k, a.lr)
