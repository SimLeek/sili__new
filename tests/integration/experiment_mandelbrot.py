"""
Long-running experiment harness for test_mandelbrot_rl.

PURPOSE (per direct request): "We know these functions are useful, but we
need to know exactly HOW they're useful." The energy-*.md docs
(energy-params.md, energy-personality.md, energy-proofs.md) make specific,
testable claims about what each parameter axis does -- this file turns
those into overnight-runnable experiment grids with measured, plotted
outcomes instead of predictions. The docs' claims also explicitly mix with
how LEARNED a network is, so 'learnedness' (steps, and the metrics that
track it: nnz, recon, probe, compression) is itself an axis, not a
nuisance variable.

DESIGN:
  - Each run is a SUBPROCESS invoking test_mandelbrot_rl's CLI with
    --json-out (true process parallelism -- deliberately subprocess rather
    than threads/OpenMP: separate processes are fully parallel and a crash
    in one run cannot take down the harness or the other runs).
  - test_mandelbrot_rl writes its JSON atomically at EVERY report interval,
    so partial results survive crashes/power loss and can be plotted
    mid-run.
  - Results land in --outdir as one JSON per run, named by experiment and
    config. Re-running skips completed runs (done=True in the JSON), so
    the harness is resumable and multiple machines can share a directory
    via any file sync and merge trivially.
  - `--plot` reads whatever JSONs exist (complete or partial) and produces
    PNGs per experiment: metric-vs-steps curves per config, plus
    final-metric-vs-parameter summaries.

USAGE:
  # list experiments and their run counts
  python -m tests.integration.experiment_mandelbrot --list

  # run one experiment group, 3 subprocesses in parallel, overnight budget
  python -m tests.integration.experiment_mandelbrot \\
      --experiment energy_exploration --parallel 3 \\
      --steps 500000 --timeout 28800 --outdir results/exp1

  # plot whatever has finished (or partially finished) so far
  python -m tests.integration.experiment_mandelbrot \\
      --experiment energy_exploration --outdir results/exp1 --plot

Multiple machines: give each a different --experiment (or --shard i/n on
the same experiment), point at machine-local outdirs, and copy the JSONs
into one directory afterward; --plot only reads JSONs, so merged plotting
needs no coordination.

WHAT THE METRICS MEAN (all recorded per report interval by
test_mandelbrot_rl, definitions there):
  probe          held-out reconstruction MSE (8 fixed views, never trained)
  recon          online next-view reconstruction MSE (windowed)
  nnz            total grown synapses (sparse core only) -- structure size
  fire_frac      mean fraction of hidden neurons active -- compare directly
                 against the energy density/p settings (Theorem 3 premise)
  vcr / hcr      zlib compression ratio of the view / hidden-state windows
                 (joint-entropy proxies; lower = more redundant/compressible)
  ncd_view_hidden  Normalized Compression Distance between the two windows:
                 a cheap MI proxy for whether the hidden code is ABOUT the
                 input manifold or about private dynamics (Theorems 3/5 --
                 the column-energy idea targets exactly this failure mode).
                 zlib-NCD is an approximation and can exceed 1; values are
                 comparable ACROSS runs, which is what grids need.
  coverage       distinct (x, y, log-zoom) cells visited -- exploration
  mean_mag       per-dimension continuous action magnitudes (continuous
                 mode) -- 'personality' expression at the behavior level

Hypothesis -> experiment mapping (from the energy docs):
  energy-params.md Axis 3 (sigma, exploration/noise): 'high sigma =
    spontaneous exploration not driven by input' -> energy_exploration grid:
    expect coverage up with sigma, and ncd_view_hidden up (hidden less
    input-locked) at high sigma.
  Axis 1/2 (delta/gamma ratio sets target |h| and tempo) -> energy_drive
    and energy_activation_cost grids: expect fire_frac and hcr to track
    delta/gamma; recon quality should peak at moderate ratios.
  Axes 6-8 (beta/lambda_KL: sparsity -> representational independence,
    Theorem 3) -> energy_sparsity grid over (density, precision, p):
    expect hcr to FALL (less redundancy) as enforced sparsity rises, until
    capacity starvation shows up as probe MSE rising.
  Personality doc (tau/alpha as N-like regulation) -> energy_regulation
    grid over (setpoint, reactivity): expression should show up in
    mean_mag dynamics and coverage, not necessarily in recon.
  'Parameters mix with how learned the net is' -> every grid runs long
    with full history, so the SAME config can be compared early vs late;
    the learnedness experiment additionally sweeps seeds at one config to
    separate config-effect from run-variance.
  'Sparse nets need to be larger; fan-out should change with neuron count'
    -> shape_vs_k grid: hidden x base_connections, same step budget,
    watching probe/nnz/fire_frac. (The 50x-at-same-compute claim is
    testable here up to the point where the DENSE heads (Wa, value heads)
    start dominating step cost -- they scale with hidden, and are the
    known bottleneck for very large hidden until they're sparsified too;
    see refactoring_todo.md.)

Run: python -m tests.integration.experiment_mandelbrot --list
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

# ── Experiment definitions ────────────────────────────────────────────────────
# Each experiment: list of (run_name, {cli_flag: value}) pairs. Flags are
# passed to test_mandelbrot_rl verbatim (see its --help). Step budget /
# timeout / core / action-mode come from the harness CLI so the same grid
# can be run cheap (smoke) or overnight without editing this file.


def _grid(prefix, axis_name, values, extra=None):
    out = []
    for v in values:
        cfg = dict(extra or {})
        cfg[axis_name] = v
        out.append((f"{prefix}_{str(v).replace('.', 'p')}", cfg))
    return out


EXPERIMENTS = {
    # Axis 3 (sigma): spontaneous exploration vs input-driven behavior
    'energy_exploration': _grid('sigma', '--energy-exploration',
                                [0.0, 0.002, 0.01, 0.05, 0.2]),
    # Axis 1 (delta): metabolic tempo
    'energy_drive': _grid('delta', '--energy-drive',
                          [0.0002, 0.001, 0.005, 0.02]),
    # Axis 2 (gamma): cost of sustained activation (delta/gamma sets |h*|)
    'energy_activation_cost': _grid('gamma', '--energy-activation-cost',
                                    [0.02, 0.05, 0.15, 0.4]),
    # Axes 6-8: sparsity -> independence (Theorem 3). Joint sweep of the
    # enforcement strength and the hard ceiling.
    'energy_sparsity': [
        (f"beta{str(d).replace('.','p')}_p{str(p).replace('.','p')}",
         {'--energy-density': d, '--energy-p': p})
        for d, p in itertools.product([0.02, 0.05, 0.15], [0.04, 0.1, 0.3])
    ],
    # tau/alpha regulation ('personality' axes)
    'energy_regulation': [
        (f"tau{str(t).replace('.','p')}_alpha{str(a).replace('.','p')}",
         {'--energy-setpoint': t, '--energy-reactivity': a})
        for t, a in itertools.product([0.5, 1.0, 1.5], [0.002, 0.01, 0.05])
    ],
    # network shape vs per-row fan-out (sparse scaling claims)
    'shape_vs_k': [
        (f"h{h}_k{k}", {'--hidden': h, '--base-connections': k})
        for h, k in itertools.product([256, 1024, 4096], [3, 6, 12])
    ],
    # same config, several seeds: separates config-effect from run variance,
    # and gives the 'learnedness' curves (every metric vs steps) with error
    # bars. The energy docs' claims explicitly mix with learnedness -- this
    # is the control group every other grid gets compared against.
    'learnedness': _grid('seed', '--seed', [0, 1, 2, 3]),
    # action mechanism comparison at fixed everything-else
    'action_mode': [
        ('discrete_reinforce',  {'--action-mode': 'discrete',
                                 '--agent': 'reinforce'}),
        ('discrete_rtac',       {'--action-mode': 'discrete',
                                 '--agent': 'rtac'}),
        ('continuous_reinforce', {'--action-mode': 'continuous',
                                  '--agent': 'reinforce'}),
        ('continuous_rtac',     {'--action-mode': 'continuous',
                                 '--agent': 'rtac'}),
        ('random_baseline',     {'--policy': 'random'}),
    ],
}


# ── Runner ────────────────────────────────────────────────────────────────────

def _run_cmd(name, cfg, args):
    json_path = os.path.join(args.outdir, f"{args.experiment}__{name}.json")
    log_path = json_path.replace('.json', '.log')
    cmd = [sys.executable, '-m', 'tests.integration.test_mandelbrot_rl',
           '--core', args.core, '--steps', str(args.steps),
           '--timeout', str(args.timeout),
           '--report-every', str(args.report_every),
           '--json-out', json_path]
    if args.action_mode and '--action-mode' not in cfg:
        cmd += ['--action-mode', args.action_mode]
    for k, v in cfg.items():
        cmd += [k, str(v)]
    return json_path, log_path, cmd


def _is_done(json_path):
    try:
        with open(json_path) as fh:
            return json.load(fh).get('done', False)
    except (OSError, json.JSONDecodeError):
        return False


def run_experiment(args):
    runs = EXPERIMENTS[args.experiment]
    if args.shard:
        i, n = (int(x) for x in args.shard.split('/'))
        runs = runs[i::n]
        print(f"shard {i}/{n}: {len(runs)} runs of this experiment")
    os.makedirs(args.outdir, exist_ok=True)

    pending = []
    for name, cfg in runs:
        json_path, log_path, cmd = _run_cmd(name, cfg, args)
        if _is_done(json_path):
            print(f"  [skip, already done] {name}")
            continue
        pending.append((name, json_path, log_path, cmd))

    active = []   # (name, Popen, log filehandle)
    failures = []
    t0 = time.time()
    while pending or active:
        while pending and len(active) < args.parallel:
            name, json_path, log_path, cmd = pending.pop(0)
            lf = open(log_path, 'w')
            print(f"  [start] {name}   ({len(pending)} queued)")
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            active.append((name, p, lf))
        time.sleep(5)
        still = []
        for name, p, lf in active:
            rc = p.poll()
            if rc is None:
                still.append((name, p, lf))
                continue
            lf.close()
            status = 'ok' if rc == 0 else f'FAILED rc={rc} (see .log)'
            print(f"  [done, {status}] {name}   "
                  f"({(time.time()-t0)/60:.1f} min elapsed)")
            if rc != 0:
                failures.append(name)
        active = still
    print(f"\nexperiment '{args.experiment}' complete: "
          f"{len(runs) - len(failures)} ok, {len(failures)} failed")
    if failures:
        print("failed runs:", ', '.join(failures))
    return 1 if failures else 0


# ── Plotter ───────────────────────────────────────────────────────────────────

CURVE_METRICS = ['probe', 'recon', 'nnz', 'fire_frac', 'vcr', 'hcr',
                 'ncd_view_hidden', 'coverage']


def plot_experiment(args):
    import matplotlib
    matplotlib.use('Agg')   # headless: overnight boxes have no display
    import matplotlib.pyplot as plt

    prefix = f"{args.experiment}__"
    data = {}
    for fn in sorted(os.listdir(args.outdir)):
        if not (fn.startswith(prefix) and fn.endswith('.json')):
            continue
        try:
            with open(os.path.join(args.outdir, fn)) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        name = fn[len(prefix):-len('.json')]
        if d.get('history'):
            data[name] = d
    if not data:
        print(f"no readable JSONs for '{args.experiment}' in {args.outdir}")
        return 1

    # 1) metric-vs-steps curves, one panel per metric, one line per config
    ncols = 2
    nrows = (len(CURVE_METRICS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows))
    axes = axes.ravel()
    for ax, metric in zip(axes, CURVE_METRICS):
        for name, d in data.items():
            xs = [h['step'] for h in d['history'] if h.get(metric) is not None]
            ys = [h[metric] for h in d['history'] if h.get(metric) is not None]
            if xs:
                ax.plot(xs, ys, label=name, alpha=0.85)
        ax.set_title(metric)
        ax.set_xlabel('step')
        partial = sum(0 if d.get('done') else 1 for d in data.values())
        if partial:
            ax.set_title(f"{metric}  ({partial} run(s) still partial)")
    for ax in axes[len(CURVE_METRICS):]:
        ax.axis('off')
    axes[0].legend(fontsize=7, loc='best')
    fig.suptitle(f"{args.experiment}: metrics vs steps")
    fig.tight_layout()
    curves_png = os.path.join(args.outdir, f"{args.experiment}_curves.png")
    fig.savefig(curves_png, dpi=110)
    plt.close(fig)

    # 2) final-value summary per config (bar per metric)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows))
    axes = axes.ravel()
    names = list(data.keys())
    for ax, metric in zip(axes, CURVE_METRICS):
        vals = []
        for n in names:
            hs = [h[metric] for h in data[n]['history']
                  if h.get(metric) is not None]
            vals.append(hs[-1] if hs else float('nan'))
        ax.bar(range(len(names)), vals)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha='right', fontsize=7)
        ax.set_title(f"final {metric}")
    for ax in axes[len(CURVE_METRICS):]:
        ax.axis('off')
    fig.suptitle(f"{args.experiment}: final values per config")
    fig.tight_layout()
    finals_png = os.path.join(args.outdir, f"{args.experiment}_finals.png")
    fig.savefig(finals_png, dpi=110)
    plt.close(fig)

    print(f"wrote {curves_png}")
    print(f"wrote {finals_png}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--experiment', choices=sorted(EXPERIMENTS), default=None)
    ap.add_argument('--list', action='store_true',
                    help='list experiments and their run counts, then exit')
    ap.add_argument('--outdir', default='results/mandelbrot')
    ap.add_argument('--parallel', type=int, default=3,
                    help='concurrent subprocesses (2-4 recommended; each run '
                         'is itself mostly single-core, so true process '
                         'parallelism is the win here)')
    ap.add_argument('--steps', type=int, default=500000)
    ap.add_argument('--timeout', type=float, default=28800,
                    help='per-run wall-clock budget, seconds (default 8h)')
    ap.add_argument('--report-every', type=int, default=2000)
    ap.add_argument('--core', default='sparse',
                    choices=['sparse', 'dense', 'mistral'])
    ap.add_argument('--action-mode', default='continuous',
                    choices=['discrete', 'continuous'],
                    help='default for grids that do not set it themselves')
    ap.add_argument('--shard', default=None, metavar='I/N',
                    help='run only every Nth config starting at I (for '
                         'splitting one experiment across machines, e.g. '
                         '0/3, 1/3, 2/3)')
    ap.add_argument('--plot', action='store_true',
                    help='plot whatever JSONs exist instead of running')
    args = ap.parse_args()

    if args.list or not args.experiment:
        print("experiments:")
        for name, runs in sorted(EXPERIMENTS.items()):
            print(f"  {name:24s} {len(runs):3d} runs")
        print("\n(pick one with --experiment; see --help for budget/parallel/"
              "shard flags)")
        return 0
    if args.plot:
        return plot_experiment(args)
    return run_experiment(args)


if __name__ == '__main__':
    sys.exit(main())
