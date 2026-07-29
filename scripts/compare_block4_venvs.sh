#!/usr/bin/env bash
# Runs scripts/bench_block4_layer.py against BOTH the pre-block4 baseline
# venv and this branch's own venv, INTERLEAVED across --repeats runs (not
# baseline-fully-then-new-fully), then diffs the aggregated reports.
#
# Interleaving matters: speed is a statistical measurement, not a single
# number, and this machine's CPU frequency/thermal state drifts over a long
# run (confirmed by direct comparison earlier: even the pre-growth,
# block4-still-empty timing showed a ~40% gap purely from running baseline
# and new as two big sequential blocks rather than interleaved). Alternating
# baseline/new every repeat spreads each venv's samples across the same
# stretch of wall-clock time instead of two separate thermal regimes.
#
# Requires the comparison infrastructure set up earlier this session:
#   /home/simleek/claude_code/sili__new_baseline/  -- plain clone of
#     sili__new at main (pre-block4), with sili installed editable into...
#   /home/simleek/claude_code/.venv_baseline/       -- isolated venv
#   /home/simleek/claude_code/.venv/                -- this repo's own venv
#
# Usage: ./scripts/compare_block4_venvs.sh [--repeats N] [extra args passed
#   through to bench_block4_layer.py each repeat, e.g. --n-in 1024
#   --density 0.05]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CLAUDE_CODE_DIR="$(cd "$REPO_DIR/.." && pwd)"

BASELINE_REPO="$CLAUDE_CODE_DIR/sili__new_baseline"
BASELINE_VENV="$CLAUDE_CODE_DIR/.venv_baseline"
NEW_VENV="$CLAUDE_CODE_DIR/.venv"

for d in "$BASELINE_REPO" "$BASELINE_VENV" "$NEW_VENV"; do
    if [ ! -d "$d" ]; then
        echo "Missing comparison infrastructure: $d" >&2
        echo "See TODO_DUAL_BLOCK4.md's 'Comparison infrastructure' section." >&2
        exit 1
    fi
done

# Pull --repeats out of the argument list (bash-side only, not passed
# through to the python script); everything else is forwarded verbatim.
REPEATS=7
PASSTHROUGH_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
mkdir -p "$TMP_DIR/baseline" "$TMP_DIR/new"

BENCH_SCRIPT="$SCRIPT_DIR/bench_block4_layer.py"

# The benchmark script only imports the INSTALLED `sili` package (resolved
# by whichever venv is active), not anything relative to cwd -- but Python
# puts the invoking script's cwd/dir on sys.path[0], so running this from
# INSIDE a checkout that happens to have its own sili/ subdirectory would
# silently shadow the properly editable-installed package with local
# source, defeating the whole comparison (this exact gotcha already bit
# `import sili` earlier this session when run from inside sili__new_baseline
# -- see TODO_DUAL_BLOCK4.md). Run from $TMP_DIR (guaranteed sili-free)
# every time, both venvs, so the venv is the only thing that changes.

# The two venvs resolve to DIFFERENT numpy/BLAS builds (confirmed: this
# repo's venv links scipy-openblas64 with MAX_THREADS=64, .venv_baseline
# links plain system BLAS) -- sili's own forward_dense/backward_dense never
# call BLAS at all, but an oversubscribed 64-thread pool competing for this
# machine's 8 hardware threads causes real contention for ANYTHING CPU-bound
# sharing that process, block4-unrelated pure-C++ loops included. Confirmed
# by a native (no Python) A/B comparison landing at ~parity (0.95x-0.99x)
# while the unpinned Python benchmark showed a spurious ~0.5-0.6x. Pin BLAS
# threading to 1 so the two venvs' asymmetric numpy builds stop being a
# confound neither venv's own code is responsible for.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "== Interleaving $REPEATS repeats: baseline ($BASELINE_REPO) vs this branch ($REPO_DIR) ==" >&2
for i in $(seq 1 "$REPEATS"); do
    echo "-- repeat $i/$REPEATS: baseline --" >&2
    (
        source "$BASELINE_VENV/bin/activate"
        cd "$TMP_DIR"
        python3 "$BENCH_SCRIPT" "${PASSTHROUGH_ARGS[@]}" > "$TMP_DIR/baseline/$i.json"
    )
    echo "-- repeat $i/$REPEATS: this branch --" >&2
    (
        source "$NEW_VENV/bin/activate"
        cd "$TMP_DIR"
        python3 "$BENCH_SCRIPT" "${PASSTHROUGH_ARGS[@]}" > "$TMP_DIR/new/$i.json"
    )
done

python3 "$SCRIPT_DIR/diff_bench_reports.py" "$TMP_DIR/baseline" "$TMP_DIR/new"
