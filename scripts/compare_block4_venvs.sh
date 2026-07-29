#!/usr/bin/env bash
# Runs scripts/bench_block4_layer.py against BOTH the pre-block4 baseline
# venv and this branch's own venv, then diffs the two JSON reports.
#
# Requires the comparison infrastructure set up earlier this session:
#   /home/simleek/claude_code/sili__new_baseline/  -- plain clone of
#     sili__new at main (pre-block4), with sili installed editable into...
#   /home/simleek/claude_code/.venv_baseline/       -- isolated venv
#   /home/simleek/claude_code/.venv/                -- this repo's own venv
#
# Usage: ./scripts/compare_block4_venvs.sh [extra args passed through to
#   bench_block4_layer.py, e.g. --n-in 1024 --n-out 1024 --density 0.01]

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

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

BENCH_SCRIPT="$SCRIPT_DIR/bench_block4_layer.py"

# The benchmark script only imports the INSTALLED `sili` package (resolved
# by whichever venv is active), not anything relative to cwd -- but Python
# puts the invoking script's cwd/dir on sys.path[0], so running this from
# INSIDE a checkout that happens to have its own sili/ subdirectory would
# silently shadow the properly editable-installed package with local
# source, defeating the whole comparison (this exact gotcha already bit
# `import sili` earlier this session when run from inside sili__new_baseline
# -- see TODO_DUAL_BLOCK4.md). Run from $TMP_DIR (guaranteed sili-free)
# both times to make the venv, not cwd, the only thing that changes.
echo "== Running under baseline venv (pre-block4, sili from $BASELINE_REPO) ==" >&2
(
    source "$BASELINE_VENV/bin/activate"
    cd "$TMP_DIR"
    python3 "$BENCH_SCRIPT" "$@" > "$TMP_DIR/baseline.json"
)

echo "== Running under this branch's venv ($NEW_VENV, sili from $REPO_DIR) ==" >&2
(
    source "$NEW_VENV/bin/activate"
    cd "$TMP_DIR"
    python3 "$BENCH_SCRIPT" "$@" > "$TMP_DIR/new.json"
)

python3 "$SCRIPT_DIR/diff_bench_reports.py" "$TMP_DIR/baseline.json" "$TMP_DIR/new.json"
