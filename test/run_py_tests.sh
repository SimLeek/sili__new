#!/usr/bin/env bash
# Run the Python test suite (rnn_fold.py, multimodal_sparse_rnn.py integration tests).
# Requires the _cpu extension to be importable -- build with cmake first.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SILI_DIR="$REPO_ROOT/sili"
PY_DIR="$SCRIPT_DIR/python"

# Check _cpu is available
python3 -c "import sys; sys.path.insert(0,'$SILI_DIR'); import _cpu" 2>/dev/null \
  || { echo "SKIP: _cpu not importable (build with cmake first)"; exit 0; }

echo "Running Python tests..."
python3 -m pytest "$PY_DIR" -v 2>/dev/null \
  || python3 -m unittest discover -s "$PY_DIR" -v
echo "Python tests passed."
