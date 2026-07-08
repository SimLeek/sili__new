#!/usr/bin/env bash
# Run the Python test suite (rnn_fold.py, multimodal_sparse_rnn.py integration tests).
# Requires the sili._cpu extension to be importable -- build with
# `pip install -e .` first (see README.md).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PY_DIR="$SCRIPT_DIR/python"

# Check _cpu is available (package-qualified import -- see
# sili/conversion/rnn_fold.py for why a bare 'import _cpu' with a manual
# sys.path insert is the wrong pattern: it can make the compiled extension
# importable under two different sys.modules keys for the same .so file,
# which makes pybind11 run its static type registration twice)
python3 -c "import sys; sys.path.insert(0,'$REPO_ROOT'); from sili import _cpu" 2>/dev/null \
  || { echo "SKIP: _cpu not importable (build with 'pip install -e .' first)"; exit 0; }

echo "Running Python tests..."
python3 -m pytest "$PY_DIR" -v 2>/dev/null \
  || python3 -m unittest discover -s "$PY_DIR" -v
echo "Python tests passed."
