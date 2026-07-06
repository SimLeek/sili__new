#!/usr/bin/env bash
# Regenerate both API documentation sets.
#
#   docs/doxygen/html/index.html  -- C++ core (sili/lib/headers, cpu_backend.cpp)
#   docs/pdoc/index.html          -- Python package (sili/*)
#
# Both output directories are gitignored -- these are build artifacts,
# regenerated here or by CI, not committed source.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "== Doxygen (C++) =="
command -v doxygen >/dev/null || { echo "doxygen not found (apt-get install doxygen graphviz)"; exit 1; }
rm -rf docs/doxygen
doxygen Doxyfile
echo "-> docs/doxygen/html/index.html"

echo ""
echo "== pdoc (Python) =="
python3 -m pip show pdoc >/dev/null 2>&1 || { echo "pdoc not found (pip install pdoc)"; exit 1; }
rm -rf docs/pdoc
python3 -m pdoc --output-directory docs/pdoc sili
echo "-> docs/pdoc/index.html"
