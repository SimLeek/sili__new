#!/usr/bin/env bash
# Run both the C++ and Python test suites.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_cpp_tests.sh"
bash "$SCRIPT_DIR/run_py_tests.sh"
