#!/usr/bin/env python3
"""Generates compile_commands.json for clang-tidy against sili/cpu_backend.cpp.

There's no top-level CMakeLists.txt for the pybind extension (only
tests/unit/ has one, which is unrelated) -- the real build is a single
setuptools Extension in setup.py. clang-tidy needs a compile_commands.json
to know the include paths/flags, so this mirrors that Extension's
include_dirs/extra_compile_args by hand (kept in sync manually -- setup.py
can't be imported directly, its setup() call isn't guarded by
`if __name__ == "__main__"`).
"""

import json
import sysconfig
from pathlib import Path

import pybind11

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "sili" / "cpu_backend.cpp"

args = [
    "c++",
    "-std=c++20",
    "-Wall",
    "-fPIC",
    "-march=native",
    "-fopenmp",
    "-ffast-math",
    "-fno-finite-math-only",
    f"-I{pybind11.get_include()}",
    f"-I{sysconfig.get_path('include')}",
    f"-I{ROOT / 'sili' / 'lib' / 'headers'}",
    "-c",
    str(SOURCE),
    "-o",
    str(SOURCE.with_suffix(".o")),
]

entry = {"directory": str(ROOT), "arguments": args, "file": str(SOURCE)}

out_path = ROOT / "compile_commands.json"
out_path.write_text(json.dumps([entry], indent=2))
print(f"wrote {out_path}")
