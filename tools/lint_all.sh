#!/usr/bin/env bash
# tools/lint_all.sh — run the full linter suite.
#   Hard gates fail the run (exit 1). Advisory tools report but never fail.
#   --fix   auto-fix what can be auto-fixed (ruff --fix, clang-format -i, codespell -w)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

FIX=0
[ "${1:-}" = "--fix" ] && FIX=1

HARD_FAIL=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; HARD_FAIL=1; }
adv()  { printf '  \033[33mADVIS\033[0m %s\n' "$1"; }
skip() { printf '  \033[90mSKIP\033[0m  %s\n' "$1"; }

hard() {  # hard <desc> <cmd...>
    local d="$1"; shift
    if "$@" >/tmp/lint_out 2>&1; then ok "$d"; else bad "$d"; sed 's/^/        /' /tmp/lint_out | head -25; fi
}
advisory() {  # advisory <desc> <cmd...>
    local d="$1"; shift
    if "$@" >/tmp/lint_out 2>&1; then ok "$d"; else adv "$d (findings, non-blocking)"; sed 's/^/        /' /tmp/lint_out | head -15; fi
}

echo "── python ──────────────────────────────────────────────"
[ "$FIX" -eq 1 ] && ruff check --fix sili tests tools >/dev/null 2>&1
hard  "ruff check"          ruff check sili tests tools
hard  "ruff format"         ruff format --check sili tests tools
hard  "bandit"              bandit -r sili -c .bandit -q
[ "$FIX" -eq 1 ] && codespell -w -q2 sili tests docs README.md >/dev/null 2>&1
hard  "codespell"           codespell --skip "*.sst,*.safetensors" sili tests docs README.md
hard  "deptry"              deptry .
# Advisory, not a hard gate: 468 pre-existing errors across 14 files in
# sili/ (never annotated before this lint setup existed) would otherwise
# block every future commit touching those files. Promote to hard once
# sili/ is actually clean.
advisory "mypy (strict)"       mypy sili
advisory "pydoclint"           pydoclint --style=numpy --check-return-types=False sili
advisory "vulture (dead code)"   vulture sili tools --min-confidence 80
advisory "radon (complexity)"    radon cc sili -s -nb --min B
# Whole-tree baseline scan (NOT scoped to changed files, unlike the hooks
# above and unlike clang-tidy's readability-function-size which only runs
# on changed files) -- catches pre-existing giant functions like
# disldo_backward (2403 lines, CCN=251, linear_disldo.hpp) that a
# per-commit hook would never surface since nobody's editing that file
# right now. Run this manually / in CI, not just on diffs.
advisory "lizard (function length/complexity, whole tree)" lizard sili tools -w

echo "── shell ───────────────────────────────────────────────"
mapfile -t sh_files < <(find tests docs -name '*.sh' 2>/dev/null)
if [ "${#sh_files[@]}" -gt 0 ]; then
    hard "shellcheck" shellcheck --external-sources "${sh_files[@]}"
else
    skip "shellcheck (no .sh files found)"
fi

echo "── C++ ─────────────────────────────────────────────────"
mapfile -t hpp_files < <(find sili -name '*.hpp' 2>/dev/null)
mapfile -t cpp_files < <(find sili -maxdepth 2 -name '*.cpp' 2>/dev/null)
cpp_all=("${hpp_files[@]}" "${cpp_files[@]}")
if [ "$FIX" -eq 1 ]; then
    [ "${#cpp_all[@]}" -gt 0 ] && clang-format -i "${cpp_all[@]}" && ok "clang-format (applied)"
elif command -v clang-format >/dev/null && [ "${#cpp_all[@]}" -gt 0 ]; then
    hard "clang-format --check" clang-format --dry-run --Werror "${cpp_all[@]}"
else
    skip "clang-format"
fi

if [ -f compile_commands.json ]; then
    advisory "clang-tidy (full)" bash -c \
      "find sili -name '*.cpp' -print0 | xargs -0 -n1 run-clang-tidy -p . -header-filter='.*' -quiet 2>/dev/null | grep -E 'warning:|error:' | head -40"
else
    skip "clang-tidy (no compile_commands.json — build with cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON or run: bear -- make)"
fi

if command -v cppcheck >/dev/null && [ "${#hpp_files[@]}" -gt 0 ]; then
    advisory "cppcheck" cppcheck --enable=all --std=c++20 -q \
        --suppress=missingIncludeSystem -I sili/lib/headers "${cpp_all[@]}" 2>&1 | head -30
else
    skip "cppcheck"
fi

echo "── comments ────────────────────────────────────────────"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    advisory "comment check (offline)" python3 tools/comment_check.py
    echo "        (LLM judge: not configured — see conversation notes on lintrule / DIY verifier)"
else
    skip "comment check (not a git repo)"
fi

echo "────────────────────────────────────────────────────────"
if [ "$HARD_FAIL" -eq 0 ]; then
    printf '\033[32mall hard gates passed\033[0m\n'; exit 0
else
    printf '\033[31msome hard gates failed\033[0m\n'; exit 1
fi
