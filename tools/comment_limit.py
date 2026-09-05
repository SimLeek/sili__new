import argparse
import os
import sys
from collections.abc import Iterator

# Configuration
TARGET_EXTENSIONS = (".py", ".cpp", ".h", ".hpp")
MAX_COMMENT_RATIO = 0.30  # 30% cutoff threshold


def analyze_file(filepath: str) -> tuple[int, int, int, list[int]]:  # noqa: C901, PLR0912 -- a line-classifier state machine reads clearer flat
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    total_lines = len(lines)
    if total_lines == 0:
        return 0, 0, 0, []

    comment_lines_count = 0
    bypass_count = 0
    in_multiline_comment = False
    multiline_quote: str | None = None  # '"""' or "'''" -- which delimiter opened the block

    comment_indices: list[int] = []

    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue

        # 1. Check for manual 'noqa' approval bypass first
        if "noqa: comment" in stripped.lower():
            bypass_count += 1
            continue

        # 2. Python multiline strings/docstrings (''' or """). Closing
        # delimiters can land mid-line (e.g. a triple-quoted data literal
        # like `"...text..."""`.split()`), not just at line-start -- track
        # parity of the OPENING delimiter's count on each line instead of
        # only checking line-start, or the block never closes and every
        # subsequent line in the file gets misclassified as a comment.
        if filepath.endswith(".py") and in_multiline_comment:
            assert multiline_quote is not None
            comment_lines_count += 1
            comment_indices.append(idx)
            if stripped.count(multiline_quote) % 2 == 1:
                in_multiline_comment = False
                multiline_quote = None
            continue

        if filepath.endswith(".py") and (stripped.startswith(('"""', "'''"))):
            quote = '"""' if stripped.startswith('"""') else "'''"
            comment_lines_count += 1
            comment_indices.append(idx)
            if stripped.count(quote) % 2 == 1:
                in_multiline_comment = True
                multiline_quote = quote
            continue

        # 3. C++ Multiline Blocks (/* */)
        if filepath.endswith((".cpp", ".h", ".hpp")):
            if "/*" in stripped and "*/" in stripped:
                comment_lines_count += 1
                comment_indices.append(idx)
                continue
            if "/*" in stripped:
                in_multiline_comment = True
                comment_lines_count += 1
                comment_indices.append(idx)
                continue
            if "*/" in stripped:
                in_multiline_comment = False
                comment_lines_count += 1
                comment_indices.append(idx)
                continue

        # 4. Standard Inline Comments
        if stripped.startswith(("#", "//")):
            comment_lines_count += 1
            comment_indices.append(idx)

    return total_lines, comment_lines_count, bypass_count, comment_indices


def get_line_spans(indices: list[int]) -> list[str]:
    """Converts a sorted list of line numbers into continuous start-end string spans."""
    if not indices:
        return []

    spans: list[str] = []
    start = indices[0]
    end = indices[0]

    for i in indices[1:]:
        if i == end + 1:
            end = i
        else:
            spans.append(f"{start}-{end}" if start != end else f"{start}")
            start = i
            end = i
    spans.append(f"{start}-{end}" if start != end else f"{start}")
    return spans


def iter_target_files(paths: list[str]) -> Iterator[str]:
    """Yield candidate files from a mix of directories and individual file
    paths, so this doubles as a pre-commit per-file hook (which passes a
    list of changed files) and a manual whole-tree scan (`comment_limit.py`
    with no args)."""
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(TARGET_EXTENSIONS) and file != os.path.basename(__file__):
                        yield os.path.join(root, file)
        elif path.endswith(TARGET_EXTENSIONS) and os.path.basename(path) != os.path.basename(__file__):
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce strict comment density limits.")
    parser.add_argument("paths", nargs="*", default=["."], help="Directories and/or files to scan.")
    parser.add_argument("--silent", action="store_true", help="Suppress standard logging outputs.")
    args = parser.parse_args()

    failed_files = 0

    if not args.silent:
        print(f"Scanning {', '.join(args.paths)} for comment limits (> {MAX_COMMENT_RATIO:.0%})...")

    for full_path in iter_target_files(args.paths):
        file = os.path.basename(full_path)
        total, comments, bypasses, comment_indices = analyze_file(full_path)

        if total == 0:
            continue

        ratio = comments / total

        if ratio > MAX_COMMENT_RATIO:
            failed_files += 1
            spans = get_line_spans(comment_indices)
            spans_str = ", ".join(spans)

            if args.silent:
                print(f"{file}:{spans_str}")
            else:
                print(f"FAIL: {full_path} is {ratio:.1%} comments ({comments}/{total} lines).")
                print(f"   Lines: {file}:{spans_str}")

        # Bypasses always print unless explicitly running under --silent
        if bypasses > 0 and not args.silent:
            print(f"   INFO: ({bypasses} lines skipped via 'noqa: comment' in {file})")

    if failed_files > 0:
        if not args.silent:
            print(f"\nValidation failed. {failed_files} file(s) exceeded the 30% limit.")
        sys.exit(1)

    if not args.silent:
        print("\nALL PASSED: All files are within structural comment thresholds.")
    sys.exit(0)


if __name__ == "__main__":
    main()
