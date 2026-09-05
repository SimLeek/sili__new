#!/usr/bin/env python3
"""Offline comment checker for sili__new.

First-pass filter for the AI-comment problem, zero dependencies:
flags comments added in the current diff that

  1. NARRATE the change   ("added this", "new", "now we", "updated")
  2. RESTATE the code     (every content word appears in the adjacent code)
  3. are near-empty filler ("# fix", "# todo", "// ok")

Emits reviewdog-friendly `path:line:col: message` output so findings can be
piped straight to inline PR comments. This is the triage pass; an LLM judge
(same output format) bolts on top for anything this clears.

Exit 1 if any finding, 0 otherwise. Advisory in lint_all.sh.
"""

import re
import subprocess
import sys

COMMENT_RE = re.compile(r"^\s*(?:#|//)\s?(.*)$")
NARRATE_RE = re.compile(
    r"\b(added|adding|new\b|newly|now we|updated|changed|this line|this code|"
    r"this block|for the new|per the new)\b",
    re.IGNORECASE,
)
GENERIC_OK = re.compile(r"^(fix|todo|ok|done|hack|note|tmp|temp|x+)$", re.IGNORECASE)
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "and",
    "or",
    "not",
    "no",
    "yes",
    "if",
    "then",
    "else",
    "when",
    "while",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "at",
    "by",
    "as",
    "from",
    "we",
    "you",
    "i",
    "he",
    "she",
    "they",
    "do",
    "does",
    "did",
    "done",
    "will",
    "would",
    "can",
    "could",
    "should",
    "here",
    "there",
    "value",
    "values",
    "type",
    "types",
}
MIN_CONTENT_CHARS = 4  # words shorter than this carry no restatement signal


def extract_code_words(lines: list[str]) -> set[str]:
    words: set[str] = set()
    for ln in lines:
        for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ln):
            if len(w) >= MIN_CONTENT_CHARS:
                words.add(w.lower())
    return words


def flag_comment(text: str, code_words: set[str]) -> str | None:
    t = text.strip()
    if not t:
        return None
    if NARRATE_RE.search(t):
        return "narrates the change instead of explaining it"
    if GENERIC_OK.match(t) or len(t) < 12:
        return "near-empty filler comment"
    content = [
        w.lower()
        for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", t)
        if len(w) >= MIN_CONTENT_CHARS and w.lower() not in STOPWORDS
    ]
    if content and all(w in code_words for w in content):
        return "restates the adjacent code (every content word appears there)"
    return None


def main() -> int:  # noqa: C901 -- a diff-line-state-machine reads clearer flat than split up
    try:
        git_diff_args = ("git", "diff", "--unified=3", "HEAD")
        diff = subprocess.run(git_diff_args, capture_output=True, text=True, check=True).stdout  # noqa: S603
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("comment_check: not in a git repo / no diff; nothing to do")
        return 0

    findings = 0
    path: str | None = None
    new_line, buf_line = 0, 0
    buf: list[str] = []

    def flush(context_after: list[str]) -> None:
        nonlocal findings, buf
        if not buf:
            return
        reason = flag_comment(" ".join(buf), extract_code_words(context_after))
        if reason:
            findings += 1
            print(f'{path}:{buf_line}:1: [comment-check] {reason}: "{" ".join(buf)[:80]}"')
        buf = []

    lines = diff.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("+++ b/"):
            flush([])
            path = ln[6:]
            continue
        if ln.startswith("@@"):
            flush([])
            m = re.search(r"\+(\d+)", ln)
            new_line = int(m.group(1)) - 1 if m else 0
            continue
        if ln.startswith("+") and not ln.startswith("+++"):
            new_line += 1
            m = COMMENT_RE.match(ln[1:])
            if m:
                if not buf:
                    buf_line = new_line
                buf.append(m.group(1))
            else:
                flush([ctx_ln[1:] for ctx_ln in lines[i + 1 : i + 6] if ctx_ln[:1] in "+ "])
        elif ln.startswith("-") and not ln.startswith("---"):
            pass  # removed lines don't shift new-file numbering
        else:
            flush([ctx_ln[1:] for ctx_ln in lines[i + 1 : i + 6] if ctx_ln[:1] in "+ "])
            new_line += 1
    flush([])

    if findings == 0:
        print(f"comment_check: clean ({path or 'no comments in diff'})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
