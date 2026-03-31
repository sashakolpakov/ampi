#!/usr/bin/env python3
"""
LLM C++ verification script — single-pass review via Claude Haiku.

Usage:
    python verify/scripts/llm_verify.py --diff /tmp/cpp_changes.diff --out /tmp/llm_report.md
"""

import argparse
import os
import sys
import anthropic

SYSTEM = """\
You are a senior C++ code reviewer specialising in high-performance numerical
code, pybind11 extensions, mmap, and concurrent data structures.

You will be given a unified diff. Your job is to find genuine bugs or safety
issues introduced by this change.

STRICT SCOPE RULES — violations of these rules make the review useless:
1. Only flag code on lines beginning with '+' (newly added lines).
   Lines beginning with ' ' (context) or '-' (removed) are NOT your concern.
2. Do not flag issues that are already fixed elsewhere in the same diff.
3. Do not flag style, naming, or speculative concerns.
4. Do not flag pre-existing code that is merely visible as context.

For each real issue found, output exactly:
- Severity: CRITICAL | WARNING
- Location: file:line (the '+' line number in the new file)
- Description: one or two sentences stating the concrete problem.

If there are no genuine issues, output exactly: No issues found.
No preamble. No commentary. Plain Markdown.
"""

MAX_DIFF_CHARS = 32_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff", required=True)
    parser.add_argument("--out",  required=True)
    args = parser.parse_args()

    diff_text = open(args.diff).read().strip()
    if not diff_text:
        open(args.out, "w").write("No C++ changes — verification skipped.\n")
        return

    if len(diff_text) > MAX_DIFF_CHARS:
        diff_text = diff_text[:MAX_DIFF_CHARS] + "\n\n[diff truncated]"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM,
        messages=[
            {"role": "user", "content": f"```diff\n{diff_text}\n```"},
        ],
    )

    report = message.content[0].text.strip()
    open(args.out, "w").write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
