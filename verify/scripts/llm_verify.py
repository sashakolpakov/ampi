#!/usr/bin/env python3
"""
LLM C++ verification script — two-pass review.

Pass 1: Generate a list of potential issues from the diff.
Pass 2: Filter out false positives given the full diff context.

Usage:
    python verify/scripts/llm_verify.py --diff /tmp/cpp_changes.diff --out /tmp/llm_report.md
"""

import argparse
import os
import sys
from openai import OpenAI

PASS1_SYSTEM = """\
You are a senior C++ code reviewer specialising in high-performance numerical
code, pybind11 extensions, and concurrent data structures.

Read the unified diff below and produce a numbered list of potential issues.
For each issue include:
- Severity: CRITICAL / WARNING / NOTE
- Location: file:line if determinable
- Description: one or two sentences

Cast a wide net — include anything suspicious even if you are not certain.
Do not filter yet. Output plain text, no preamble.
"""

PASS2_SYSTEM = """\
You are a senior C++ code reviewer. You have been given:
1. A unified diff of C++ changes.
2. A candidate issue list produced by a first-pass reviewer.

Your job is to remove false positives from the issue list.
For each candidate issue decide: KEEP or REMOVE.
Remove an issue if:
- It is about code on context lines (lines starting with ' ') or removed lines
  (starting with '-') — only lines starting with '+' are newly introduced code.
- It is a style preference with no correctness impact.
- The concern is already handled correctly in the diff.
- It is speculative with no concrete evidence in the diff.

Output only the kept issues, renumbered from 1, in the same format.
If nothing remains, write "No issues found."
Output plain Markdown suitable for a GitHub PR comment.
"""

MAX_DIFF_CHARS = 24_000


def call(client, system, user, max_tokens=1024):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


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

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    # Pass 1 — generate candidate issues
    raw_issues = call(
        client,
        PASS1_SYSTEM,
        f"```diff\n{diff_text}\n```",
        max_tokens=1024,
    )

    # Pass 2 — remove false positives
    pass2_user = (
        f"## Diff\n\n```diff\n{diff_text}\n```\n\n"
        f"## Candidate issues\n\n{raw_issues}"
    )
    final_report = call(
        client,
        PASS2_SYSTEM,
        pass2_user,
        max_tokens=1024,
    )

    open(args.out, "w").write(final_report + "\n")
    print(final_report)


if __name__ == "__main__":
    main()
