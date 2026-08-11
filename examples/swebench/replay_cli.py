#!/usr/bin/env python3
"""A stand-in for ``claude`` that replays a canned attempt at a demo instance.

Only the demo uses this. It exists so the loader and the scoring protocol can be
exercised end to end without credentials, a 4 GB Docker image, or a clone of
Django — the adapter, the worktree isolation, the patch application and the
grader are all the real thing; only the agent is canned.

``--model`` selects the behaviour:

    solves       makes the fix the issue asked for
    regresses    satisfies the new test by breaking an old one
    edits_tests  tries to win by weakening the test file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BEHAVIOURS = ("solves", "regresses", "edits_tests")

# ---------------------------------------------------------------------------
# What each behaviour writes, per instance
# ---------------------------------------------------------------------------

_SLUG_FIXED = '''"""Turn titles into URL slugs."""

import re


def slugify(text: str) -> str:
    """Lowercase ``text`` and replace non-alphanumeric runs with hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower().strip()).strip("-")
'''

# Collapses the run, but also eats the edges rather than stripping them, so
# "  hello  " comes back as "-hello-": the new test passes, an old one breaks.
_SLUG_REGRESSED = '''"""Turn titles into URL slugs."""

import re


def slugify(text: str) -> str:
    """Lowercase ``text`` and replace non-alphanumeric runs with hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower())
'''

_TEXT_FIXED = '''"""Text helpers for rendering summaries."""


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` characters, on a word boundary."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if not text[limit - 1].isspace():
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "\\u2026"
'''

# Always appends the ellipsis, including when nothing was cut — so
# test_no_ellipsis_when_nothing_was_cut and the two pre-existing tests break.
_TEXT_REGRESSED = '''"""Text helpers for rendering summaries."""


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` characters, on a word boundary."""
    cut = text[: limit - 1]
    if len(text) > limit and not text[limit - 1].isspace():
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "\\u2026"
'''

_GUTTED_SLUG_TESTS = '''from slugify import slugify


def test_lowercases():
    assert slugify("Hello") == "hello"
'''

_GUTTED_TEXT_TESTS = '''from textutil import truncate


def test_short_text_is_unchanged():
    assert truncate("hello", 10) == "hello"
'''

# (instance keyword in the prompt) -> per-behaviour {path: contents}
_ATTEMPTS = {
    "slugify": {
        "solves": {"slugify.py": _SLUG_FIXED},
        "regresses": {"slugify.py": _SLUG_REGRESSED},
        "edits_tests": {"tests/test_slugify.py": _GUTTED_SLUG_TESTS},
    },
    "truncate": {
        "solves": {"textutil.py": _TEXT_FIXED},
        "regresses": {"textutil.py": _TEXT_REGRESSED},
        "edits_tests": {"tests/test_textutil.py": _GUTTED_TEXT_TESTS},
    },
}

_COST = {"solves": 0.41, "regresses": 0.53, "edits_tests": 0.29}


# ---------------------------------------------------------------------------


def _events(cwd: Path, files: dict[str, str], behaviour: str) -> list[dict]:
    """A plausible stream-json run that writes ``files``."""
    session = "demo"

    def assistant(content, tokens=(6400, 320), stop="tool_use"):
        return {
            "type": "assistant", "session_id": session,
            "message": {"role": "assistant", "id": "m", "model": "replayed",
                        "stop_reason": stop,
                        "usage": {"input_tokens": tokens[0],
                                  "output_tokens": tokens[1]},
                        "content": content},
        }

    def result(tid, text):
        return {"type": "user", "session_id": session,
                "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tid,
                     "content": text, "is_error": False}]}}

    events: list[dict] = [
        {"type": "system", "subtype": "init", "session_id": session,
         "cwd": str(cwd)},
        assistant([{"type": "tool_use", "id": "r0", "name": "Glob",
                    "input": {"pattern": "**/*.py"}}]),
        result("r0", "\n".join(sorted(files))),
    ]
    for n, (path, content) in enumerate(files.items()):
        tid = f"w{n}"
        events += [
            assistant([{"type": "tool_use", "id": tid, "name": "Write",
                        "input": {"file_path": str(cwd / path),
                                  "content": content}}]),
            result(tid, f"The file {path} has been updated."),
        ]
    events += [
        assistant([{"type": "tool_use", "id": "b0", "name": "Bash",
                    "input": {"command": "python -m pytest tests/ -q"}}]),
        result("b0", "tests pass"),
        assistant([{"type": "text", "text": "Done."}], tokens=(400, 30),
                  stop="end_turn"),
        {"type": "result", "subtype": "success", "session_id": session,
         "is_error": False, "num_turns": 3 + len(files),
         "duration_ms": 42000, "total_cost_usd": _COST[behaviour],
         "result": "Done.", "permission_denials": []},
    ]
    return events


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-p", "--print", dest="prompt", default="")
    parser.add_argument("--model", default="solves")
    args, _ = parser.parse_known_args(argv)

    if args.model not in BEHAVIOURS:
        print(f"replay: unknown behaviour {args.model!r}", file=sys.stderr)
        return 2

    key = next((k for k in _ATTEMPTS if k in args.prompt), None)
    if key is None:
        print("replay: no canned attempt matches this prompt", file=sys.stderr)
        return 2

    cwd = Path(os.getcwd())
    files = _ATTEMPTS[key][args.model]
    for path, content in files.items():
        target = cwd / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    for event in _events(cwd, files, args.model):
        print(json.dumps(event), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
