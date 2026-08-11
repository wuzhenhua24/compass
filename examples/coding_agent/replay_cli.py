#!/usr/bin/env python3
"""A stand-in for the ``claude`` CLI that replays a canned run.

This is what makes the example runnable for free. Point the adapter's
``cli_path`` at this file instead of at ``claude`` and **nothing else changes**:
the same ``claude_code`` adapter spawns it, parses its stream-json, computes the
same git diff, and hands the same transcript to the same graders. Going live is
deleting one line of config.

It accepts the flags the adapter passes (``-p``, ``--model``,
``--output-format`` …), ignores the ones that only mean something to the real
CLI, and uses two of them:

- ``-p`` picks which requirement's run to replay.
- ``--model`` picks the **behaviour** — ``honest`` / ``cheats`` / ``overreach``.
  In a live run that flag really is the model, which is the point: the
  comparison axis is a config key either way, so the leaderboard machinery is
  exercised for real.

Replaying means two things, in order: apply the run's file edits to the working
directory, then emit its events on stdout. Both matter — the adapter's diff
comes from the filesystem and its transcript comes from the stream, so a replay
that only did one of them would produce a self-contradictory trial.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import fixtures  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-p", "--print", dest="prompt", default="")
    parser.add_argument("--model", default="honest")
    # Accepted and ignored: they steer the real CLI, not a replay.
    parser.add_argument("--output-format", default="stream-json")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--append-system-prompt", default="")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--allowedTools", default="")
    parser.add_argument("--disallowedTools", default="")
    parser.add_argument("--permission-mode", default="")
    parser.add_argument("--max-turns", default="")
    args, _unknown = parser.parse_known_args(argv)
    return args


def _apply_edit(cwd: Path, tool_input: dict) -> None:
    """Apply one Edit exactly as the real tool would: a single replacement."""
    path = cwd / tool_input["file_path"]
    old = tool_input["old_string"]
    new = tool_input["new_string"]
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"replay: cannot read {path}: {e}") from e
    if old not in content:
        # Loud, because the alternative is a trial that silently grades an
        # unedited tree — a green run that proves nothing.
        raise SystemExit(
            f"replay: anchor not found in {tool_input['file_path']}.\n"
            f"  The fixture in fixtures.py no longer matches project/. "
            f"Anchor was:\n    {old.splitlines()[0][:70]}"
        )
    path.write_text(content.replace(old, new, 1), encoding="utf-8")


def _apply_write(cwd: Path, tool_input: dict) -> None:
    path = cwd / tool_input["file_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tool_input.get("content", ""), encoding="utf-8")


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    cwd = Path(os.getcwd())

    try:
        events = fixtures.run_for(args.prompt, args.model)
    except KeyError as e:
        print(f"replay: {e}", file=sys.stderr)
        return 2

    for event in events:
        # The init event announces the working directory; the importer reads it
        # to make state-delta targets relative.
        if event.get("type") == "system" and event.get("subtype") == "init":
            event = {**event, "cwd": str(cwd)}

        if event.get("type") == "assistant":
            for block in event["message"].get("content", []):
                if block.get("type") != "tool_use":
                    continue
                if block["name"] == "Edit":
                    _apply_edit(cwd, block["input"])
                elif block["name"] == "Write":
                    _apply_write(cwd, block["input"])
                # Read/Bash change nothing on disk in a replay.
            event = _absolutize(event, cwd)

        print(json.dumps(event), flush=True)

    return 0


def _absolutize(event: dict, cwd: Path) -> dict:
    """Report tool paths absolute, as the real CLI does.

    Fixtures store them relative so they replay anywhere; the importer then
    turns them back into relative state-delta targets using the cwd from the
    init event. Round-tripping like this is not busywork — it exercises the
    same relativization a live run depends on.
    """
    content = []
    for block in event["message"].get("content", []):
        if block.get("type") == "tool_use" and "file_path" in block.get("input", {}):
            block = {
                **block,
                "input": {
                    **block["input"],
                    "file_path": str(cwd / block["input"]["file_path"]),
                },
            }
        content.append(block)
    return {**event, "message": {**event["message"], "content": content}}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
