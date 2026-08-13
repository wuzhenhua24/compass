#!/usr/bin/env python3
"""A stand-in for the ``claude`` CLI that acts out how a skill changes a run.

Point the adapter's ``cli_path`` at this file instead of at ``claude`` and
nothing else changes: the same ``claude_code`` adapter installs the skill,
spawns this, parses its stream-json, computes the diff and hands the transcript
to the same graders. Going live is deleting one line of config.

**It reads the installed skill to decide what to do.** Not a lookup table keyed
on a flag — it opens ``.claude/skills/report-writer/SKILL.md`` in its working
directory, exactly where the adapter put it. So the skill axis is genuinely
under test here: break the install and this replay stops finding a skill,
reports a baseline run, and the numbers move. A canned table keyed on
``--model`` would have gone on producing a beautiful comparison of nothing.

The behaviour it acts out, per prompt shape:

                       baseline        v1                    v2
    explicit           no skill        loads it              loads it + script
    implicit           no skill        loads it 1 run in 3   loads it + script
    contextual         no skill        never loads it        loads it + script
    near-miss (x2)     no skill        never loads it        never loads it
    near-miss (chart)  no skill        never loads it        **loads it** — the
                                                             price of a pushier
                                                             description

That last row is the point of the example. v2 is better on every positive case
*and* introduces one false positive, which is what a real description rewrite
does and what a positives-only eval cannot see.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

SKILL_MD = Path(".claude/skills/report-writer/SKILL.md")
SCRIPT = ".claude/skills/report-writer/scripts/summarize.py"

REPORT = """# Q3 report

## Headline
Q3 revenue came in at 460,220.25, up 40,648.75 (+9.7%) on Q2.

## Numbers

| period | total |
|--------|-------|
| Q2 | 419,571.25 |
| Q3 | 460,220.25 |

## What changed
North carried the quarter (+21,480) and west added 26,909. South slipped
8,000 and is the only region below its Q2 number.
"""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-p", "--print", dest="prompt", default="")
    # Accepted and ignored: they steer the real CLI, not a replay.
    for flag in (
        "--model",
        "--output-format",
        "--append-system-prompt",
        "--system-prompt",
        "--allowedTools",
        "--disallowedTools",
        "--permission-mode",
        "--max-turns",
        "--setting-sources",
    ):
        parser.add_argument(flag, default="")
    parser.add_argument("--verbose", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    return args


def installed_version() -> str:
    """``"v1"`` / ``"v2"`` / ``""`` — read off the skill the adapter installed."""
    try:
        text = SKILL_MD.read_text(encoding="utf-8")
    except OSError:
        return ""
    for marker in ("v1", "v2"):
        if f"compass eval fixture: {marker}" in text:
            return marker
    return "unknown"


def prompt_shape(prompt: str) -> str:
    """Which of the example's prompt shapes this is."""
    if "report-writer" in prompt:
        return "explicit"
    if "图表" in prompt or "chart" in prompt.lower():
        return "near_miss_chart"
    if "JSON" in prompt or "json" in prompt:
        return "near_miss_convert"
    if "措辞" in prompt or "改一下" in prompt:
        return "near_miss_edit"
    if "复盘" in prompt or "汇报" in prompt:
        return "contextual"
    return "implicit"


def loads_the_skill(version: str, shape: str) -> bool:
    if not version:
        return False
    if version == "v2":
        # Better recall on everything it should catch — and one it should not.
        return shape != "near_miss_convert" and shape != "near_miss_edit"
    # v1's vague description only wins when the user names the skill, and
    # sometimes when they describe the task in the skill's own words. The
    # coin-flip is not decoration: it is why `trials:` exists, and why a
    # trigger rate needs more than one run to mean anything.
    if shape == "explicit":
        return True
    if shape == "implicit":
        return random.random() < 1 / 3
    return False


# ---------------------------------------------------------------------------
# Event stream
# ---------------------------------------------------------------------------


def _assistant(turn_id: str, content: list[dict], usage: dict) -> dict:
    return {
        "type": "assistant",
        "session_id": "skill-eval",
        "message": {
            "role": "assistant",
            "id": turn_id,
            "model": "claude-replay",
            "stop_reason": "tool_use" if content[-1]["type"] == "tool_use" else "end_turn",
            "usage": usage,
            "content": content,
        },
    }


def _result_for(tool_use_id: str, text: str) -> dict:
    return {
        "type": "user",
        "session_id": "skill-eval",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": text,
                    "is_error": False,
                }
            ],
        },
    }


def build_events(version: str, loaded: bool, writes_report: bool) -> list[dict]:
    """The run, as the CLI would have streamed it."""
    events: list[dict] = []
    n = 0

    def step(name: str, tool_input: dict, output: str, tokens: int) -> None:
        nonlocal n
        n += 1
        call_id = f"tu{n}"
        events.append(
            _assistant(
                f"m{n}",
                [{"type": "tool_use", "id": call_id, "name": name, "input": tool_input}],
                {"input_tokens": tokens, "output_tokens": 120},
            )
        )
        events.append(_result_for(call_id, output))

    if loaded:
        step("Skill", {"skill": "report-writer"}, "Loaded skill: report-writer", 2400)

    if not writes_report:
        # A near-miss the skill correctly stayed out of: the agent just does
        # the small thing that was asked.
        step("Read", {"file_path": "data/q3.csv"}, "period,region,amount…", 900)
        events.append(
            _assistant(
                "mz",
                [{"type": "text", "text": "Done — that did not need the report workflow."}],
                {"input_tokens": 400, "output_tokens": 60},
            )
        )
        cost = 0.021
    elif loaded and version == "v2":
        step("Bash", {"command": f"python {SCRIPT} data/q3.csv"},
             '{"latest": "Q3", "delta": 40648.75}', 2600)
        step("Write", {"file_path": "report.md"}, "Wrote report.md", 1500)
        events.append(
            _assistant(
                "mz",
                [{"type": "text", "text": "Wrote report.md — Q3 up 9.7% on Q2."}],
                {"input_tokens": 500, "output_tokens": 90},
            )
        )
        cost = 0.049
    else:
        # No script to lean on: read the CSV, do the arithmetic by hand, write.
        step("Read", {"file_path": "data/q3.csv"}, "period,region,amount…", 1800)
        awk = "awk -F, '{s[$1]+=$3} END {for (p in s) print p, s[p]}' data/q3.csv"
        step("Bash", {"command": awk}, "Q2 419571.25\nQ3 460220.25", 1200)
        step("Bash", {"command": "python -c \"print(460220.25 - 419571.25)\""},
             "40648.75", 900)
        step("Write", {"file_path": "report.md"}, "Wrote report.md", 1600)
        events.append(
            _assistant(
                "mz",
                [{"type": "text", "text": "Wrote report.md — Q3 up 9.7% on Q2."}],
                {"input_tokens": 500, "output_tokens": 90},
            )
        )
        cost = 0.061 if loaded else 0.052

    events.append(
        {
            "type": "result",
            "subtype": "success",
            "session_id": "skill-eval",
            "is_error": False,
            "num_turns": n + 1,
            "duration_ms": 4200 + n * 900,
            "total_cost_usd": cost,
            "result": events[-1]["message"]["content"][0]["text"],
            "permission_denials": [],
        }
    )
    return events


def main() -> int:
    args = _parse_args(sys.argv[1:])
    version = installed_version()
    shape = prompt_shape(args.prompt)
    loaded = loads_the_skill(version, shape)
    writes_report = shape in ("explicit", "implicit", "contextual")

    # The diff comes from the filesystem and the transcript comes from the
    # stream: a replay that only did one of them would produce a trial that
    # contradicts itself.
    if writes_report:
        Path("report.md").write_text(REPORT, encoding="utf-8")

    time.sleep(0.05)  # so latency_budget has something non-zero to measure
    for event in build_events(version, loaded, writes_report):
        print(json.dumps(event), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
