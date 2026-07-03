"""Evaluate a doc-grounded ops QA agent with Compass.

Runs the dataset in ``dataset.yaml`` and grades each question along the axis set
that fits its sample type (answerable / unanswerable / live / forbidden_write).
By default it uses the offline fixtures so the whole pipeline is demonstrable
without the real bot or an API key::

    python examples/ops_qa/eval.py

To evaluate the *real* bot, implement ``run_agent`` to drive it and reconstruct
a Transcript (the commented ``claude_agent_sdk`` sketch), and add the ``rubric``
LLM-judge for semantic correctness / groundedness.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml

from compass.core.transcript import Transcript
from compass.graders import GradeContext, get_grader
from compass.integrations import reconstruct_transcript_from_wire

# The example's own modules live alongside this file; make them importable.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import fixtures as _fixtures  # noqa: E402  provides canned runs
import graders as _graders  # noqa: E402, F401  import registers the domain graders

# ---------------------------------------------------------------------------
# Agent runner — swap this for your real bot
# ---------------------------------------------------------------------------


def run_agent(case: dict) -> Transcript:
    """Return the Transcript for one question. Demo: replay a canned fixture."""
    events = _fixtures.fixtures().get(case["id"])
    if events is None:
        raise KeyError(f"no fixture for case {case['id']!r}")
    return reconstruct_transcript_from_wire(events)


# Real-agent version (requires `pip install claude-agent-sdk` + credentials):
#
# from claude_agent_sdk import query, ClaudeAgentOptions
# from compass.integrations import reconstruct_transcript_from_stream
# async def run_agent_live(case):
#     opts = ClaudeAgentOptions(
#         allowed_tools=["Read", "Grep", "Glob", "Bash"], cwd="/path/to/ops-qa-bot")
#     stream = query(prompt=case["question"], options=opts)
#     return await reconstruct_transcript_from_stream(stream)


# ---------------------------------------------------------------------------
# Per-type grader selection
# ---------------------------------------------------------------------------


def graders_for_case(case: dict) -> list[tuple]:
    """Return [(grader, is_gate)] for a case, chosen by its sample type.

    Gate graders must all pass for the case to pass; non-gate graders still
    contribute detail/score (e.g. cost, retrieval quality).
    """
    typ = case["type"]
    # Always-on process checks (no_write_ops is a P0 gate for every type).
    common = [
        (get_grader("no_write_ops")({}), True),
        (get_grader("cost_budget")({"max_cost_usd": 0.20}), False),
        (get_grader("loop_detection")({}), False),
    ]

    if typ == "answerable":
        return [
            (get_grader("key_facts")({"facts": case.get("key_facts", [])}), True),
            (get_grader("retrieval_hit")({"expected_doc": case.get("expected_doc", "")}), False),
        ] + common
    if typ == "unanswerable":
        return [(get_grader("abstention")({}), True)] + common
    if typ == "live":
        required = case.get("require_tools", ["Bash"])
        return [(get_grader("tool_usage")({"required_tools": required}), True)] + common
    if typ == "forbidden_write":
        # no_write_ops (in `common`) is the P0 gate here.
        return common
    return common


def build_context(case: dict, transcript: Transcript) -> GradeContext:
    return GradeContext(
        prompt=case["question"],
        reference_answer=case.get("reference_answer", "") or "",
        transcript=transcript,
        outcome=transcript.outcome,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def evaluate_case(case: dict) -> dict:
    transcript = run_agent(case)
    ctx = build_context(case, transcript)
    rows = []
    gate_passed = True
    for grader, is_gate in graders_for_case(case):
        res = await grader.grade(ctx)
        rows.append((grader.name, is_gate, res))
        if is_gate and not res.passed:
            gate_passed = False
    return {"case": case, "passed": gate_passed, "rows": rows,
            "answer": ctx.answer}


async def main() -> int:
    dataset = yaml.safe_load((_HERE / "dataset.yaml").read_text(encoding="utf-8"))
    cases = dataset["cases"]
    print(f"\n=== {dataset['name']} — {len(cases)} cases ===\n")

    n_pass = 0
    for case in cases:
        report = await evaluate_case(case)
        status = "PASS" if report["passed"] else "FAIL"
        n_pass += report["passed"]
        print(f"[{status}] {case['id']}  ({case['type']})")
        print(f"       Q: {case['question']}")
        print(f"       A: {report['answer'][:80]}")
        for name, is_gate, res in report["rows"]:
            mark = "✓" if res.passed else "✗"
            tag = " [gate]" if is_gate else ""
            print(f"         {mark} {name:14}{tag:7} {res.score:.2f}  {res.reasoning}")
        print()

    print(f"=== {n_pass}/{len(cases)} cases passed ===")
    print("Tip: run each case `trials` times and require all-pass for pass^k "
          "(reliability); add `rubric` for semantic correctness / groundedness.")
    return 0 if n_pass == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
