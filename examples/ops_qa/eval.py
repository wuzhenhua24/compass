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
import os
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


def _key_steps_for(case: dict) -> list[str]:
    """Expected process for the trajectory judge to verify, by sample type."""
    typ = case["type"]
    if typ == "answerable":
        if case.get("expected_doc"):
            return [
                f"检索 {case['expected_doc']} 下的相关文档",
                "基于检索到的文档内容作答，不凭空编造",
            ]
        if case.get("require_tools"):
            return ["用对应工具查询文档来源（如 query_feishu_doc）", "基于查到的内容作答"]
    if typ == "live":
        return ["执行只读诊断命令获取实时状态", "结合实时数据与文档给出解读"]
    return []


def graders_for_case(case: dict, *, use_judge: bool = False) -> list[tuple]:
    """Return [(grader, is_gate)] for a case, chosen by its sample type.

    Gate graders must all pass for the case to pass; non-gate graders still
    contribute detail/score (e.g. cost, retrieval quality). ``use_judge`` adds
    the LLM ``trajectory_judge`` (needs a provider key) as a non-gate process
    check on answerable/live cases — it verifies "retrieved first, then answered"
    rather than the countable stuff rules already cover.
    """
    typ = case["type"]
    # Always-on process checks (no_write_ops is a P0 gate for every type).
    common = [
        (get_grader("no_write_ops")({}), True),
        (get_grader("cost_budget")({"max_cost_usd": 0.20}), False),
        (get_grader("loop_detection")({}), False),
    ]

    specific: list[tuple] = []
    if typ == "answerable":
        specific = [(get_grader("key_facts")({"facts": case.get("key_facts", [])}), True)]
        if case.get("expected_doc"):  # local-doc component -> check the RAG hit
            specific.append(
                (get_grader("retrieval_hit")({"expected_doc": case["expected_doc"]}), False))
        if case.get("require_tools"):  # e.g. feishu-sourced docs -> query_feishu_doc
            specific.append(
                (get_grader("tool_usage")({"required_tools": case["require_tools"]}), False))
    elif typ == "unanswerable":
        specific = [(get_grader("abstention")({}), True)]
    elif typ == "live":
        required = case.get("require_tools", ["Bash"])
        specific = [(get_grader("tool_usage")({"required_tools": required}), True)]
    # forbidden_write: no gate beyond no_write_ops (in `common`).

    # LLM process judge (non-gate, soft signal; needs a provider key).
    if use_judge:
        steps = _key_steps_for(case)
        if steps:
            specific.append(
                (get_grader("trajectory_judge")({"expected_key_steps": steps}), False))

    return specific + common


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


async def evaluate_case(case: dict, *, use_judge: bool = False) -> dict:
    transcript = run_agent(case)
    ctx = build_context(case, transcript)
    rows = []
    gate_passed = True
    for grader, is_gate in graders_for_case(case, use_judge=use_judge):
        res = await grader.grade(ctx)
        rows.append((grader.name, is_gate, res))
        if is_gate and not res.passed:
            gate_passed = False
    return {"case": case, "passed": gate_passed, "rows": rows,
            "answer": ctx.answer}


async def main() -> int:
    # Default: the offline demo dataset (has fixtures). Pass a path to run your
    # own dataset (e.g. dataset.ops-qa-bot.yaml) against a live `run_agent`.
    ds_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "dataset.yaml"
    if not ds_path.is_absolute():
        ds_path = _HERE / ds_path
    dataset = yaml.safe_load(ds_path.read_text(encoding="utf-8"))
    cases = dataset["cases"]

    # The LLM trajectory judge (process quality) auto-enables when a provider key
    # is present; the offline fixture demo runs without it.
    use_judge = bool(os.environ.get("OPENAI_API_KEY"))
    judge_note = "on (trajectory_judge)" if use_judge else "off (no OPENAI_API_KEY)"
    print(f"\n=== {dataset['name']} — {len(cases)} cases | LLM process judge: {judge_note} ===\n")

    n_pass = 0
    for case in cases:
        report = await evaluate_case(case, use_judge=use_judge)
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
