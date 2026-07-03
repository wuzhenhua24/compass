"""LLM-as-judge over the *trajectory* (process), not the final result.

Rule-based process graders (``tool_usage`` / ``loop_detection`` / ``cost_budget``
/ ``efficiency``) only catch the countable/mechanical. Qualitative judgments —
is the tool-call chain reasonable? did it miss a key step? did it over-explore?
was tool selection appropriate? — need an LLM reading the whole trajectory.

``trajectory_judge`` fills that gap: a TRANSCRIPT-scope Model grader that
serializes the transcript (tool calls + reasoning) and judges it against a
rubric. It **reuses** :class:`RubricGrader`'s LLM plumbing, structured-output
schema, and result assembly — only *what* is evaluated changes (the process,
not the outcome).

Positioning: the *machinery* (serialize trajectory → LLM judge → structured
score) is reusable and ships with the framework; the *criteria* / expected key
steps are supplied via config (domain-specific). See README "核心 vs 领域".
"""

from __future__ import annotations

import json
from typing import Any

from compass.graders.base import GradeContext, GraderScope
from compass.graders.model.rubric import RubricGrader, _build_evaluation_schema
from compass.graders.registry import register_grader

# Sensible default rubric when the caller supplies none — the four qualitative
# process axes rules can't cover.
_DEFAULT_CRITERIA: list[dict[str, Any]] = [
    {"name": "plan_soundness",
     "description": "The tool-call sequence follows a logical plan; ordering and "
                    "dependencies make sense (e.g. gather info before concluding)."},
    {"name": "completeness",
     "description": "No key step is missing (e.g. did not answer without retrieving, "
                    "did not conclude without verifying)."},
    {"name": "non_redundancy",
     "description": "No over-exploration: avoids redundant, repeated, or unnecessary "
                    "tool calls / detours."},
    {"name": "tool_choice",
     "description": "The tools chosen fit each subtask (not heavier or less appropriate "
                    "than needed)."},
]


@register_grader("trajectory_judge")
class TrajectoryJudgeGrader(RubricGrader):
    """LLM judge over the execution trajectory (process quality).

    Scope: TRANSCRIPT — evaluates *how* the agent worked (tool calls + reasoning),
    not the final output. Reuses :class:`RubricGrader`'s LLM/structured-output
    machinery; only the content being judged is the serialized trajectory.

    Configuration (all optional):
        criteria:            rubric criteria (name/description/weight); defaults to
                             plan_soundness / completeness / non_redundancy / tool_choice
        expected_key_steps:  list[str] — a checklist the judge verifies were done
        include_outcome:     also show the final answer for context (default True)
        max_tool_calls:      cap tool calls serialized into the prompt (default 60)
        max_output_chars:    truncate each tool output (default 400)
        max_input_chars:     truncate each tool input (default 300)
        max_reasoning_steps: cap reasoning steps serialized (default 40)
        pass_threshold:      overall score to pass (default 0.7)
        model / provider:    inherited from rubric (default gpt-4o / openai)

    Example::

        graders:
          - name: trajectory_judge
            config:
              expected_key_steps: ["检索对应文档", "基于文档作答"]
              criteria:
                - name: plan_soundness
                  description: "调用链顺序/依赖是否合理"
    """

    name = "trajectory_judge"
    grader_scope = GraderScope.TRANSCRIPT

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        # Default the rubric when none provided, then rebuild the schema so the
        # structured-output call matches these criteria.
        if not self.criteria:
            from compass.graders.model.rubric import Criterion
            self.criteria = [Criterion.from_dict(c) for c in _DEFAULT_CRITERIA]
            self._schema = _build_evaluation_schema(self.criteria)

        cfg = self.config
        self.expected_key_steps = [str(s) for s in cfg.get("expected_key_steps", [])]
        self.include_outcome = cfg.get("include_outcome", True)
        self.max_tool_calls = int(cfg.get("max_tool_calls", 60))
        self.max_output_chars = int(cfg.get("max_output_chars", 400))
        self.max_input_chars = int(cfg.get("max_input_chars", 300))
        self.max_reasoning_steps = int(cfg.get("max_reasoning_steps", 40))
        # Only used in the "nothing to judge" error message.
        self.source = "transcript"

    # -- what gets judged: the serialized trajectory, not the outcome ------

    def _extract_content(self, context: GradeContext) -> str | None:
        tool_calls = context.tool_calls
        reasoning = context.reasoning_steps
        if not tool_calls and not reasoning:
            return None  # empty transcript -> nothing to judge

        lines: list[str] = [
            "The following is an AI agent's execution trajectory for the user "
            "request. Judge the PROCESS (planning, completeness, redundancy, tool "
            "choice), not only whether the final answer looks right.",
            "",
            "Tool call sequence:",
        ]
        shown = tool_calls[: self.max_tool_calls]
        for i, tc in enumerate(shown, 1):
            tag = ""
            if tc.turn_index is not None:
                tag += f" turn={tc.turn_index}"
            if tc.agent_name:
                tag += f" agent={tc.agent_name}"
            inp = _compact(tc.input, self.max_input_chars)
            out = _compact(tc.output, self.max_output_chars)
            lines.append(f"[{i}] {tc.tool_name}({inp}) -> {out} [{tc.status}]{tag}")
        if not shown:
            lines.append("(no tool calls)")
        omitted = len(tool_calls) - len(shown)
        if omitted > 0:
            lines.append(f"... ({omitted} more tool calls omitted)")

        if reasoning:
            lines.append("")
            lines.append("Reasoning steps:")
            for s in reasoning[: self.max_reasoning_steps]:
                lines.append(f"- {_compact(s, self.max_output_chars)}")

        if self.include_outcome and context.answer:
            lines.append("")
            lines.append(f"Final answer: {_compact(context.answer, self.max_output_chars)}")

        return "\n".join(lines)

    # -- context the judge sees: request + key-steps checklist + stats ----

    def _build_context_info(self, context: GradeContext) -> str:
        parts: list[str] = []
        if context.prompt:
            parts.append(f"User request: {context.prompt}")
        if self.expected_key_steps:
            checklist = "\n".join(f"- {s}" for s in self.expected_key_steps)
            parts.append(
                "Expected key steps the trajectory should contain (verify each was "
                f"done):\n{checklist}"
            )
        if context.transcript is not None:
            cost = context.transcript.sum_cost().total_usd
            dur = context.transcript.total_duration_ms
            parts.append(
                f"Trajectory stats: {len(context.tool_calls)} tool calls, "
                f"{dur:.0f}ms, ${cost:.4f}."
            )
        return "\n\n".join(parts)


def _compact(value: Any, limit: int) -> str:
    """Flatten a value to a single truncated line for the judge prompt."""
    if value is None:
        return ""
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            s = str(value)
    s = " ".join(s.split())  # collapse whitespace/newlines
    return s if len(s) <= limit else s[:limit] + "…"
