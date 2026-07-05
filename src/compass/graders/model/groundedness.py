"""LLM-as-judge for groundedness: is the answer supported by observed evidence?

The canonical trace-level failure this targets is the *empty tool result
hallucination*: the user asks for a number, the agent calls the right-looking
tool, the tool returns an empty list, and the agent fabricates a plausible
answer. To the user the final answer looks fine; judged against the only
evidence the agent actually observed, it is unsupported. Outcome-only graders
(semantic match, rubric) cannot catch this — a fabricated answer can be
fluent, well-formed, and even correct by luck.

``groundedness`` is a BOTH-scope Model grader: it reads the final answer from
the Outcome and the tool observations from the Transcript, then asks an LLM
judge whether the answer's claims are supported by, and faithful to, what the
tools actually returned. Empty/error tool results are flagged explicitly so
the judge checks nothing was fabricated from them.

Reuses :class:`RubricGrader`'s LLM plumbing, structured-output schema, and
result assembly — only *what* is judged changes (answer vs. evidence), same
as ``trajectory_judge``.
"""

from __future__ import annotations

import fnmatch
from typing import Any

from compass.graders.base import GradeContext, GraderScope
from compass.graders.model.rubric import RubricGrader, _build_evaluation_schema
from compass.graders.model.trajectory import _compact
from compass.graders.registry import register_grader

# Default rubric: the three ways an answer can detach from its evidence.
_DEFAULT_CRITERIA: list[dict[str, Any]] = [
    {"name": "claim_support",
     "description": "Every factual claim in the answer (numbers, names, dates, "
                    "quotes, statuses) is supported by the tool observations shown. "
                    "Claims with no supporting observation are unsupported — general "
                    "knowledge is acceptable only if clearly presented as such."},
    {"name": "no_fabrication",
     "description": "Where a tool returned an EMPTY or ERROR result, the answer does "
                    "not present content as if it came from that call. Fabricating a "
                    "plausible answer from empty evidence is the canonical failure — "
                    "score it 0. Honestly reporting that nothing was found scores 1."},
    {"name": "faithful_use",
     "description": "The answer does not distort what the observations actually said: "
                    "no misquoting, changed units or magnitudes, dropped qualifiers, "
                    "or conclusions the evidence contradicts."},
]


def _is_empty_result(output: Any) -> bool:
    """Whether a tool output carries no usable evidence."""
    if output is None:
        return True
    if isinstance(output, (list, dict)):
        return len(output) == 0
    if isinstance(output, str):
        return output.strip().lower() in {"", "[]", "{}", "null", "none"}
    return False


@register_grader("groundedness")
class GroundednessGrader(RubricGrader):
    """LLM judge for answer-vs-evidence groundedness.

    Scope: BOTH — the answer comes from the Outcome, the evidence (tool
    observations) from the Transcript. Neither alone can answer "is this
    claim supported by what the agent actually saw?".

    Configuration (all optional):
        criteria:         rubric criteria; defaults to claim_support /
                          no_fabrication / faithful_use
        evidence_tools:   fnmatch patterns — only matching tool calls count as
                          evidence (default: all)
        exclude_tools:    fnmatch patterns to skip (default ["llm.*"]: imported
                          llm.generation spans carry the model's own text, which
                          must not count as evidence for itself)
        max_tool_calls:   cap evidence calls serialized into the prompt (default 40)
        max_output_chars: truncate each observation (default 600 — evidence needs
                          more room than trajectory judging)
        max_input_chars:  truncate each tool input (default 200)
        pass_threshold / model / provider: inherited from rubric

    Example::

        graders:
          - name: groundedness
            config:
              evidence_tools: ["Read", "Grep", "db.*"]
    """

    name = "groundedness"
    grader_scope = GraderScope.BOTH

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        if not self.criteria:
            from compass.graders.model.rubric import Criterion
            self.criteria = [Criterion.from_dict(c) for c in _DEFAULT_CRITERIA]
            self._schema = _build_evaluation_schema(self.criteria)

        cfg = self.config
        self.evidence_tools = [str(p) for p in cfg.get("evidence_tools", [])]
        self.exclude_tools = [str(p) for p in cfg.get("exclude_tools", ["llm.*"])]
        self.max_tool_calls = int(cfg.get("max_tool_calls", 40))
        self.max_output_chars = int(cfg.get("max_output_chars", 600))
        self.max_input_chars = int(cfg.get("max_input_chars", 200))
        # Only used in the "nothing to judge" error message.
        self.source = "outcome answer + transcript evidence"

    # -- evidence selection -------------------------------------------------

    def _evidence_calls(self, context: GradeContext) -> list:
        calls = context.tool_calls
        if self.evidence_tools:
            calls = [
                tc for tc in calls
                if any(fnmatch.fnmatch(tc.tool_name, p) for p in self.evidence_tools)
            ]
        if self.exclude_tools:
            calls = [
                tc for tc in calls
                if not any(fnmatch.fnmatch(tc.tool_name, p) for p in self.exclude_tools)
            ]
        return calls

    # -- what gets judged: the answer against the observations --------------

    def _extract_content(self, context: GradeContext) -> str | None:
        answer = context.answer
        if not answer:
            return None  # nothing to judge for groundedness

        evidence = self._evidence_calls(context)

        lines: list[str] = [
            "Judge whether the agent's final answer is GROUNDED in the evidence "
            "it actually observed. The evidence below is the complete set of "
            "tool observations available to the agent — anything in the answer "
            "that is not supported by it (or clearly framed as general "
            "knowledge) is unsupported.",
            "",
            f"Final answer:\n{answer}",
            "",
            "Evidence (tool observations):",
        ]

        shown = evidence[: self.max_tool_calls]
        for i, tc in enumerate(shown, 1):
            inp = _compact(tc.input, self.max_input_chars)
            out = _compact(tc.output, self.max_output_chars)
            flags = ""
            if tc.status != "ok":
                flags += " [ERROR: no evidence produced]"
            elif _is_empty_result(tc.output):
                flags += " [EMPTY RESULT: no evidence produced]"
            lines.append(f"[{i}] {tc.tool_name}({inp}) -> {out}{flags}")
        omitted = len(evidence) - len(shown)
        if omitted > 0:
            lines.append(f"... ({omitted} more observations omitted)")
        if not evidence:
            lines.append(
                "(no evidence tool calls recorded — any factual claim in the "
                "answer is unsupported unless clearly framed as general knowledge "
                "or an honest refusal)"
            )

        return "\n".join(lines)

    # -- context the judge sees: request + evidence stats -------------------

    def _build_context_info(self, context: GradeContext) -> str:
        parts: list[str] = []
        if context.prompt:
            parts.append(f"User request: {context.prompt}")

        evidence = self._evidence_calls(context)
        empty = sum(
            1 for tc in evidence
            if tc.status != "ok" or _is_empty_result(tc.output)
        )
        if evidence:
            stats = f"Evidence stats: {len(evidence)} observation(s)"
            if empty:
                stats += (
                    f", of which {empty} returned EMPTY/ERROR — verify the answer "
                    "does not present fabricated content as coming from those calls."
                )
            else:
                stats += "."
            parts.append(stats)
        return "\n\n".join(parts)
