"""Structured rubric prompt template using XML tags.

Provides a reusable template for building structured evaluation prompts
with XML-delimited sections: <role>, <scope_constraints>, <metrics_and_scoring>,
<verdict_rules>, and <output_schema>.

Design inspired by Anthropic's Image Evals approach for improving
LLM judge consistency and controllability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RubricPromptTemplate:
    """Structured prompt template for LLM-based evaluation graders.

    Encapsulates optional structured sections that, when populated,
    produce XML-tagged prompts. When empty, callers fall back to
    their existing plain-text prompt format.

    Attributes:
        role: Role description for the evaluator (e.g., "expert text rendering evaluator").
        scope_constraints: List of constraints limiting evaluation scope.
        verdict_rules: List of rules governing how verdicts/scores are determined.
    """

    role: str = ""
    scope_constraints: list[str] = field(default_factory=list)
    verdict_rules: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RubricPromptTemplate:
        """Create a RubricPromptTemplate from a grader config dict.

        Reads optional keys ``role``, ``scope_constraints``, and
        ``verdict_rules`` from *config*. Missing keys result in empty defaults.
        """
        return cls(
            role=config.get("role", ""),
            scope_constraints=config.get("scope_constraints", []),
            verdict_rules=config.get("verdict_rules", []),
        )

    @property
    def has_structured_sections(self) -> bool:
        """Return True if any structured section has content."""
        return bool(self.role or self.scope_constraints or self.verdict_rules)

    # ------------------------------------------------------------------
    # Rendering helpers used by RubricGrader
    # ------------------------------------------------------------------

    def render(
        self,
        *,
        criteria_text: str,
        content: str,
        context_info: str = "",
        instructions: str = "",
    ) -> str:
        """Render a full evaluation prompt with XML-tagged sections.

        Used by :class:`RubricGrader` when structured sections are present.
        """
        parts: list[str] = []

        if self.role:
            parts.append(f"<role>\n{self.role}\n</role>")

        if self.scope_constraints:
            items = "\n".join(f"- {c}" for c in self.scope_constraints)
            parts.append(f"<scope_constraints>\n{items}\n</scope_constraints>")

        parts.append(f"<metrics_and_scoring>\n{criteria_text}\n</metrics_and_scoring>")

        if self.verdict_rules:
            items = "\n".join(f"- {r}" for r in self.verdict_rules)
            parts.append(f"<verdict_rules>\n{items}\n</verdict_rules>")

        if context_info:
            parts.append(f"<context>\n{context_info}\n</context>")

        parts.append(f"<content>\n{content}\n</content>")

        instr = instructions or (
            "1. Evaluate the content against EACH criterion independently\n"
            "2. Assign a score from 0.0 (completely fails) to 1.0 "
            "(perfectly meets) for each criterion\n"
            "3. Provide clear reasoning for each score\n"
            "4. Calculate an overall score as a weighted average of "
            "criteria scores\n"
            "5. Provide an overall summary and specific suggestions "
            "for improvement\n"
            "6. Be objective and consistent in your evaluation"
        )
        parts.append(f"<instructions>\n{instr}\n</instructions>")

        parts.append(
            "<output_schema>\n"
            "Respond with a JSON object following the specified schema.\n"
            "</output_schema>"
        )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Rendering helpers used by VLMJudgeGrader
    # ------------------------------------------------------------------

    def render_system_prompt(self) -> str:
        """Render a structured system prompt for VLM-style graders.

        Returns a default evaluator system prompt when no structured
        sections are present.
        """
        if not self.has_structured_sections:
            return self._default_system_prompt()

        parts: list[str] = []

        if self.role:
            parts.append(f"<role>\n{self.role}\n</role>")

        if self.scope_constraints:
            items = "\n".join(f"- {c}" for c in self.scope_constraints)
            parts.append(f"<scope_constraints>\n{items}\n</scope_constraints>")

        if self.verdict_rules:
            items = "\n".join(f"- {r}" for r in self.verdict_rules)
            parts.append(f"<verdict_rules>\n{items}\n</verdict_rules>")

        return "\n\n".join(parts)

    def render_user_prompt(
        self,
        *,
        criteria_text: str,
        prompt: str = "",
        output_format: str = "",
    ) -> str:
        """Render a structured user prompt for VLM-style graders."""
        parts: list[str] = []

        if prompt:
            parts.append(f"<context>\nOriginal prompt: {prompt}\n</context>")

        parts.append(
            "<metrics_and_scoring>\n"
            "Evaluate this image against the following criteria:\n"
            f"{criteria_text}\n"
            "</metrics_and_scoring>"
        )

        fmt = output_format or (
            "For each criterion, state PASS, PARTIAL, or FAIL "
            "with a brief explanation.\n"
            "Then provide an overall score from 0.0 to 1.0.\n\n"
            "Respond in JSON format:\n"
            '{"score": <float>, "criterion_results": '
            '{"<criterion>": "PASS|PARTIAL|FAIL", ...}, '
            '"reasoning": "<explanation>"}'
        )
        parts.append(f"<output_schema>\n{fmt}\n</output_schema>")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _default_system_prompt() -> str:
        """Fallback system prompt when no structured sections exist."""
        return (
            "You are an image quality evaluator. Analyze the given image "
            "against the provided criteria.\n\n"
            "For each criterion, respond with:\n"
            "- PASS if the criterion is clearly met\n"
            "- FAIL if the criterion is not met\n"
            "- PARTIAL if partially met\n\n"
            "Provide brief reasoning for each judgment.\n\n"
            "Finally, give an overall score from 0.0 to 1.0 based on "
            "how well the image meets all criteria."
        )
