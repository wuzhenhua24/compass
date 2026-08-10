"""Exact-match grader.

Backs the ``expected.equals`` / ``expected.equals_json`` shorthand, which
``ExpectedConfig.to_graders()`` expands into ``GraderConfig(name="exact_match")``.
Without it that shorthand produced an unregistered grader name: the case failed
with a "Grader not found" error even when the answer was exactly right.
"""

from __future__ import annotations

import json
from typing import Any

from compass.graders.base import (
    CodeGrader,
    GradeContext,
    GradeResult,
    GraderScope,
    GraderType,
)
from compass.graders.registry import register_grader


@register_grader("exact_match")
class ExactMatchGrader(CodeGrader):
    """Compare the agent's answer against an expected value.

    Config:
        expected:       exact string to match
        expected_json:  structure to match after parsing the answer as JSON
        strip:          trim surrounding whitespace before comparing (default True)
        case_sensitive: default True; ignored for ``expected_json``

    Exactly one of ``expected`` / ``expected_json`` is required. Comparing
    against nothing is a configuration error, not a failed answer, so it is
    reported unscored rather than as a zero.
    """

    name = "exact_match"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME
    version = "1.0"

    async def grade(self, context: GradeContext) -> GradeResult:
        has_text = "expected" in self.config
        has_json = "expected_json" in self.config

        if has_text == has_json:
            return self._unscored(
                "exact_match requires exactly one of config.expected / "
                "config.expected_json"
            )

        answer = context.answer
        if has_json:
            return self._match_json(answer, self.config["expected_json"])
        return self._match_text(answer, str(self.config["expected"]))

    # ------------------------------------------------------------------

    def _match_text(self, answer: str, expected: str) -> GradeResult:
        actual = answer
        if self.config.get("strip", True):
            actual, expected = actual.strip(), expected.strip()
        if not self.config.get("case_sensitive", True):
            actual, expected = actual.lower(), expected.lower()

        ok = actual == expected
        return self._result(
            ok,
            details={
                "expected": expected,
                "actual": actual[:500],
                "actual_length": len(answer),
            },
            failure_tag="text_mismatch",
        )

    def _match_json(self, answer: str, expected: Any) -> GradeResult:
        try:
            actual = json.loads(answer) if answer.strip() else None
        except json.JSONDecodeError as e:
            # The answer was supposed to be JSON and is not — that is a real
            # measurement of the output, so it scores 0 rather than unscored.
            return self._result(
                False,
                details={"parse_error": str(e), "actual": answer[:500]},
                failure_tag="invalid_json",
            )

        ok = actual == expected
        return self._result(
            ok,
            details={"expected": expected, "actual": actual},
            failure_tag="json_mismatch",
        )

    def _result(
        self, ok: bool, *, details: dict[str, Any], failure_tag: str
    ) -> GradeResult:
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            grader_version=self.version,
            passed=ok,
            score=1.0 if ok else 0.0,
            details=details,
            failure_tags=[] if ok else [failure_tag],
        )

    def _unscored(self, message: str) -> GradeResult:
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            grader_version=self.version,
            passed=False,
            # Misconfiguration measured nothing about the agent.
            score=None,
            failure_tags=["grader_misconfigured"],
            error=message,
        )
