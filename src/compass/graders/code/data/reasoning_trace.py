"""Reasoning Trace Grader for Data Agent evaluation.

Evaluates the agent's analytical reasoning process.
"""

from __future__ import annotations

import json
import re
from typing import Any

from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.registry import register_grader


@register_grader("reasoning_trace")
class ReasoningTraceGrader(CodeGrader):
    """Evaluates the agent's analytical reasoning process.

    Assesses whether the agent:
    - Correctly understood the question
    - Used appropriate tables/columns
    - Applied correct filters and transformations
    - Showed logical reasoning steps

    This grader examines the agent's reasoning trace (if available) or
    infers reasoning quality from the output structure.

    Config:
        source: str — Where to get reasoning trace (default "output_data.reasoning").
        required_steps: list[str] — Steps that must be present.
            Options: "understand_question", "identify_tables", "plan_query",
                    "validate_result", "explain_answer"
        expected_tables: list[str] — Tables that should be referenced.
        expected_columns: list[str] — Columns that should be used.
        check_self_correction: bool — Check for error recovery (default True).

    Example YAML:
        graders:
          - name: reasoning_trace
            config:
              required_steps: [understand_question, identify_tables, validate_result]
              expected_tables: [users, orders]
              check_self_correction: true
    """

    grader_type = GraderType.CODE
    grader_scope = GraderScope.BOTH  # Needs both outcome and transcript

    DEFAULT_STEPS = [
        "understand_question",
        "identify_tables",
        "plan_query",
        "validate_result",
    ]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.source = self.config.get("source", "output_data.reasoning")
        self.required_steps = self.config.get("required_steps", self.DEFAULT_STEPS)
        self.expected_tables = self.config.get("expected_tables", [])
        self.expected_columns = self.config.get("expected_columns", [])
        self.check_self_correction = self.config.get("check_self_correction", True)

    async def grade(self, context: GradeContext) -> GradeResult:
        """Evaluate reasoning trace quality."""
        findings = {
            "steps_found": [],
            "steps_missing": [],
            "tables_found": [],
            "tables_missing": [],
            "columns_found": [],
            "columns_missing": [],
            "self_correction_detected": False,
        }

        # Extract reasoning content
        reasoning = self._extract_reasoning(context)
        sql = self._extract_sql(context)

        # Combine all text for analysis
        all_text = ""
        if reasoning:
            all_text += str(reasoning) + " "
        if sql:
            all_text += sql + " "
        if context.outcome and context.outcome.output_data:
            all_text += json.dumps(context.outcome.output_data)

        all_text_upper = all_text.upper()

        # Check required steps (heuristic detection)
        step_indicators = {
            "understand_question": ["QUESTION", "ASKING", "NEED TO FIND", "WANT TO KNOW"],
            "identify_tables": ["TABLE", "FROM", "DATASET", "SOURCE"],
            "plan_query": ["QUERY", "SELECT", "JOIN", "WHERE", "GROUP"],
            "validate_result": ["VERIFY", "CHECK", "VALIDATE", "CORRECT", "RESULT"],
            "explain_answer": ["ANSWER", "RESULT", "FOUND", "SHOWS", "MEANS"],
        }

        for step in self.required_steps:
            indicators = step_indicators.get(step, [step.upper()])
            if any(ind in all_text_upper for ind in indicators):
                findings["steps_found"].append(step)
            else:
                findings["steps_missing"].append(step)

        # Check expected tables
        for table in self.expected_tables:
            if table.upper() in all_text_upper:
                findings["tables_found"].append(table)
            else:
                findings["tables_missing"].append(table)

        # Check expected columns
        for col in self.expected_columns:
            if col.upper() in all_text_upper:
                findings["columns_found"].append(col)
            else:
                findings["columns_missing"].append(col)

        # Check for self-correction patterns
        if self.check_self_correction:
            correction_patterns = [
                r"ERROR|FAILED|WRONG|INCORRECT",
                r"RETRY|TRY AGAIN|ADJUST",
                r"ACTUALLY|INSTEAD|CORRECTION",
            ]
            for pattern in correction_patterns:
                if re.search(pattern, all_text_upper):
                    findings["self_correction_detected"] = True
                    break

        # Calculate score
        scores = []

        # Steps score
        if self.required_steps:
            steps_score = len(findings["steps_found"]) / len(self.required_steps)
            scores.append(steps_score)

        # Tables score
        if self.expected_tables:
            tables_score = len(findings["tables_found"]) / len(self.expected_tables)
            scores.append(tables_score)

        # Columns score
        if self.expected_columns:
            cols_score = len(findings["columns_found"]) / len(self.expected_columns)
            scores.append(cols_score)

        final_score = sum(scores) / len(scores) if scores else 1.0

        # Determine pass/fail
        critical_missing = (
            len(findings["tables_missing"]) > 0 and self.expected_tables
        )
        passed = final_score >= 0.7 and not critical_missing

        return GradeResult(
            score=final_score,
            passed=passed,
            details=findings,
            error=(
                f"Missing tables: {findings['tables_missing']}"
                if findings["tables_missing"]
                else None
            ),
        )

    def _extract_reasoning(self, context: GradeContext) -> Any:
        """Extract reasoning from context."""
        if not context.outcome:
            return None

        if self.source.startswith("output_data."):
            key = self.source[12:]
            data = context.outcome.output_data
            if isinstance(data, dict):
                return data.get(key)

        return None

    def _extract_sql(self, context: GradeContext) -> str | None:
        """Extract SQL from output."""
        if not context.outcome:
            return None

        data = context.outcome.output_data
        if isinstance(data, dict):
            return data.get("sql")
        return None
