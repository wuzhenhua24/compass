"""Rubric-based grader using LLM with structured output.

This grader evaluates agent outputs against a multi-criteria rubric,
using structured output to ensure consistent, parseable evaluation results.

Key design choices:
1. Structured output schema enforces consistent evaluation format
2. Per-criterion scoring enables fine-grained diagnostics
3. Weighted criteria support prioritizing important dimensions
4. Provider-agnostic design works with OpenAI and Anthropic
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from compass.graders.base import (
    GradeContext,
    GradeResult,
    GraderScope,
    GraderType,
    ModelGrader,
)
from compass.graders.model.prompt_template import RubricPromptTemplate
from compass.graders.registry import register_grader


@dataclass
class Criterion:
    """A single evaluation criterion in the rubric."""

    name: str
    description: str
    weight: float = 1.0
    # Optional: specific scoring guidance
    score_guidance: dict[str, str] | None = None  # e.g., {"0": "Completely wrong", "1": "Perfect"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Criterion:
        """Create Criterion from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            weight=data.get("weight", 1.0),
            score_guidance=data.get("score_guidance"),
        )


@dataclass
class CriterionResult:
    """Evaluation result for a single criterion."""

    name: str
    score: float  # 0.0 to 1.0
    reasoning: str
    weight: float = 1.0

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class RubricEvaluation:
    """Complete evaluation result from the LLM."""

    overall_score: float
    criteria_results: list[CriterionResult]
    overall_reasoning: str
    confidence: float = 1.0
    suggestions: list[str] = field(default_factory=list)

    @classmethod
    def from_structured_output(cls, data: dict[str, Any], criteria: list[Criterion]) -> RubricEvaluation:
        """Parse structured output from LLM into RubricEvaluation."""
        criteria_results = []
        criteria_scores = data.get("criteria_scores", {})

        for criterion in criteria:
            score_data = criteria_scores.get(criterion.name, {})
            criteria_results.append(
                CriterionResult(
                    name=criterion.name,
                    score=float(score_data.get("score", 0.0)),
                    reasoning=score_data.get("reasoning", ""),
                    weight=criterion.weight,
                )
            )

        return cls(
            overall_score=float(data.get("overall_score", 0.0)),
            criteria_results=criteria_results,
            overall_reasoning=data.get("overall_reasoning", ""),
            confidence=float(data.get("confidence", 1.0)),
            suggestions=data.get("suggestions", []),
        )


def _build_evaluation_schema(criteria: list[Criterion]) -> dict[str, Any]:
    """Build JSON Schema for structured evaluation output."""
    criteria_properties = {}
    for criterion in criteria:
        criteria_properties[criterion.name] = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": f"Score for '{criterion.name}' (0=worst, 1=best)",
                },
                "reasoning": {
                    "type": "string",
                    "description": f"Explanation for the score on '{criterion.name}'",
                },
            },
            "required": ["score", "reasoning"],
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "properties": {
            "overall_score": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Overall evaluation score (0=worst, 1=best)",
            },
            "criteria_scores": {
                "type": "object",
                "properties": criteria_properties,
                "required": [c.name for c in criteria],
                "additionalProperties": False,
                "description": "Per-criterion evaluation scores",
            },
            "overall_reasoning": {
                "type": "string",
                "description": "Overall evaluation summary",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence in this evaluation (0=uncertain, 1=certain)",
            },
            "suggestions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific suggestions for improvement",
            },
        },
        "required": ["overall_score", "criteria_scores", "overall_reasoning"],
        "additionalProperties": False,
    }


def _build_evaluation_prompt(
    content: str,
    criteria: list[Criterion],
    context_info: str = "",
    template: RubricPromptTemplate | None = None,
) -> str:
    """Build the evaluation prompt for the LLM.

    When *template* is provided and has structured sections, delegates to
    ``template.render()`` to produce an XML-tagged prompt. Otherwise
    returns the legacy markdown-formatted prompt for backward compatibility.
    """
    criteria_text = "\n".join(
        f"- **{c.name}** (weight: {c.weight}): {c.description}"
        + (
            "\n  Scoring guide: " + ", ".join(f"{k}={v}" for k, v in c.score_guidance.items())
            if c.score_guidance
            else ""
        )
        for c in criteria
    )

    if template is not None and template.has_structured_sections:
        return template.render(
            criteria_text=criteria_text,
            content=content,
            context_info=context_info,
        )

    return f"""You are an expert evaluator. Evaluate the following content against the provided rubric.

## Evaluation Criteria

{criteria_text}

## Context
{context_info if context_info else "No additional context provided."}

## Content to Evaluate

{content}

## Instructions

1. Evaluate the content against EACH criterion independently
2. Assign a score from 0.0 (completely fails) to 1.0 (perfectly meets) for each criterion
3. Provide clear reasoning for each score
4. Calculate an overall score as a weighted average of criteria scores
5. Provide an overall summary and specific suggestions for improvement
6. Be objective and consistent in your evaluation

Respond with a JSON object following the specified schema."""


@register_grader("rubric")
class RubricGrader(ModelGrader):
    """Rubric-based evaluation using LLM with structured output.

    Scope: OUTCOME - evaluates the final output against a rubric.

    This grader uses structured output to ensure consistent, parseable
    evaluation results with per-criterion scores and reasoning.

    Configuration:
        criteria: List of evaluation criteria, each with:
            - name: Criterion identifier
            - description: What this criterion measures
            - weight: Importance weight (default 1.0)
            - score_guidance: Optional dict mapping scores to descriptions

        pass_threshold: Score threshold to pass (default 0.7)
        model: LLM model to use (default "gpt-4o")
        provider: LLM provider - "openai" or "anthropic" (default "openai")
        source: Content source - "output_data", "text_artifact", or "metadata.<key>"
        context_template: Optional template for additional context

    Example configuration:
        graders:
          - name: rubric
            config:
              criteria:
                - name: accuracy
                  description: "Factual correctness of the response"
                  weight: 2.0
                - name: completeness
                  description: "Coverage of all required points"
                  weight: 1.5
                - name: clarity
                  description: "Clear and well-structured presentation"
                  weight: 1.0
              pass_threshold: 0.7
              model: gpt-4o
              provider: openai
    """

    name = "rubric"
    grader_type = GraderType.MODEL
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        # Parse criteria
        criteria_config = self.config.get("criteria", [])
        self.criteria = [Criterion.from_dict(c) for c in criteria_config]

        # Evaluation settings
        self.pass_threshold = self.config.get("pass_threshold", 0.7)
        self.model = self.config.get("model", "gpt-4o")
        self.provider = self.config.get("provider", "openai")
        self.source = self.config.get("source", "output_data")
        self.context_template = self.config.get("context_template", "")

        # Structured prompt template (optional)
        self._template = RubricPromptTemplate.from_config(self.config)

        # Build schema once
        self._schema = _build_evaluation_schema(self.criteria) if self.criteria else None

    def _extract_content(self, context: GradeContext) -> str | None:
        """Extract content to evaluate from context."""
        if self.source == "output_data":
            if context.outcome and context.outcome.output_data:
                data = context.outcome.output_data
                return data if isinstance(data, str) else json.dumps(data, indent=2)
        elif self.source == "text_artifact":
            artifact = context.text_artifact
            if artifact:
                return artifact.content
        elif self.source.startswith("metadata."):
            key = self.source[9:]  # Remove "metadata." prefix
            if context.metadata and key in context.metadata:
                data = context.metadata[key]
                return data if isinstance(data, str) else json.dumps(data, indent=2)

        return None

    def _build_context_info(self, context: GradeContext) -> str:
        """Build context information string."""
        if self.context_template:
            # Simple template substitution
            info = self.context_template
            info = info.replace("{prompt}", context.prompt or "")
            info = info.replace("{negative_prompt}", context.negative_prompt or "")
            return info
        elif context.prompt:
            return f"Original prompt: {context.prompt}"
        return ""

    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade content against the rubric using LLM."""
        # Validate configuration
        if not self.criteria:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error="No criteria configured in rubric",
            )

        # Extract content
        content = self._extract_content(context)
        if not content:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error=f"No content found in source '{self.source}'",
            )

        # Build prompt
        context_info = self._build_context_info(context)
        prompt = _build_evaluation_prompt(content, self.criteria, context_info, self._template)

        try:
            # Call LLM with structured output
            evaluation_data = await self._call_llm_structured(prompt)

            # Parse into RubricEvaluation
            evaluation = RubricEvaluation.from_structured_output(evaluation_data, self.criteria)

            # Build result
            passed = evaluation.overall_score >= self.pass_threshold

            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=passed,
                score=evaluation.overall_score,
                reasoning=evaluation.overall_reasoning,
                details={
                    "criteria_results": [
                        {
                            "name": cr.name,
                            "score": cr.score,
                            "weight": cr.weight,
                            "weighted_score": cr.weighted_score,
                            "reasoning": cr.reasoning,
                        }
                        for cr in evaluation.criteria_results
                    ],
                    "confidence": evaluation.confidence,
                    "suggestions": evaluation.suggestions,
                    "pass_threshold": self.pass_threshold,
                    "model": self.model,
                    "provider": self.provider,
                },
            )
        except Exception as e:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error=f"LLM evaluation failed: {str(e)}",
            )

    async def _call_llm_structured(self, prompt: str) -> dict[str, Any]:
        """Call LLM with structured output and return parsed JSON."""
        if self.provider == "openai":
            return await self._call_openai_structured(prompt)
        elif self.provider == "anthropic":
            return await self._call_anthropic_structured(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _call_openai_structured(self, prompt: str) -> dict[str, Any]:
        """Call OpenAI API with structured output."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required for OpenAI provider. Install with: pip install openai")

        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        default_system = (
            "You are an expert evaluator. "
            "Respond only with valid JSON matching the required schema."
        )
        system_content = self._template.role or default_system

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_content,
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rubric_evaluation",
                    "strict": True,
                    "schema": self._schema,
                },
            },
            temperature=0.1,  # Low temperature for consistent evaluations
        )

        content = response.choices[0].message.content
        return json.loads(content)

    async def _call_anthropic_structured(self, prompt: str) -> dict[str, Any]:
        """Call Anthropic API with tool use for structured output."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package required for Anthropic provider. Install with: pip install anthropic")

        client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        # Use tool use for structured output with Claude
        response = await client.messages.create(
            model=self.model if "claude" in self.model else "claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=[
                {
                    "name": "submit_evaluation",
                    "description": "Submit the rubric evaluation results",
                    "input_schema": self._schema,
                }
            ],
            tool_choice={"type": "tool", "name": "submit_evaluation"},
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract tool use result
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_evaluation":
                return block.input

        raise ValueError("No tool use response from Claude")

    def validate_config(self) -> bool:
        """Validate grader configuration."""
        if not self.criteria:
            return False
        for criterion in self.criteria:
            if not criterion.name or not criterion.description:
                return False
        return True
