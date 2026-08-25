"""Tests for RubricGrader with structured output."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from compass.core.artifacts import TextArtifact
from compass.core.transcript import Outcome, Transcript
from compass.graders.base import GradeContext
from compass.graders.model.prompt_template import RubricPromptTemplate
from compass.graders.model.rubric import (
    Criterion,
    CriterionResult,
    RubricEvaluation,
    RubricGrader,
    _build_evaluation_prompt,
    _build_evaluation_schema,
)


class TestCriterion:
    """Tests for Criterion dataclass."""

    def test_from_dict_basic(self):
        """Basic criterion creation from dict."""
        data = {
            "name": "accuracy",
            "description": "Factual correctness",
        }
        criterion = Criterion.from_dict(data)

        assert criterion.name == "accuracy"
        assert criterion.description == "Factual correctness"
        assert criterion.weight == 1.0
        assert criterion.score_guidance is None

    def test_from_dict_with_weight(self):
        """Criterion with custom weight."""
        data = {
            "name": "completeness",
            "description": "Coverage of all points",
            "weight": 2.0,
        }
        criterion = Criterion.from_dict(data)

        assert criterion.weight == 2.0

    def test_from_dict_with_score_guidance(self):
        """Criterion with scoring guidance."""
        data = {
            "name": "clarity",
            "description": "Clear presentation",
            "score_guidance": {
                "0": "Incomprehensible",
                "0.5": "Somewhat clear",
                "1": "Crystal clear",
            },
        }
        criterion = Criterion.from_dict(data)

        assert criterion.score_guidance is not None
        assert criterion.score_guidance["0"] == "Incomprehensible"


class TestCriterionResult:
    """Tests for CriterionResult dataclass."""

    def test_weighted_score(self):
        """Weighted score calculation."""
        result = CriterionResult(
            name="accuracy",
            score=0.8,
            reasoning="Good accuracy",
            weight=2.0,
        )

        assert result.weighted_score == 1.6


class TestRubricEvaluation:
    """Tests for RubricEvaluation parsing."""

    def test_from_structured_output(self):
        """Parse structured output into RubricEvaluation."""
        criteria = [
            Criterion(name="accuracy", description="Factual correctness", weight=2.0),
            Criterion(name="clarity", description="Clear presentation", weight=1.0),
        ]

        data = {
            "overall_score": 0.85,
            "criteria_scores": {
                "accuracy": {"score": 0.9, "reasoning": "Very accurate"},
                "clarity": {"score": 0.8, "reasoning": "Clear enough"},
            },
            "overall_reasoning": "Good overall performance",
            "confidence": 0.95,
            "suggestions": ["Add more examples"],
        }

        evaluation = RubricEvaluation.from_structured_output(data, criteria)

        assert evaluation.overall_score == 0.85
        assert evaluation.overall_reasoning == "Good overall performance"
        assert evaluation.confidence == 0.95
        assert evaluation.suggestions == ["Add more examples"]
        assert len(evaluation.criteria_results) == 2
        assert evaluation.criteria_results[0].name == "accuracy"
        assert evaluation.criteria_results[0].score == 0.9
        assert evaluation.criteria_results[0].weight == 2.0

    def test_from_structured_output_missing_criterion(self):
        """Handle missing criterion in output gracefully."""
        criteria = [
            Criterion(name="accuracy", description="Factual correctness"),
            Criterion(name="clarity", description="Clear presentation"),
        ]

        data = {
            "overall_score": 0.5,
            "criteria_scores": {
                "accuracy": {"score": 0.5, "reasoning": "OK"},
                # clarity is missing
            },
            "overall_reasoning": "Partial evaluation",
        }

        evaluation = RubricEvaluation.from_structured_output(data, criteria)

        assert len(evaluation.criteria_results) == 2
        # Missing criterion gets default values
        clarity_result = next(r for r in evaluation.criteria_results if r.name == "clarity")
        assert clarity_result.score == 0.0
        assert clarity_result.reasoning == ""


class TestBuildEvaluationSchema:
    """Tests for schema building."""

    def test_schema_structure(self):
        """Verify schema has correct structure."""
        criteria = [
            Criterion(name="accuracy", description="Factual correctness"),
            Criterion(name="clarity", description="Clear presentation"),
        ]

        schema = _build_evaluation_schema(criteria)

        assert schema["type"] == "object"
        assert "overall_score" in schema["properties"]
        assert "criteria_scores" in schema["properties"]
        assert "overall_reasoning" in schema["properties"]

        criteria_props = schema["properties"]["criteria_scores"]["properties"]
        assert "accuracy" in criteria_props
        assert "clarity" in criteria_props

    def test_schema_required_fields(self):
        """Required fields are properly set."""
        criteria = [Criterion(name="test", description="Test criterion")]
        schema = _build_evaluation_schema(criteria)

        assert "overall_score" in schema["required"]
        assert "criteria_scores" in schema["required"]
        assert "overall_reasoning" in schema["required"]


class TestBuildEvaluationPrompt:
    """Tests for prompt building."""

    def test_prompt_includes_criteria(self):
        """Prompt includes all criteria."""
        criteria = [
            Criterion(name="accuracy", description="Factual correctness", weight=2.0),
            Criterion(name="clarity", description="Clear presentation", weight=1.0),
        ]

        prompt = _build_evaluation_prompt("Test content", criteria)

        assert "accuracy" in prompt
        assert "Factual correctness" in prompt
        assert "weight: 2.0" in prompt
        assert "clarity" in prompt
        assert "Clear presentation" in prompt
        assert "Test content" in prompt

    def test_prompt_includes_context(self):
        """Prompt includes context info."""
        criteria = [Criterion(name="test", description="Test")]
        prompt = _build_evaluation_prompt("Content", criteria, "Additional context here")

        assert "Additional context here" in prompt

    def test_prompt_includes_score_guidance(self):
        """Prompt includes scoring guidance if provided."""
        criteria = [
            Criterion(
                name="quality",
                description="Quality check",
                score_guidance={"0": "Bad", "1": "Good"},
            )
        ]

        prompt = _build_evaluation_prompt("Content", criteria)

        assert "Scoring guide:" in prompt
        assert "0=Bad" in prompt
        assert "1=Good" in prompt


class TestRubricGrader:
    """Tests for RubricGrader."""

    @pytest.fixture
    def basic_config(self):
        """Basic grader configuration."""
        return {
            "criteria": [
                {"name": "accuracy", "description": "Factual correctness", "weight": 2.0},
                {"name": "clarity", "description": "Clear presentation", "weight": 1.0},
            ],
            "pass_threshold": 0.7,
            "model": "gpt-4o",
            "provider": "openai",
        }

    @pytest.fixture
    def make_context(self):
        """Factory to create GradeContext."""

        def _make(content: str, prompt: str = "") -> GradeContext:
            transcript = Transcript(task_id="test-task", trial_id="test-trial")
            outcome = Outcome(output_data=content)
            return GradeContext(
                transcript=transcript,
                outcome=outcome,
                prompt=prompt,
            )

        return _make

    def test_init_parses_criteria(self, basic_config):
        """Initialization parses criteria correctly."""
        grader = RubricGrader(basic_config)

        assert len(grader.criteria) == 2
        assert grader.criteria[0].name == "accuracy"
        assert grader.criteria[0].weight == 2.0
        assert grader.pass_threshold == 0.7

    def test_init_default_values(self):
        """Default values are set correctly."""
        grader = RubricGrader({})

        assert grader.criteria == []
        assert grader.pass_threshold == 0.7
        assert grader.model == "gpt-4o"
        assert grader.provider == "openai"

    @pytest.mark.asyncio
    async def test_grade_no_criteria_fails(self, make_context):
        """Grade fails when no criteria configured."""
        grader = RubricGrader({})
        context = make_context("Test content")

        result = await grader.grade(context)

        assert result.passed is False
        assert "No criteria configured" in result.error

    @pytest.mark.asyncio
    async def test_grade_no_content_fails(self, basic_config):
        """Grade fails when no content available."""
        grader = RubricGrader(basic_config)
        context = GradeContext(
            transcript=Transcript(task_id="test", trial_id="test"),
            outcome=Outcome(),  # No output_data
        )

        result = await grader.grade(context)

        assert result.passed is False
        assert "No content found" in result.error

    @pytest.mark.asyncio
    async def test_grade_success(self, basic_config, make_context):
        """Successful grading with mocked LLM."""
        grader = RubricGrader(basic_config)
        context = make_context("Test content to evaluate", "Evaluate this")

        mock_response = {
            "overall_score": 0.85,
            "criteria_scores": {
                "accuracy": {"score": 0.9, "reasoning": "Very accurate"},
                "clarity": {"score": 0.8, "reasoning": "Clear"},
            },
            "overall_reasoning": "Good overall",
            "confidence": 0.95,
            "suggestions": ["Add examples"],
        }

        with patch.object(grader, "_call_llm_structured", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await grader.grade(context)

        assert result.passed is True
        assert result.score == 0.85
        assert result.reasoning == "Good overall"
        assert len(result.details["criteria_results"]) == 2
        assert result.details["confidence"] == 0.95
        assert result.details["suggestions"] == ["Add examples"]

    @pytest.mark.asyncio
    async def test_grade_below_threshold(self, basic_config, make_context):
        """Grading below threshold fails."""
        grader = RubricGrader(basic_config)
        context = make_context("Poor content")

        mock_response = {
            "overall_score": 0.5,  # Below 0.7 threshold
            "criteria_scores": {
                "accuracy": {"score": 0.4, "reasoning": "Inaccurate"},
                "clarity": {"score": 0.6, "reasoning": "Unclear"},
            },
            "overall_reasoning": "Needs improvement",
        }

        with patch.object(grader, "_call_llm_structured", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await grader.grade(context)

        assert result.passed is False
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_grade_llm_error(self, basic_config, make_context):
        """Handle LLM errors gracefully."""
        grader = RubricGrader(basic_config)
        context = make_context("Test content")

        with patch.object(grader, "_call_llm_structured", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("API error")
            result = await grader.grade(context)

        assert result.passed is False
        assert "LLM evaluation failed" in result.error

    @pytest.mark.asyncio
    async def test_extract_content_from_output_data(self, basic_config, make_context):
        """Extract content from output_data (default)."""
        grader = RubricGrader(basic_config)
        context = make_context("Content from output_data")

        content = grader._extract_content(context)
        assert content == "Content from output_data"

    @pytest.mark.asyncio
    async def test_extract_content_from_text_artifact(self, basic_config):
        """Extract content from text_artifact."""
        config = {**basic_config, "source": "text_artifact"}
        grader = RubricGrader(config)

        outcome = Outcome(artifacts=[TextArtifact(content="Content from artifact")])
        context = GradeContext(
            transcript=Transcript(task_id="test", trial_id="test"),
            outcome=outcome,
        )

        content = grader._extract_content(context)
        assert content == "Content from artifact"

    @pytest.mark.asyncio
    async def test_extract_content_from_metadata(self, basic_config):
        """Extract content from metadata."""
        config = {**basic_config, "source": "metadata.response"}
        grader = RubricGrader(config)

        context = GradeContext(
            transcript=Transcript(task_id="test", trial_id="test"),
            outcome=Outcome(),
            metadata={"response": "Content from metadata"},
        )

        content = grader._extract_content(context)
        assert content == "Content from metadata"

    @pytest.mark.asyncio
    async def test_extract_content_json_serialization(self, basic_config, make_context):
        """Dict content is JSON serialized."""
        grader = RubricGrader(basic_config)

        outcome = Outcome(output_data={"key": "value", "nested": {"a": 1}})
        context = GradeContext(
            transcript=Transcript(task_id="test", trial_id="test"),
            outcome=outcome,
        )

        content = grader._extract_content(context)
        parsed = json.loads(content)
        assert parsed["key"] == "value"
        assert parsed["nested"]["a"] == 1

    def test_validate_config_valid(self, basic_config):
        """Valid config passes validation."""
        grader = RubricGrader(basic_config)
        assert grader.validate_config() is True

    def test_validate_config_no_criteria(self):
        """No criteria fails validation."""
        grader = RubricGrader({})
        assert grader.validate_config() is False

    def test_validate_config_invalid_criterion(self):
        """Invalid criterion fails validation."""
        config = {
            "criteria": [
                {"name": "", "description": "Missing name"},  # Empty name
            ]
        }
        grader = RubricGrader(config)
        assert grader.validate_config() is False

    @pytest.mark.asyncio
    async def test_context_template_substitution(self, basic_config, make_context):
        """Context template substitution works."""
        config = {
            **basic_config,
            "context_template": "Task: {prompt}, Avoid: {negative_prompt}",
        }
        grader = RubricGrader(config)

        context = GradeContext(
            transcript=Transcript(task_id="test", trial_id="test"),
            outcome=Outcome(output_data="content"),
            prompt="Write a poem",
            negative_prompt="No violence",
        )

        context_info = grader._build_context_info(context)
        assert "Task: Write a poem" in context_info
        assert "Avoid: No violence" in context_info

    @pytest.mark.asyncio
    async def test_criteria_results_in_details(self, basic_config, make_context):
        """Criteria results are properly structured in details."""
        grader = RubricGrader(basic_config)
        context = make_context("Test content")

        mock_response = {
            "overall_score": 0.8,
            "criteria_scores": {
                "accuracy": {"score": 0.9, "reasoning": "Accurate"},
                "clarity": {"score": 0.7, "reasoning": "Clear"},
            },
            "overall_reasoning": "Good",
        }

        with patch.object(grader, "_call_llm_structured", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await grader.grade(context)

        criteria_results = result.details["criteria_results"]

        # Check accuracy criterion
        accuracy = next(r for r in criteria_results if r["name"] == "accuracy")
        assert accuracy["score"] == 0.9
        assert accuracy["weight"] == 2.0
        assert accuracy["weighted_score"] == 1.8
        assert accuracy["reasoning"] == "Accurate"

        # Check clarity criterion
        clarity = next(r for r in criteria_results if r["name"] == "clarity")
        assert clarity["score"] == 0.7
        assert clarity["weight"] == 1.0


class TestRubricGraderProviders:
    """Tests for different LLM providers."""

    @pytest.fixture
    def openai_config(self):
        return {
            "criteria": [{"name": "test", "description": "Test"}],
            "provider": "openai",
            "model": "gpt-4o",
        }

    @pytest.fixture
    def anthropic_config(self):
        return {
            "criteria": [{"name": "test", "description": "Test"}],
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
        }

    @pytest.mark.asyncio
    async def test_unsupported_provider(self):
        """Unsupported provider raises error."""
        config = {
            "criteria": [{"name": "test", "description": "Test"}],
            "provider": "unsupported",
        }
        grader = RubricGrader(config)

        with pytest.raises(ValueError, match="Unsupported provider"):
            await grader._call_llm_structured("test prompt")

    @pytest.mark.asyncio
    async def test_openai_import_error(self, openai_config):
        """OpenAI import error is handled.

        The provider plumbing moved to ``compass.adapters.llm`` so one place
        knows how to ask each provider for a schema; this checks the message a
        user without the package actually sees.
        """
        from compass.adapters.llm import structured_completion

        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai package required"):
                await structured_completion(
                    "test", schema={}, model="gpt-4o", provider="openai"
                )

    @pytest.mark.asyncio
    async def test_anthropic_import_error(self, anthropic_config):
        """Anthropic import error is handled."""
        from compass.adapters.llm import structured_completion

        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic package required"):
                await structured_completion(
                    "test", schema={}, model="claude-sonnet-4-20250514",
                    provider="anthropic",
                )


class TestRubricGraderStructuredPrompt:
    """Tests for RubricGrader integration with RubricPromptTemplate."""

    @pytest.fixture
    def structured_config(self):
        """Config with structured template fields."""
        return {
            "criteria": [
                {"name": "accuracy", "description": "Factual correctness", "weight": 2.0},
                {"name": "clarity", "description": "Clear presentation", "weight": 1.0},
            ],
            "pass_threshold": 0.7,
            "model": "gpt-4o",
            "provider": "openai",
            "role": "You are an expert text rendering evaluator",
            "scope_constraints": [
                "Only evaluate text accuracy, NOT aesthetic quality",
            ],
            "verdict_rules": [
                "If any text is garbled, overall score must be below 0.5",
            ],
        }

    @pytest.fixture
    def make_context(self):
        """Factory to create GradeContext."""

        def _make(content: str, prompt: str = "") -> GradeContext:
            transcript = Transcript(task_id="test-task", trial_id="test-trial")
            outcome = Outcome(output_data=content)
            return GradeContext(
                transcript=transcript,
                outcome=outcome,
                prompt=prompt,
            )

        return _make

    def test_init_with_structured_config(self, structured_config):
        """Template is created from structured config fields."""
        grader = RubricGrader(structured_config)

        assert grader._template.has_structured_sections is True
        assert grader._template.role == "You are an expert text rendering evaluator"
        assert len(grader._template.scope_constraints) == 1
        assert len(grader._template.verdict_rules) == 1

    def test_init_without_structured_config(self):
        """Template without structured fields has no sections."""
        grader = RubricGrader({
            "criteria": [{"name": "test", "description": "Test"}],
        })
        assert grader._template.has_structured_sections is False

    def test_prompt_with_template_includes_xml_tags(self, structured_config):
        """Prompt built with a structured template includes XML tags."""
        criteria = [
            Criterion(name="accuracy", description="Factual correctness", weight=2.0),
        ]
        template = RubricPromptTemplate.from_config(structured_config)

        prompt = _build_evaluation_prompt("Test content", criteria, "Some context", template)

        assert "<role>" in prompt
        assert "<scope_constraints>" in prompt
        assert "<verdict_rules>" in prompt
        assert "<metrics_and_scoring>" in prompt
        assert "<content>" in prompt

    def test_prompt_without_template_matches_legacy(self):
        """Prompt without template matches the legacy format exactly."""
        criteria = [
            Criterion(name="accuracy", description="Factual correctness", weight=2.0),
        ]

        prompt_no_template = _build_evaluation_prompt("Test content", criteria, "ctx")
        prompt_none_template = _build_evaluation_prompt("Test content", criteria, "ctx", None)
        prompt_empty_template = _build_evaluation_prompt(
            "Test content", criteria, "ctx", RubricPromptTemplate()
        )

        # All three should produce the same legacy format
        assert prompt_no_template == prompt_none_template
        assert prompt_no_template == prompt_empty_template
        assert "<role>" not in prompt_no_template
        assert "## Evaluation Criteria" in prompt_no_template

    @pytest.mark.asyncio
    async def test_grade_with_structured_prompt(self, structured_config, make_context):
        """Grading with structured config passes XML-tagged prompt to LLM."""
        grader = RubricGrader(structured_config)
        context = make_context("Test content to evaluate", "Evaluate this")

        mock_response = {
            "overall_score": 0.85,
            "criteria_scores": {
                "accuracy": {"score": 0.9, "reasoning": "Very accurate"},
                "clarity": {"score": 0.8, "reasoning": "Clear"},
            },
            "overall_reasoning": "Good overall",
        }

        with patch.object(grader, "_call_llm_structured", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await grader.grade(context)

            # Verify the prompt passed to LLM contains XML tags
            call_args = mock_llm.call_args[0][0]
            assert "<role>" in call_args
            assert "<scope_constraints>" in call_args
            assert "<verdict_rules>" in call_args

        assert result.passed is True
        assert result.score == 0.85
