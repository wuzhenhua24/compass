"""Tests for evaluators."""

import pytest
from PIL import Image

from compass.eval import register_evaluator, get_evaluator, list_evaluators, Evaluator, EvalContext
from compass.eval.registry import unregister_evaluator
from compass.core.result import EvaluatorResult


class TestEvaluatorRegistry:
    """Tests for evaluator registry."""

    def test_register_and_get(self):
        """Test registering and retrieving an evaluator."""

        @register_evaluator("test_eval")
        class TestEvaluator(Evaluator):
            async def evaluate(self, image, context):
                return EvaluatorResult(name="test", score=1.0, passed=True)

        evaluator_cls = get_evaluator("test_eval")
        assert evaluator_cls == TestEvaluator

        # Cleanup
        unregister_evaluator("test_eval")

    def test_get_nonexistent(self):
        """Test getting nonexistent evaluator raises KeyError."""
        with pytest.raises(KeyError):
            get_evaluator("nonexistent_evaluator")

    def test_list_evaluators(self):
        """Test listing evaluators."""
        evaluators = list_evaluators()
        assert isinstance(evaluators, list)
        # Built-in evaluators should be registered
        assert "semantic_match" in evaluators
        assert "aesthetic_score" in evaluators
        assert "vlm_judge" in evaluators
        assert "safety_check" in evaluators


class TestEvalContext:
    """Tests for EvalContext."""

    def test_basic_context(self):
        """Test creating basic context."""
        context = EvalContext(prompt="A cat")

        assert context.prompt == "A cat"
        assert context.negative_prompt == ""
        assert context.params == {}

    def test_full_context(self):
        """Test creating context with all fields."""
        context = EvalContext(
            prompt="A cat",
            negative_prompt="blurry",
            params={"width": 1024},
            metadata={"source": "test"},
        )

        assert context.prompt == "A cat"
        assert context.negative_prompt == "blurry"
        assert context.params["width"] == 1024
        assert context.metadata["source"] == "test"


class TestSemanticMatchEvaluator:
    """Tests for SemanticMatchEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_result(self):
        """Test that evaluate returns a valid result."""
        evaluator_cls = get_evaluator("semantic_match")
        evaluator = evaluator_cls({"threshold": 0.2})

        # Create a simple test image
        image = Image.new("RGB", (100, 100), color="red")
        context = EvalContext(prompt="A red square")

        result = await evaluator.evaluate(image, context)

        assert isinstance(result, EvaluatorResult)
        assert isinstance(result.score, float)
        assert isinstance(result.passed, bool)


class TestSafetyCheckEvaluator:
    """Tests for SafetyCheckEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_result(self):
        """Test that safety check returns a valid result."""
        evaluator_cls = get_evaluator("safety_check")
        evaluator = evaluator_cls({"checks": ["nsfw"]})

        image = Image.new("RGB", (100, 100), color="white")
        context = EvalContext(prompt="Test")

        result = await evaluator.evaluate(image, context)

        assert isinstance(result, EvaluatorResult)
        assert "results" in result.metadata
