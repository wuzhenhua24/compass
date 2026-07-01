"""VLM (Vision Language Model) based evaluator."""

import base64
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from compass.core.result import EvaluatorResult
from compass.eval.base import Evaluator, EvalContext
from compass.eval.registry import register_evaluator


@register_evaluator("vlm_judge")
class VLMJudgeEvaluator(Evaluator):
    """Evaluator using VLM for complex semantic evaluation."""

    name = "vlm_judge"

    DEFAULT_SYSTEM_PROMPT = """You are an image quality evaluator. Analyze the given image against the provided criteria.
For each criterion, respond with:
- PASS if the criterion is met
- FAIL if the criterion is not met

Then provide an overall score from 0 to 1 based on how many criteria passed."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.model = self.config.get("model", "gpt-4o")
        self.criteria = self.config.get("criteria", [])
        self.api_key = self.config.get("api_key", "")
        self.api_base = self.config.get("api_base", "https://api.openai.com/v1")
        self.threshold = self.config.get("threshold", 0.7)

    async def evaluate(self, image: Image.Image, context: EvalContext) -> EvaluatorResult:
        """Evaluate image using VLM.

        Args:
            image: Generated image.
            context: Evaluation context with prompt.

        Returns:
            EvaluatorResult with VLM evaluation.
        """
        try:
            # Build criteria from config or context
            criteria = self.criteria or self._extract_criteria(context)

            if not criteria:
                return EvaluatorResult(
                    name=self.name,
                    score=0.0,
                    passed=False,
                    error="No evaluation criteria provided",
                )

            result = await self._call_vlm(image, context.prompt, criteria)

            return EvaluatorResult(
                name=self.name,
                score=result["score"],
                passed=result["score"] >= self.threshold,
                metadata={
                    "model": self.model,
                    "criteria": criteria,
                    "criterion_results": result["criterion_results"],
                    "reasoning": result.get("reasoning", ""),
                },
            )
        except Exception as e:
            return EvaluatorResult(
                name=self.name,
                score=0.0,
                passed=False,
                error=str(e),
            )

    def _extract_criteria(self, context: EvalContext) -> list[str]:
        """Extract evaluation criteria from context."""
        # Default criteria based on prompt
        return [
            f"The image matches the description: {context.prompt}",
            "The image has good visual quality",
            "The image does not contain artifacts or distortions",
        ]

    async def _call_vlm(
        self, image: Image.Image, prompt: str, criteria: list[str]
    ) -> dict[str, Any]:
        """Call VLM API for evaluation.

        Args:
            image: Image to evaluate.
            prompt: Original prompt.
            criteria: List of criteria to check.

        Returns:
            Dictionary with score and criterion results.
        """
        # Convert image to base64
        image_base64 = self._image_to_base64(image)

        # Build evaluation prompt
        criteria_text = "\n".join(f"- {c}" for c in criteria)
        user_prompt = f"""Original prompt: {prompt}

Evaluate this image against the following criteria:
{criteria_text}

For each criterion, state PASS or FAIL with a brief explanation.
Then provide an overall score from 0.0 to 1.0."""

        # TODO: Implement actual API call
        # This is a placeholder - actual implementation would call OpenAI/Anthropic API

        # Placeholder return
        return {
            "score": 0.8,
            "criterion_results": {c: "PASS" for c in criteria},
            "reasoning": "Placeholder evaluation",
        }

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
