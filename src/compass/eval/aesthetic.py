"""Aesthetic scoring evaluator."""

from typing import Any

from PIL import Image

from compass.core.result import EvaluatorResult
from compass.eval.base import Evaluator, EvalContext
from compass.eval.registry import register_evaluator


@register_evaluator("aesthetic_score")
class AestheticScoreEvaluator(Evaluator):
    """Evaluator for image aesthetic quality."""

    name = "aesthetic_score"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.model_name = self.config.get("model", "laion/aesthetic-predictor-v2")
        self.min_score = self.config.get("min_score", 5.0)
        self.max_score = self.config.get("max_score", 10.0)
        self._model = None

    async def evaluate(self, image: Image.Image, context: EvalContext) -> EvaluatorResult:
        """Evaluate aesthetic quality of image.

        Args:
            image: Generated image.
            context: Evaluation context.

        Returns:
            EvaluatorResult with aesthetic score.
        """
        try:
            raw_score = await self._compute_aesthetic_score(image)
            # Normalize to 0-1 range
            normalized_score = (raw_score - 1) / (self.max_score - 1)
            normalized_score = max(0.0, min(1.0, normalized_score))

            passed = raw_score >= self.min_score

            return EvaluatorResult(
                name=self.name,
                score=normalized_score,
                passed=passed,
                metadata={
                    "model": self.model_name,
                    "raw_score": raw_score,
                    "min_score": self.min_score,
                },
            )
        except Exception as e:
            return EvaluatorResult(
                name=self.name,
                score=0.0,
                passed=False,
                error=str(e),
            )

    async def _compute_aesthetic_score(self, image: Image.Image) -> float:
        """Compute aesthetic score.

        Args:
            image: Image to evaluate.

        Returns:
            Aesthetic score (typically 1-10).
        """
        # Lazy load model
        if self._model is None:
            await self._load_model()

        # TODO: Implement actual aesthetic scoring
        # This is a placeholder - actual implementation would use LAION aesthetic predictor

        # Placeholder return
        return 6.0

    async def _load_model(self) -> None:
        """Load aesthetic model lazily."""
        # TODO: Implement model loading
        pass
