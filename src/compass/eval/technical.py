"""Technical quality evaluator."""

from typing import Any

from PIL import Image
import statistics

from compass.core.result import EvaluatorResult
from compass.eval.base import Evaluator, EvalContext
from compass.eval.registry import register_evaluator


@register_evaluator("technical_quality")
class TechnicalQualityEvaluator(Evaluator):
    """Evaluator for technical image quality metrics."""

    name = "technical_quality"

    AVAILABLE_CHECKS = ["resolution", "sharpness", "noise_level", "contrast", "brightness"]

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.checks = self.config.get("checks", ["resolution", "sharpness"])
        self.min_resolution = self.config.get("min_resolution", 512 * 512)
        self.min_sharpness = self.config.get("min_sharpness", 100)

    async def evaluate(self, image: Image.Image, context: EvalContext) -> EvaluatorResult:
        """Evaluate technical quality of image.

        Args:
            image: Generated image.
            context: Evaluation context.

        Returns:
            EvaluatorResult with technical quality metrics.
        """
        try:
            check_results = {}
            scores = []

            for check in self.checks:
                if check not in self.AVAILABLE_CHECKS:
                    continue

                score, details = await self._run_check(check, image)
                check_results[check] = {
                    "score": score,
                    "details": details,
                }
                scores.append(score)

            # Average score across all checks
            overall_score = sum(scores) / len(scores) if scores else 0.0
            passed = overall_score >= 0.7

            return EvaluatorResult(
                name=self.name,
                score=overall_score,
                passed=passed,
                metadata={
                    "checks": self.checks,
                    "results": check_results,
                },
            )
        except Exception as e:
            return EvaluatorResult(
                name=self.name,
                score=0.0,
                passed=False,
                error=str(e),
            )

    async def _run_check(
        self, check_type: str, image: Image.Image
    ) -> tuple[float, dict[str, Any]]:
        """Run a specific technical quality check.

        Args:
            check_type: Type of check to run.
            image: Image to check.

        Returns:
            Tuple of (score 0-1, details).
        """
        match check_type:
            case "resolution":
                return self._check_resolution(image)
            case "sharpness":
                return self._check_sharpness(image)
            case "noise_level":
                return self._check_noise(image)
            case "contrast":
                return self._check_contrast(image)
            case "brightness":
                return self._check_brightness(image)
            case _:
                return 1.0, {"message": f"Unknown check: {check_type}"}

    def _check_resolution(self, image: Image.Image) -> tuple[float, dict[str, Any]]:
        """Check image resolution.

        Args:
            image: Image to check.

        Returns:
            Tuple of (score, details).
        """
        width, height = image.size
        total_pixels = width * height

        # Score based on resolution meeting minimum
        if total_pixels >= self.min_resolution:
            score = 1.0
        else:
            score = total_pixels / self.min_resolution

        return score, {
            "width": width,
            "height": height,
            "total_pixels": total_pixels,
            "min_required": self.min_resolution,
        }

    def _check_sharpness(self, image: Image.Image) -> tuple[float, dict[str, Any]]:
        """Check image sharpness using Laplacian variance.

        Args:
            image: Image to check.

        Returns:
            Tuple of (score, details).
        """
        # Convert to grayscale
        gray = image.convert("L")

        # Simple edge detection approximation using pixel differences
        pixels = list(gray.getdata())
        width, height = gray.size

        edge_values = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                idx = y * width + x
                # Laplacian approximation
                lap = (
                    pixels[idx - width] +
                    pixels[idx + width] +
                    pixels[idx - 1] +
                    pixels[idx + 1] -
                    4 * pixels[idx]
                )
                edge_values.append(abs(lap))

        if not edge_values:
            return 0.5, {"sharpness_value": 0, "min_required": self.min_sharpness}

        # Variance of Laplacian
        sharpness = statistics.variance(edge_values) if len(edge_values) > 1 else 0

        # Normalize score
        score = min(1.0, sharpness / self.min_sharpness) if self.min_sharpness > 0 else 1.0

        return score, {
            "sharpness_value": sharpness,
            "min_required": self.min_sharpness,
        }

    def _check_noise(self, image: Image.Image) -> tuple[float, dict[str, Any]]:
        """Check image noise level.

        Args:
            image: Image to check.

        Returns:
            Tuple of (score, details) where higher score = less noise.
        """
        # Convert to grayscale
        gray = image.convert("L")
        pixels = list(gray.getdata())

        # Estimate noise using local variance
        width, height = gray.size
        block_size = 8
        local_variances = []

        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = []
                for by in range(block_size):
                    for bx in range(block_size):
                        idx = (y + by) * width + (x + bx)
                        block.append(pixels[idx])
                if len(block) > 1:
                    local_variances.append(statistics.variance(block))

        if not local_variances:
            return 1.0, {"noise_estimate": 0}

        # Median of local variances as noise estimate
        noise_estimate = statistics.median(local_variances)

        # Lower noise = higher score (inverse relationship)
        # Assume noise > 500 is very noisy
        score = max(0.0, 1.0 - noise_estimate / 500)

        return score, {"noise_estimate": noise_estimate}

    def _check_contrast(self, image: Image.Image) -> tuple[float, dict[str, Any]]:
        """Check image contrast.

        Args:
            image: Image to check.

        Returns:
            Tuple of (score, details).
        """
        gray = image.convert("L")
        min_val, max_val = gray.getextrema()

        contrast = max_val - min_val

        # Ideal contrast is around 200-255
        score = min(1.0, contrast / 200)

        return score, {
            "min_value": min_val,
            "max_value": max_val,
            "contrast": contrast,
        }

    def _check_brightness(self, image: Image.Image) -> tuple[float, dict[str, Any]]:
        """Check image brightness.

        Args:
            image: Image to check.

        Returns:
            Tuple of (score, details).
        """
        gray = image.convert("L")
        pixels = list(gray.getdata())
        avg_brightness = sum(pixels) / len(pixels) if pixels else 128

        # Ideal brightness is around 100-160 (middle range)
        if 100 <= avg_brightness <= 160:
            score = 1.0
        elif avg_brightness < 100:
            score = avg_brightness / 100
        else:
            score = max(0.0, 1.0 - (avg_brightness - 160) / 95)

        return score, {"average_brightness": avg_brightness}
