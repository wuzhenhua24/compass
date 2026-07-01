"""Technical quality grader for image quality metrics."""

import statistics
from typing import Any

from PIL import Image

from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.registry import register_grader


@register_grader("technical_quality")
class TechnicalQualityGrader(CodeGrader):
    """Deterministic technical quality grader.

    Scope: OUTCOME - evaluates the final image's technical properties.
    """

    name = "technical_quality"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    AVAILABLE_CHECKS = ["resolution", "sharpness", "noise", "contrast", "brightness"]

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.checks = self.config.get("checks", ["resolution", "sharpness"])
        self.min_resolution = self.config.get("min_resolution", 512 * 512)
        self.min_sharpness = self.config.get("min_sharpness", 100)
        self.max_noise = self.config.get("max_noise", 500)
        self.min_contrast = self.config.get("min_contrast", 100)

    async def grade(self, context: GradeContext) -> GradeResult:
        """Run technical quality checks on the outcome image."""
        image = context.image

        if image is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error="No image in outcome",
            )

        check_results = {}
        scores = []

        for check in self.checks:
            if check not in self.AVAILABLE_CHECKS:
                continue

            score, details = self._run_check(check, image)
            check_results[check] = {
                "score": score,
                "passed": score >= 0.7,
                "details": details,
            }
            scores.append(score)

        if not scores:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=True,
                score=1.0,
                details={"message": "No checks configured"},
            )

        overall_score = sum(scores) / len(scores)
        passed = overall_score >= 0.7

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=overall_score,
            details={
                "checks": check_results,
                "overall_score": overall_score,
            },
        )

    def _run_check(self, check_type: str, image: Image.Image) -> tuple[float, dict]:
        """Run a specific check."""
        match check_type:
            case "resolution":
                return self._check_resolution(image)
            case "sharpness":
                return self._check_sharpness(image)
            case "noise":
                return self._check_noise(image)
            case "contrast":
                return self._check_contrast(image)
            case "brightness":
                return self._check_brightness(image)
            case _:
                return 1.0, {}

    def _check_resolution(self, image: Image.Image) -> tuple[float, dict]:
        width, height = image.size
        total_pixels = width * height
        score = 1.0 if total_pixels >= self.min_resolution else total_pixels / self.min_resolution
        return score, {"width": width, "height": height, "total_pixels": total_pixels}

    def _check_sharpness(self, image: Image.Image) -> tuple[float, dict]:
        gray = image.convert("L")
        pixels = list(gray.getdata())
        width, height = gray.size

        if width < 3 or height < 3:
            return 0.5, {"sharpness_value": 0}

        edge_values = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                idx = y * width + x
                lap = (
                    pixels[idx - width] + pixels[idx + width]
                    + pixels[idx - 1] + pixels[idx + 1]
                    - 4 * pixels[idx]
                )
                edge_values.append(abs(lap))

        if not edge_values or len(edge_values) < 2:
            return 0.5, {"sharpness_value": 0}

        sharpness = statistics.variance(edge_values)
        score = min(1.0, sharpness / self.min_sharpness)
        return score, {"sharpness_value": sharpness, "min_required": self.min_sharpness}

    def _check_noise(self, image: Image.Image) -> tuple[float, dict]:
        gray = image.convert("L")
        pixels = list(gray.getdata())
        width, height = gray.size

        block_size = 8
        local_variances = []

        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = []
                for by in range(block_size):
                    for bx in range(block_size):
                        idx = (y + by) * width + (x + bx)
                        if idx < len(pixels):
                            block.append(pixels[idx])
                if len(block) > 1:
                    local_variances.append(statistics.variance(block))

        if not local_variances:
            return 1.0, {"noise_estimate": 0}

        noise_estimate = statistics.median(local_variances)
        score = max(0.0, 1.0 - noise_estimate / self.max_noise)
        return score, {"noise_estimate": noise_estimate, "max_allowed": self.max_noise}

    def _check_contrast(self, image: Image.Image) -> tuple[float, dict]:
        gray = image.convert("L")
        min_val, max_val = gray.getextrema()
        contrast = max_val - min_val
        score = min(1.0, contrast / self.min_contrast)
        return score, {"min_value": min_val, "max_value": max_val, "contrast": contrast}

    def _check_brightness(self, image: Image.Image) -> tuple[float, dict]:
        gray = image.convert("L")
        pixels = list(gray.getdata())
        avg_brightness = sum(pixels) / len(pixels) if pixels else 128

        if 100 <= avg_brightness <= 160:
            score = 1.0
        elif avg_brightness < 100:
            score = avg_brightness / 100
        else:
            score = max(0.0, 1.0 - (avg_brightness - 160) / 95)

        return score, {"average_brightness": avg_brightness}
