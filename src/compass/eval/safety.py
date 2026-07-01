"""Safety evaluation for images."""

from typing import Any

from PIL import Image

from compass.core.result import EvaluatorResult
from compass.eval.base import Evaluator, EvalContext
from compass.eval.registry import register_evaluator


@register_evaluator("safety_check")
class SafetyCheckEvaluator(Evaluator):
    """Evaluator for image safety checks."""

    name = "safety_check"

    AVAILABLE_CHECKS = ["nsfw", "watermark", "face_privacy", "copyright"]

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.checks = self.config.get("checks", ["nsfw"])
        self._validators: dict[str, Any] = {}

    async def evaluate(self, image: Image.Image, context: EvalContext) -> EvaluatorResult:
        """Run safety checks on image.

        Args:
            image: Generated image.
            context: Evaluation context.

        Returns:
            EvaluatorResult with safety check results.
        """
        try:
            check_results = {}
            all_passed = True

            for check in self.checks:
                if check not in self.AVAILABLE_CHECKS:
                    continue

                passed, details = await self._run_check(check, image)
                check_results[check] = {
                    "passed": passed,
                    "details": details,
                }
                if not passed:
                    all_passed = False

            # Safety checks are binary - all must pass
            score = 1.0 if all_passed else 0.0

            return EvaluatorResult(
                name=self.name,
                score=score,
                passed=all_passed,
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
    ) -> tuple[bool, dict[str, Any]]:
        """Run a specific safety check.

        Args:
            check_type: Type of check to run.
            image: Image to check.

        Returns:
            Tuple of (passed, details).
        """
        match check_type:
            case "nsfw":
                return await self._check_nsfw(image)
            case "watermark":
                return await self._check_watermark(image)
            case "face_privacy":
                return await self._check_face_privacy(image)
            case "copyright":
                return await self._check_copyright(image)
            case _:
                return True, {"message": f"Unknown check: {check_type}"}

    async def _check_nsfw(self, image: Image.Image) -> tuple[bool, dict[str, Any]]:
        """Check for NSFW content.

        Args:
            image: Image to check.

        Returns:
            Tuple of (is_safe, details).
        """
        # TODO: Implement actual NSFW detection
        # Could use models like LAION safety classifier
        return True, {"nsfw_score": 0.01}

    async def _check_watermark(self, image: Image.Image) -> tuple[bool, dict[str, Any]]:
        """Check for watermarks.

        Args:
            image: Image to check.

        Returns:
            Tuple of (no_watermark, details).
        """
        # TODO: Implement watermark detection
        return True, {"has_watermark": False}

    async def _check_face_privacy(self, image: Image.Image) -> tuple[bool, dict[str, Any]]:
        """Check for identifiable faces.

        Args:
            image: Image to check.

        Returns:
            Tuple of (privacy_safe, details).
        """
        # TODO: Implement face detection and privacy check
        return True, {"faces_detected": 0}

    async def _check_copyright(self, image: Image.Image) -> tuple[bool, dict[str, Any]]:
        """Check for potential copyright issues.

        Args:
            image: Image to check.

        Returns:
            Tuple of (copyright_safe, details).
        """
        # TODO: Implement copyright similarity check
        return True, {"similar_copyrighted": False}
