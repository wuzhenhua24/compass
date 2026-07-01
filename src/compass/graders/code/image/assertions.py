"""Image assertion grader for deterministic checks."""

from typing import Any

from PIL import Image

from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.registry import register_grader


@register_grader("image_assertions")
class ImageAssertionGrader(CodeGrader):
    """Deterministic image assertions grader.

    Scope: OUTCOME - only needs the final image.
    """

    name = "image_assertions"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.assert_size = self.config.get("assert_size")
        self.assert_min_size = self.config.get("assert_min_size")
        self.assert_max_size = self.config.get("assert_max_size")
        self.assert_format = self.config.get("assert_format")
        self.assert_mode = self.config.get("assert_mode")
        self.assert_not_blank = self.config.get("assert_not_blank", False)
        self.assert_aspect_ratio = self.config.get("assert_aspect_ratio")

    async def grade(self, context: GradeContext) -> GradeResult:
        """Run deterministic assertions on the outcome image."""
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

        checks = []
        failures = []

        if self.assert_size:
            expected_w, expected_h = self.assert_size
            actual_w, actual_h = image.size
            passed = actual_w == expected_w and actual_h == expected_h
            checks.append(("size", passed))
            if not passed:
                failures.append(
                    f"Size mismatch: expected {expected_w}x{expected_h}, got {actual_w}x{actual_h}"
                )

        if self.assert_min_size:
            min_w, min_h = self.assert_min_size
            actual_w, actual_h = image.size
            passed = actual_w >= min_w and actual_h >= min_h
            checks.append(("min_size", passed))
            if not passed:
                failures.append(
                    f"Below min size: expected >={min_w}x{min_h}, got {actual_w}x{actual_h}"
                )

        if self.assert_max_size:
            max_w, max_h = self.assert_max_size
            actual_w, actual_h = image.size
            passed = actual_w <= max_w and actual_h <= max_h
            checks.append(("max_size", passed))
            if not passed:
                failures.append(
                    f"Exceeds max size: expected <={max_w}x{max_h}, got {actual_w}x{actual_h}"
                )

        if self.assert_format:
            formats = (
                self.assert_format
                if isinstance(self.assert_format, list)
                else [self.assert_format]
            )
            formats = [f.upper() for f in formats]
            img_format = (image.format or "").upper()
            passed = img_format in formats
            checks.append(("format", passed))
            if not passed:
                failures.append(f"Format mismatch: expected {formats}, got '{img_format}'")

        if self.assert_mode:
            modes = (
                self.assert_mode
                if isinstance(self.assert_mode, list)
                else [self.assert_mode]
            )
            passed = image.mode in modes
            checks.append(("mode", passed))
            if not passed:
                failures.append(f"Mode mismatch: expected {modes}, got '{image.mode}'")

        if self.assert_not_blank:
            passed = self._check_not_blank(image)
            checks.append(("not_blank", passed))
            if not passed:
                failures.append("Image appears to be blank or near-blank")

        if self.assert_aspect_ratio:
            actual_ratio = image.width / image.height
            tolerance = self.config.get("aspect_ratio_tolerance", 0.01)
            passed = abs(actual_ratio - self.assert_aspect_ratio) <= tolerance
            checks.append(("aspect_ratio", passed))
            if not passed:
                failures.append(
                    f"Aspect ratio mismatch: expected {self.assert_aspect_ratio:.3f}, "
                    f"got {actual_ratio:.3f}"
                )

        if not checks:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=True,
                score=1.0,
                details={"message": "No assertions configured"},
            )

        passed_count = sum(1 for _, p in checks if p)
        total_count = len(checks)
        all_passed = passed_count == total_count
        score = passed_count / total_count

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=all_passed,
            score=score,
            details={
                "checks": {name: passed for name, passed in checks},
                "passed_count": passed_count,
                "total_count": total_count,
                "failures": failures,
                "image_info": {
                    "size": list(image.size),
                    "format": image.format,
                    "mode": image.mode,
                },
            },
        )

    def _check_not_blank(self, image: Image.Image, threshold: float = 0.01) -> bool:
        """Check if image is not blank."""
        import statistics

        gray = image.convert("L")
        pixels = list(gray.getdata())

        if len(set(pixels)) == 1:
            return False

        if len(pixels) > 1:
            try:
                variance = statistics.variance(pixels)
                return variance >= threshold
            except statistics.StatisticsError:
                pass

        return True
