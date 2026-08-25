"""Tests for the image editing three-dimensional evaluation graders.

Covers:
  - edit_diff utility functions
  - EditPreservationGrader (code grader)
  - EditLocalityGrader (code grader)
  - EditCorrectnessGrader (model grader — config + error paths only)
"""

from __future__ import annotations

import pytest
from PIL import Image

from compass.core.transcript import Outcome
from compass.graders.base import GradeContext, GraderScope, GraderType
from compass.graders.domains.image.edit_correctness import EditCorrectnessGrader
from compass.graders.domains.image.edit_diff import (
    compute_change_mask,
    compute_change_ratio,
    compute_diff_image,
    compute_histogram_similarity,
    compute_mse_in_region,
    ensure_comparable,
    invert_mask,
    load_edit_mask,
    mse_to_similarity,
)
from compass.graders.domains.image.edit_locality import EditLocalityGrader
from compass.graders.domains.image.edit_preservation import EditPreservationGrader

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def original_image() -> Image.Image:
    """100x100 solid red image."""
    return Image.new("RGB", (100, 100), (255, 0, 0))


@pytest.fixture
def edited_local(original_image: Image.Image) -> Image.Image:
    """Local edit: 20x20 blue square at (40,40)-(60,60)."""
    img = original_image.copy()
    for x in range(40, 60):
        for y in range(40, 60):
            img.putpixel((x, y), (0, 0, 255))
    return img


@pytest.fixture
def edited_global() -> Image.Image:
    """Global edit: entire image changed to green."""
    return Image.new("RGB", (100, 100), (0, 255, 0))


@pytest.fixture
def edit_mask() -> Image.Image:
    """White mask covering (40,40)-(60,60), black elsewhere."""
    mask = Image.new("L", (100, 100), 0)
    for x in range(40, 60):
        for y in range(40, 60):
            mask.putpixel((x, y), 255)
    return mask


def _make_context(
    *,
    original: Image.Image | None = None,
    edited: Image.Image | None = None,
    prompt: str = "test edit instruction",
    metadata: dict | None = None,
) -> GradeContext:
    """Build a GradeContext for edit grader tests."""
    outcome = Outcome(image=edited) if edited is not None else None
    return GradeContext(
        prompt=prompt,
        reference_image=original,
        outcome=outcome,
        metadata=metadata or {},
    )


# =====================================================================
# edit_diff utility tests
# =====================================================================


class TestEnsureComparable:
    def test_same_images_unchanged(self, original_image: Image.Image):
        a, b = ensure_comparable(original_image, original_image.copy())
        assert a.size == b.size
        assert a.mode == "RGB"

    def test_different_sizes_resized(self, original_image: Image.Image):
        big = Image.new("RGB", (200, 200), (0, 0, 0))
        a, b = ensure_comparable(original_image, big)
        assert b.size == a.size == (100, 100)

    def test_rgba_converted_to_rgb(self):
        rgba = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
        rgb = Image.new("RGB", (50, 50), (0, 255, 0))
        a, b = ensure_comparable(rgba, rgb)
        assert a.mode == "RGB"
        assert b.mode == "RGB"


class TestComputeDiffImage:
    def test_identical_images_all_black(self, original_image: Image.Image):
        diff = compute_diff_image(original_image, original_image.copy())
        # All pixels should be (0, 0, 0)
        extrema = diff.getextrema()
        assert all(mx == 0 for _, mx in extrema)

    def test_different_images_nonzero(
        self, original_image: Image.Image, edited_global: Image.Image
    ):
        diff = compute_diff_image(original_image, edited_global)
        extrema = diff.getextrema()
        assert any(mx > 0 for _, mx in extrema)


class TestComputeChangeMask:
    def test_local_edit_mask_shape(self, original_image: Image.Image, edited_local: Image.Image):
        mask = compute_change_mask(original_image, edited_local, pixel_threshold=10)
        assert mask.mode == "L"
        assert mask.size == (100, 100)
        # Center should be white
        assert mask.getpixel((50, 50)) == 255
        # Corner should be black
        assert mask.getpixel((0, 0)) == 0

    def test_identical_images_no_change(self, original_image: Image.Image):
        mask = compute_change_mask(original_image, original_image.copy(), pixel_threshold=10)
        ratio = compute_change_ratio(mask)
        assert ratio == 0.0


class TestComputeChangeRatio:
    def test_local_edit_ratio(self, original_image: Image.Image, edited_local: Image.Image):
        mask = compute_change_mask(original_image, edited_local, pixel_threshold=10)
        ratio = compute_change_ratio(mask)
        # 20x20 / 100x100 = 0.04
        assert abs(ratio - 0.04) < 0.01

    def test_global_edit_ratio(self, original_image: Image.Image, edited_global: Image.Image):
        mask = compute_change_mask(original_image, edited_global, pixel_threshold=10)
        ratio = compute_change_ratio(mask)
        assert ratio > 0.99


class TestMSE:
    def test_identical_mse_zero(self, original_image: Image.Image):
        full_mask = Image.new("L", (100, 100), 255)
        mse = compute_mse_in_region(original_image, original_image.copy(), full_mask)
        assert mse == 0.0

    def test_mse_to_similarity_extremes(self):
        assert mse_to_similarity(0.0) == 1.0
        assert mse_to_similarity(255.0 * 255.0) == 0.0

    def test_mse_to_similarity_midpoint(self):
        mid = 255.0 * 255.0 / 2.0
        sim = mse_to_similarity(mid)
        assert abs(sim - 0.5) < 0.01


class TestLoadEditMask:
    def test_from_bbox(self):
        mask, source = load_edit_mask(
            {"edit_region": [10, 10, 30, 30]},
            (100, 100),
        )
        assert mask is not None
        assert source == "bbox"
        assert mask.mode == "L"
        assert mask.getpixel((20, 20)) == 255
        assert mask.getpixel((0, 0)) == 0

    def test_from_pil_image(self, edit_mask: Image.Image):
        mask, source = load_edit_mask(
            {"edit_mask": edit_mask},
            (100, 100),
        )
        assert mask is not None
        assert source == "provided"

    def test_no_mask_returns_none(self):
        mask, source = load_edit_mask({}, (100, 100))
        assert mask is None
        assert source == "none"


class TestHistogramSimilarity:
    def test_identical_images(self, original_image: Image.Image):
        sim = compute_histogram_similarity(original_image, original_image.copy())
        assert sim > 0.99

    def test_different_images_lower(self, original_image: Image.Image, edited_global: Image.Image):
        sim = compute_histogram_similarity(original_image, edited_global)
        assert sim < 0.99


class TestInvertMask:
    def test_invert(self, edit_mask: Image.Image):
        inv = invert_mask(edit_mask)
        # Where original was white (inside edit), inverted should be black
        assert inv.getpixel((50, 50)) == 0
        # Where original was black (outside edit), inverted should be white
        assert inv.getpixel((0, 0)) == 255


# =====================================================================
# EditPreservationGrader tests
# =====================================================================


class TestEditPreservationGrader:
    def test_scope_and_type(self):
        grader = EditPreservationGrader()
        assert grader.grader_scope == GraderScope.OUTCOME
        assert grader.grader_type == GraderType.CODE
        assert grader.name == "edit_preservation"

    async def test_identical_images_perfect_score(self, original_image: Image.Image):
        ctx = _make_context(original=original_image, edited=original_image.copy())
        grader = EditPreservationGrader()
        result = await grader.grade(ctx)

        assert result.passed is True
        assert result.score >= 0.99

    async def test_local_edit_with_mask_high_score(
        self,
        original_image: Image.Image,
        edited_local: Image.Image,
        edit_mask: Image.Image,
    ):
        ctx = _make_context(
            original=original_image,
            edited=edited_local,
            metadata={"edit_mask": edit_mask},
        )
        grader = EditPreservationGrader()
        result = await grader.grade(ctx)

        # Preservation region (outside mask) is untouched → high score
        assert result.passed is True
        assert result.score >= 0.9
        assert result.details["mask_source"] == "provided"

    async def test_local_edit_auto_detect(
        self, original_image: Image.Image, edited_local: Image.Image
    ):
        ctx = _make_context(original=original_image, edited=edited_local)
        grader = EditPreservationGrader()
        result = await grader.grade(ctx)

        # Auto-detected: preservation region is everything unchanged → high score
        assert result.passed is True
        assert result.score >= 0.9
        assert result.details["mask_source"] == "auto_detected"

    async def test_global_edit_low_score(
        self, original_image: Image.Image, edited_global: Image.Image
    ):
        ctx = _make_context(
            original=original_image,
            edited=edited_global,
            metadata={"edit_region": [40, 40, 60, 60]},
        )
        grader = EditPreservationGrader()
        result = await grader.grade(ctx)

        # Everything outside bbox changed → low preservation
        assert result.passed is False
        assert result.score < 0.85

    async def test_no_reference_error(self, edited_local: Image.Image):
        ctx = _make_context(original=None, edited=edited_local)
        grader = EditPreservationGrader()
        result = await grader.grade(ctx)

        assert result.error is not None
        assert "reference" in result.error.lower()

    async def test_no_edited_image_error(self, original_image: Image.Image):
        ctx = _make_context(original=original_image, edited=None)
        grader = EditPreservationGrader()
        result = await grader.grade(ctx)

        assert result.error is not None
        assert "edited" in result.error.lower() or "image" in result.error.lower()


# =====================================================================
# EditLocalityGrader tests
# =====================================================================


class TestEditLocalityGrader:
    def test_scope_and_type(self):
        grader = EditLocalityGrader()
        assert grader.grader_scope == GraderScope.OUTCOME
        assert grader.grader_type == GraderType.CODE
        assert grader.name == "edit_locality"

    async def test_local_edit_with_mask_high_adherence(
        self,
        original_image: Image.Image,
        edited_local: Image.Image,
        edit_mask: Image.Image,
    ):
        ctx = _make_context(
            original=original_image,
            edited=edited_local,
            metadata={"edit_mask": edit_mask},
        )
        grader = EditLocalityGrader()
        result = await grader.grade(ctx)

        # All changes are inside the mask
        assert result.passed is True
        assert result.score >= 0.99
        assert result.details["mask_adherence"] >= 0.99
        assert result.details["has_edit_mask"] is True

    async def test_local_edit_no_mask_high_score(
        self, original_image: Image.Image, edited_local: Image.Image
    ):
        ctx = _make_context(original=original_image, edited=edited_local)
        grader = EditLocalityGrader()
        result = await grader.grade(ctx)

        # Small change area → high locality
        assert result.passed is True
        assert result.score >= 0.8
        assert result.details["has_edit_mask"] is False

    async def test_global_edit_with_mask_low_adherence(
        self,
        original_image: Image.Image,
        edited_global: Image.Image,
        edit_mask: Image.Image,
    ):
        ctx = _make_context(
            original=original_image,
            edited=edited_global,
            metadata={"edit_mask": edit_mask},
        )
        grader = EditLocalityGrader()
        result = await grader.grade(ctx)

        # Changes everywhere but mask covers only 4% → low adherence
        assert result.passed is False
        assert result.score < 0.1
        assert result.details["mask_adherence"] < 0.1

    async def test_global_edit_no_mask_low_score(
        self, original_image: Image.Image, edited_global: Image.Image
    ):
        ctx = _make_context(original=original_image, edited=edited_global)
        grader = EditLocalityGrader()
        result = await grader.grade(ctx)

        # Almost all pixels changed → low locality
        assert result.passed is False
        assert result.score < 0.1

    async def test_no_reference_error(self, edited_local: Image.Image):
        ctx = _make_context(original=None, edited=edited_local)
        grader = EditLocalityGrader()
        result = await grader.grade(ctx)

        assert result.error is not None
        assert "reference" in result.error.lower()

    async def test_identical_images_perfect_score(self, original_image: Image.Image):
        ctx = _make_context(original=original_image, edited=original_image.copy())
        grader = EditLocalityGrader()
        result = await grader.grade(ctx)

        assert result.passed is True
        assert result.score == 1.0


# =====================================================================
# EditCorrectnessGrader tests
# =====================================================================


class TestEditCorrectnessGrader:
    def test_init_default_config(self):
        grader = EditCorrectnessGrader()
        assert grader.name == "edit_correctness"
        assert grader.grader_scope == GraderScope.OUTCOME
        assert grader.grader_type == GraderType.MODEL
        assert grader.model == "gpt-4o"
        assert grader.threshold == 0.7
        assert grader.allow_placeholder is False

    def test_init_custom_config(self):
        grader = EditCorrectnessGrader({
            "model": "gpt-4-turbo",
            "threshold": 0.9,
            "criteria": ["text changed correctly"],
            "allow_placeholder": True,
        })
        assert grader.model == "gpt-4-turbo"
        assert grader.threshold == 0.9
        assert grader.criteria == ["text changed correctly"]
        assert grader.allow_placeholder is True

    async def test_no_reference_image_error(self, edited_local: Image.Image):
        grader = EditCorrectnessGrader()
        ctx = _make_context(original=None, edited=edited_local)
        result = await grader.grade(ctx)

        assert result.error is not None
        assert "reference" in result.error.lower()
        assert result.passed is False

    async def test_no_edited_image_error(self, original_image: Image.Image):
        grader = EditCorrectnessGrader()
        ctx = _make_context(original=original_image, edited=None)
        result = await grader.grade(ctx)

        assert result.error is not None
        assert result.passed is False

    async def test_placeholder_mode(
        self, original_image: Image.Image, edited_local: Image.Image
    ):
        grader = EditCorrectnessGrader({"allow_placeholder": True})
        ctx = _make_context(original=original_image, edited=edited_local)
        result = await grader.grade(ctx)

        assert result.score == 0.5
        assert "PLACEHOLDER" in result.reasoning
        assert result.details.get("issues") == ["placeholder_mode"]


# =====================================================================
# Registry integration
# =====================================================================


class TestRegistry:
    def test_graders_registered(self):
        from compass.graders.registry import get_grader

        assert get_grader("edit_preservation") is EditPreservationGrader
        assert get_grader("edit_locality") is EditLocalityGrader
        assert get_grader("edit_correctness") is EditCorrectnessGrader
