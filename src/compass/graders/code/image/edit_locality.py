"""Edit Locality grader — evaluates whether edits are confined to the target region.

Part of the "editing triangle" evaluation model:
  Transformation Correctness × **Locality** × Preservation

Measures how concentrated the changes are within the intended edit area.
"""

from __future__ import annotations

from typing import Any

from PIL import ImageStat

from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.code.image.edit_diff import (
    compute_change_mask,
    compute_change_ratio,
    ensure_comparable,
    load_edit_mask,
)
from compass.graders.registry import register_grader


@register_grader("edit_locality")
class EditLocalityGrader(CodeGrader):
    """Evaluate whether image edits are localised to the target region.

    Scope: OUTCOME
    Requires: ``context.reference_image`` (original) and ``context.image`` (edited).
    Optional: ``context.metadata["edit_mask"]`` or ``["edit_region"]``.

    Algorithm (with mask):
      1. Compute binary change mask (which pixels changed).
      2. ``mask_adherence = changes_inside_mask / total_changes``.
      3. ``score = mask_adherence``.

    Algorithm (without mask):
      1. Compute change_ratio (fraction of changed pixels).
      2. Smaller change_ratio → higher locality.
      3. ``score = max(0, 1 - change_ratio / max_change_ratio)``.
    """

    name = "edit_locality"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.threshold: float = self.config.get("threshold", 0.7)
        self.pixel_threshold: int = self.config.get("pixel_threshold", 10)
        self.max_change_ratio: float = self.config.get("max_change_ratio", 0.3)

    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade locality of image edits."""
        # --- validate inputs ---
        if context.reference_image is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error="No reference image (original) provided",
            )

        edited = context.image
        if edited is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error="No edited image in outcome",
            )

        original = context.reference_image
        original, edited = ensure_comparable(original, edited)

        # --- compute change mask ---
        change_mask = compute_change_mask(original, edited, self.pixel_threshold)
        change_ratio = compute_change_ratio(change_mask)

        # No changes at all → perfect locality
        total_changed = ImageStat.Stat(change_mask).sum[0] / 255.0
        if total_changed == 0:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=True,
                score=1.0,
                details={
                    "change_ratio": 0.0,
                    "has_edit_mask": False,
                    "mask_source": "none",
                },
            )

        # --- load edit mask ---
        edit_mask, mask_source = load_edit_mask(context.metadata, original.size)

        if edit_mask is not None:
            # With mask: measure adherence
            # Count changed pixels inside vs outside the mask
            mask_pixels = list(edit_mask.get_flattened_data())
            change_pixels = list(change_mask.get_flattened_data())

            inside = sum(
                1 for m, c in zip(mask_pixels, change_pixels, strict=True)
                if m > 127 and c > 127
            )
            outside = sum(
                1 for m, c in zip(mask_pixels, change_pixels, strict=True)
                if m <= 127 and c > 127
            )
            total = inside + outside

            mask_adherence = inside / total if total > 0 else 1.0
            leak_ratio = outside / total if total > 0 else 0.0
            score = mask_adherence

            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=score >= self.threshold,
                score=round(score, 4),
                details={
                    "change_ratio": round(change_ratio, 4),
                    "mask_adherence": round(mask_adherence, 4),
                    "leak_ratio": round(leak_ratio, 4),
                    "has_edit_mask": True,
                    "mask_source": mask_source,
                },
            )
        else:
            # Without mask: smaller change area → higher locality
            score = max(0.0, 1.0 - change_ratio / self.max_change_ratio)

            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=score >= self.threshold,
                score=round(score, 4),
                details={
                    "change_ratio": round(change_ratio, 4),
                    "has_edit_mask": False,
                    "mask_source": "none",
                },
            )
