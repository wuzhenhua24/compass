"""Edit Preservation grader — evaluates non-target region fidelity.

Part of the "editing triangle" evaluation model:
  Transformation Correctness × Locality × **Preservation**

Measures how well the areas *outside* the edit region are preserved
after an image editing operation.
"""

from __future__ import annotations

from typing import Any

from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.code.image.edit_diff import (
    compute_change_mask,
    compute_change_ratio,
    compute_histogram_similarity,
    compute_mse_in_region,
    ensure_comparable,
    invert_mask,
    load_edit_mask,
    mse_to_similarity,
)
from compass.graders.registry import register_grader


@register_grader("edit_preservation")
class EditPreservationGrader(CodeGrader):
    """Evaluate how well non-target regions are preserved after editing.

    Scope: OUTCOME
    Requires: ``context.reference_image`` (original) and ``context.image`` (edited).
    Optional: ``context.metadata["edit_mask"]`` or ``["edit_region"]``.

    Algorithm:
      1. Load or auto-detect edit mask → invert to get preservation mask.
      2. Compute MSE in preservation region → pixel similarity.
      3. Compute histogram similarity in preservation region.
      4. Combined score = pixel_weight * pixel_sim + histogram_weight * hist_sim.
    """

    name = "edit_preservation"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.threshold: float = self.config.get("threshold", 0.85)
        self.pixel_threshold: int = self.config.get("pixel_threshold", 10)
        self.pixel_weight: float = self.config.get("pixel_weight", 0.7)
        self.histogram_weight: float = self.config.get("histogram_weight", 0.3)

    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade preservation of non-target regions."""
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

        # --- determine preservation mask ---
        edit_mask, mask_source = load_edit_mask(context.metadata, original.size)

        if edit_mask is None:
            # Auto-detect: invert the change mask
            change = compute_change_mask(original, edited, self.pixel_threshold)
            preservation_mask = invert_mask(change)
            mask_source = "auto_detected"
        else:
            preservation_mask = invert_mask(edit_mask)

        # --- compute metrics ---
        mse = compute_mse_in_region(original, edited, preservation_mask)
        pixel_sim = mse_to_similarity(mse)

        hist_sim = compute_histogram_similarity(original, edited, preservation_mask)

        change_mask = compute_change_mask(original, edited, self.pixel_threshold)
        change_ratio = compute_change_ratio(change_mask)

        score = self.pixel_weight * pixel_sim + self.histogram_weight * hist_sim

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=score >= self.threshold,
            score=round(score, 4),
            details={
                "pixel_similarity": round(pixel_sim, 4),
                "histogram_similarity": round(hist_sim, 4),
                "preservation_score": round(score, 4),
                "change_ratio": round(change_ratio, 4),
                "mask_source": mask_source,
            },
        )
