"""Edit Correctness grader — VLM evaluation of transformation accuracy.

Part of the "editing triangle" evaluation model:
  **Transformation Correctness** × Locality × Preservation

Uses a Vision Language Model to judge whether the requested edit
was correctly applied by comparing the original and edited images.
"""

from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from typing import Any

from PIL import Image

from compass.graders.base import GradeContext, GradeResult, GraderScope, GraderType, ModelGrader
from compass.graders.registry import register_grader


@register_grader("edit_correctness")
class EditCorrectnessGrader(ModelGrader):
    """VLM-based grader that evaluates whether an edit instruction was executed correctly.

    Scope: OUTCOME
    Requires: ``context.reference_image`` (original), ``context.image`` (edited),
              ``context.prompt`` (edit instruction).

    Key difference from VLMJudgeGrader:
      - Sends *two* images (before + after) to the VLM.
      - Focuses specifically on "was the edit instruction followed?".
      - Uses a structured prompt for before/after comparison.
    """

    name = "edit_correctness"
    grader_type = GraderType.MODEL
    grader_scope = GraderScope.OUTCOME

    DEFAULT_SYSTEM_PROMPT = (
        "You are an image editing evaluator. You will be shown two images: "
        "the ORIGINAL image (before editing) and the EDITED image (after editing), "
        "along with the edit instruction that was requested.\n\n"
        "Your task is to evaluate whether the edit instruction was correctly applied.\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        "{\n"
        '  "score": <float 0.0-1.0>,\n'
        '  "edit_applied": <bool>,\n'
        '  "edit_accurate": <bool>,\n'
        '  "reasoning": "<explanation>",\n'
        '  "issues": ["<issue1>", ...]\n'
        "}"
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.model: str = self.config.get("model", "gpt-4o")
        self.api_key: str = self.config.get("api_key", "")
        self.api_base: str = self.config.get("api_base", "https://api.openai.com/v1")
        self.threshold: float = self.config.get("threshold", 0.7)
        self.criteria: list[str] = self.config.get("criteria", [])
        self.allow_placeholder: bool = self.config.get("allow_placeholder", False)

    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade edit correctness using VLM evaluation."""
        # --- validate inputs ---
        if context.reference_image is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=None,  # not measured
                error="No reference image (original) provided",
            )

        edited = context.image
        if edited is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=None,  # not measured
                error="No edited image in outcome",
            )

        original = context.reference_image
        prompt = context.prompt or "No edit instruction provided"

        try:
            result = await self._call_vlm(original, edited, prompt)
            score = result["score"]
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=score >= self.threshold,
                score=score,
                details={
                    "model": self.model,
                    "edit_applied": result.get("edit_applied", False),
                    "edit_accurate": result.get("edit_accurate", False),
                    "issues": result.get("issues", []),
                },
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=None,  # not measured
                error=str(e),
            )

    async def _call_vlm(
        self,
        original: Image.Image,
        edited: Image.Image,
        prompt: str,
    ) -> dict[str, Any]:
        """Call VLM API with both original and edited images."""
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            if self.allow_placeholder:
                return {
                    "score": 0.5,
                    "edit_applied": False,
                    "edit_accurate": False,
                    "reasoning": (
                        "PLACEHOLDER MODE - no API key configured. "
                        "Results are not meaningful."
                    ),
                    "issues": ["placeholder_mode"],
                }
            raise NotImplementedError(
                "VLM API key not configured. Set OPENAI_API_KEY environment variable "
                "or pass api_key in config. Alternatively, set config.allow_placeholder=true "
                "for testing (NOT recommended)."
            )

        original_b64 = self._image_to_base64(original)
        edited_b64 = self._image_to_base64(edited)

        criteria_text = ""
        if self.criteria:
            criteria_text = "\n\nAdditional evaluation criteria:\n" + "\n".join(
                f"- {c}" for c in self.criteria
            )

        user_prompt = (
            f"Edit instruction: {prompt}\n\n"
            f"The first image is the ORIGINAL (before editing).\n"
            f"The second image is the EDITED result (after editing).\n\n"
            f"Evaluate whether the edit instruction was correctly applied to the original image."
            f"{criteria_text}"
        )

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{original_b64}",
                                    },
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{edited_b64}",
                                    },
                                },
                            ],
                        },
                    ],
                    "max_tokens": 1000,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()

        content = result["choices"][0]["message"]["content"]
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            parsed = json.loads(content.strip())
            return {
                "score": float(parsed.get("score", 0.5)),
                "edit_applied": bool(parsed.get("edit_applied", False)),
                "edit_accurate": bool(parsed.get("edit_accurate", False)),
                "reasoning": str(parsed.get("reasoning", content)),
                "issues": list(parsed.get("issues", [])),
            }
        except (json.JSONDecodeError, KeyError, IndexError):
            return {
                "score": 0.5,
                "edit_applied": False,
                "edit_accurate": False,
                "reasoning": content,
                "issues": ["json_parse_error"],
            }

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
