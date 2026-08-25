"""VLM (Vision Language Model) grader."""

from typing import Any

from PIL import Image

from compass.graders.base import GradeContext, GradeResult, GraderScope, GraderType, ModelGrader
from compass.graders.domains.image._encoding import image_to_base64
from compass.graders.model.prompt_template import RubricPromptTemplate
from compass.graders.registry import register_grader


@register_grader("vlm_judge")
class VLMJudgeGrader(ModelGrader):
    """Model-based grader using VLM for complex evaluation.

    Scope: OUTCOME - evaluates the final image against criteria.
    """

    name = "vlm_judge"
    grader_type = GraderType.MODEL
    grader_scope = GraderScope.OUTCOME

    DEFAULT_SYSTEM_PROMPT = """\
You are an image quality evaluator. Analyze the given image against the provided criteria.

For each criterion, respond with:
- PASS if the criterion is clearly met
- FAIL if the criterion is not met
- PARTIAL if partially met

Provide brief reasoning for each judgment.

Finally, give an overall score from 0.0 to 1.0 based on how well the image meets all criteria."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.model = self.config.get("model", "gpt-4o")
        self.criteria = self.config.get("criteria", [])
        self.api_key = self.config.get("api_key", "")
        self.api_base = self.config.get("api_base", "https://api.openai.com/v1")
        self.threshold = self.config.get("threshold", 0.7)
        self.system_prompt = self.config.get("system_prompt", self.DEFAULT_SYSTEM_PROMPT)

        # Structured prompt template (optional)
        self._template = RubricPromptTemplate.from_config(self.config)

    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade outcome image using VLM evaluation."""
        image = context.image

        if image is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=None,  # not measured
                error="No image in outcome",
            )

        criteria = self.criteria or self._extract_criteria(context)

        if not criteria:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=None,  # not measured
                error="No evaluation criteria provided",
            )

        try:
            result = await self._call_vlm(image, context.prompt, criteria)

            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=result["score"] >= self.threshold,
                score=result["score"],
                details={
                    "model": self.model,
                    "criteria": criteria,
                    "criterion_results": result["criterion_results"],
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

    def _extract_criteria(self, context: GradeContext) -> list[str]:
        """Extract evaluation criteria from context."""
        return [
            f"The image matches the description: {context.prompt}",
            "The image has good visual quality",
            "The image does not contain artifacts or distortions",
        ]

    async def _call_vlm(
        self,
        image: Image.Image,
        prompt: str,
        criteria: list[str],
    ) -> dict[str, Any]:
        """Call VLM API for evaluation."""
        import json
        import os

        # Check for API key
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Allow placeholder mode for testing (NOT recommended for production)
            if self.config.get("allow_placeholder"):
                return {
                    "score": 0.5,
                    "criterion_results": {c: "UNKNOWN" for c in criteria},
                    "reasoning": (
                        "PLACEHOLDER MODE - no API key configured. "
                        "Results are not meaningful."
                    ),
                }
            raise NotImplementedError(
                "VLM API key not configured. Set OPENAI_API_KEY environment variable "
                "or pass api_key in config. Alternatively, set config.allow_placeholder=true "
                "for testing (NOT recommended)."
            )

        image_base64 = image_to_base64(image)

        criteria_text = "\n".join(f"- {c}" for c in criteria)

        # Use structured template when available, otherwise legacy format.
        # User-provided system_prompt config always takes priority.
        use_template = (
            self._template.has_structured_sections
            and self.system_prompt == self.DEFAULT_SYSTEM_PROMPT
        )
        if use_template:
            system_prompt = self._template.render_system_prompt()
            user_prompt = self._template.render_user_prompt(
                criteria_text=criteria_text,
                prompt=prompt or "",
            )
        else:
            system_prompt = self.system_prompt
            user_prompt = f"""Original prompt: {prompt}

Evaluate this image against the following criteria:
{criteria_text}

For each criterion, state PASS, PARTIAL, or FAIL with a brief explanation.
Then provide an overall score from 0.0 to 1.0.

Respond in JSON format:
{{"score": <float>, "criterion_results": {{"<criterion>": "PASS|PARTIAL|FAIL", ...}}, \
"reasoning": "<explanation>"}}"""

        try:
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
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{image_base64}"
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
            # Try to parse JSON from response
            try:
                # Handle markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                parsed = json.loads(content.strip())
                return {
                    "score": float(parsed.get("score", 0.5)),
                    "criterion_results": parsed.get("criterion_results", {}),
                    "reasoning": parsed.get("reasoning", content),
                }
            except (json.JSONDecodeError, KeyError, IndexError):
                # Fallback: try to extract score from text
                return {
                    "score": 0.5,
                    "criterion_results": {},
                    "reasoning": content,
                }

        except ImportError as exc:
            raise NotImplementedError(
                "httpx is required for VLM API calls. Install with: pip install httpx"
            ) from exc

