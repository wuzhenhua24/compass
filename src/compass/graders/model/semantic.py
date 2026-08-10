"""Semantic matching grader using CLIP."""

from typing import Any

from PIL import Image

from compass.graders.base import ModelGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.registry import register_grader


@register_grader("semantic_match")
class SemanticMatchGrader(ModelGrader):
    """Model-based semantic matching using CLIP.

    Scope: OUTCOME - evaluates image-text alignment of the final result.
    """

    name = "semantic_match"
    grader_type = GraderType.MODEL
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.model_name = self.config.get("model", "openai/clip-vit-base-patch32")
        self.threshold = self.config.get("threshold", 0.25)
        self._model = None
        self._processor = None

    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade semantic match between outcome image and prompt."""
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

        try:
            score = await self._compute_clip_score(image, context.prompt)
            passed = score >= self.threshold

            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=passed,
                score=score,
                details={
                    "model": self.model_name,
                    "threshold": self.threshold,
                    "prompt": context.prompt,
                    "similarity_score": score,
                },
                failure_tags=["low_similarity"] if not passed else [],
                reasoning=f"CLIP similarity: {score:.3f} (threshold: {self.threshold})",
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

    async def _compute_clip_score(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity score."""
        if self._model is None:
            await self._load_model()

        if self._model is None:
            raise NotImplementedError(
                "CLIP model not available. Install with: pip install open-clip-torch "
                "or set config.allow_placeholder=true to use placeholder scoring (NOT recommended)."
            )

        import torch

        # Preprocess image and text
        image_input = self._preprocess(image).unsqueeze(0)
        text_input = self._tokenizer([text])

        with torch.no_grad():
            image_features = self._model.encode_image(image_input)
            text_features = self._model.encode_text(text_input)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            similarity = (image_features @ text_features.T).item()

        return float(similarity)

    async def _load_model(self) -> None:
        """Load CLIP model lazily."""
        # Allow placeholder mode for testing (NOT recommended for production)
        if self.config.get("allow_placeholder"):
            return

        try:
            import open_clip

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self._model.eval()
        except ImportError:
            # Model loading failed, will raise in _compute_clip_score
            pass
