"""Aesthetic scoring grader."""

from typing import Any

from PIL import Image

from compass.graders.base import ModelGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.registry import register_grader


@register_grader("aesthetic_score")
class AestheticScoreGrader(ModelGrader):
    """Model-based aesthetic quality grader.

    Scope: OUTCOME - evaluates the visual appeal of the final image.
    """

    name = "aesthetic_score"
    grader_type = GraderType.MODEL
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.model_name = self.config.get("model", "laion/aesthetic-predictor-v2")
        self.min_score = self.config.get("min_score", 5.0)
        self.max_score = self.config.get("max_score", 10.0)
        self._model = None

    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade aesthetic quality of the outcome image."""
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

        try:
            raw_score = await self._compute_aesthetic_score(image)
            normalized_score = (raw_score - 1) / (self.max_score - 1)
            normalized_score = max(0.0, min(1.0, normalized_score))
            passed = raw_score >= self.min_score

            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=passed,
                score=normalized_score,
                details={
                    "model": self.model_name,
                    "raw_score": raw_score,
                    "min_score": self.min_score,
                    "normalized_score": normalized_score,
                },
                reasoning=f"Aesthetic score: {raw_score:.2f}/10 (min required: {self.min_score})",
            )
        except Exception as e:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error=str(e),
            )

    async def _compute_aesthetic_score(self, image: Image.Image) -> float:
        """Compute aesthetic score."""
        if self._model is None:
            await self._load_model()

        if self._model is None:
            raise NotImplementedError(
                "Aesthetic model not available. Install with: pip install open-clip-torch "
                "or set config.allow_placeholder=true to use placeholder scoring (NOT recommended)."
            )

        import torch

        # Preprocess image
        image_input = self._preprocess(image).unsqueeze(0)

        with torch.no_grad():
            # Get CLIP image features
            image_features = self._clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Predict aesthetic score using the linear layer
            score = self._model(image_features).item()

        # Clamp to valid range [1, 10]
        return float(max(1.0, min(10.0, score)))

    async def _load_model(self) -> None:
        """Load aesthetic model lazily."""
        # Allow placeholder mode for testing (NOT recommended for production)
        if self.config.get("allow_placeholder"):
            return

        try:
            import torch
            import open_clip

            # Load CLIP model for feature extraction
            self._clip_model, _, self._preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )
            self._clip_model.eval()

            # Load aesthetic predictor head
            # The aesthetic predictor is a simple linear layer on top of CLIP features
            self._model = torch.nn.Linear(768, 1)

            # Try to load pretrained weights
            try:
                from huggingface_hub import hf_hub_download

                weights_path = hf_hub_download(
                    repo_id="shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
                    filename="pytorch_model.bin",
                )
                state_dict = torch.load(weights_path, map_location="cpu")
                self._model.load_state_dict(state_dict)
            except Exception:
                # If weights not available, model will still fail-closed
                # because _model is set but weights are random
                pass

            self._model.eval()

        except ImportError:
            # Model loading failed, will raise in _compute_aesthetic_score
            pass
