"""Base evaluator class and context."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from compass.core.result import EvaluatorResult


@dataclass
class EvalContext:
    """Context for evaluation containing input information."""

    prompt: str
    negative_prompt: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    reference_image: Image.Image | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Evaluator(ABC):
    """Base class for all evaluators."""

    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize evaluator with config.

        Args:
            config: Evaluator-specific configuration.
        """
        self.config = config or {}

    @abstractmethod
    async def evaluate(self, image: Image.Image, context: EvalContext) -> EvaluatorResult:
        """Evaluate an image.

        Args:
            image: The image to evaluate.
            context: Evaluation context with input information.

        Returns:
            EvaluatorResult with score and metadata.
        """
        pass

    def validate_config(self) -> bool:
        """Validate evaluator configuration.

        Returns:
            True if config is valid.
        """
        return True


class EvaluatorProtocol(Protocol):
    """Protocol for evaluator implementations."""

    name: str

    def __init__(self, config: dict[str, Any] | None = None) -> None: ...

    async def evaluate(self, image: Image.Image, context: EvalContext) -> EvaluatorResult: ...
