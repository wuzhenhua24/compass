"""Base grader classes for the three-tier grading system.

Key design principle (from Anthropic "Demystifying Evals for AI Agents"):
  Graders should be able to independently access the execution transcript
  (how the agent worked) and the final outcome (what was produced).

  - Outcome graders:    evaluate the final result (image quality, semantic match)
  - Transcript graders: evaluate the execution trace (tool usage, cost, latency)
  - Combined graders:   evaluate both together (efficiency vs quality tradeoff)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Type
import uuid

from PIL import Image

from compass.core.artifacts import Artifact, ImageArtifact, CodeArtifact, TextArtifact
from compass.core.transcript import Outcome, Transcript


class GraderType(str, Enum):
    """Type of grader."""

    CODE = "code"
    MODEL = "model"
    HUMAN = "human"


class GraderScope(str, Enum):
    """What a grader evaluates.

    Declares which data the grader needs to perform its evaluation.
    The runner uses this to validate that required data is available
    before calling the grader.
    """

    OUTCOME = "outcome"        # Only needs final result (image, output_data)
    TRANSCRIPT = "transcript"  # Only needs execution trace (tool_calls, timing)
    BOTH = "both"              # Needs both transcript and outcome


@dataclass
class GradeContext:
    """Context for grading - provides independent access to Transcript and Outcome.

    This is the central data object passed to every grader. It cleanly separates:
    - Input:      what was requested (prompt, params)
    - Transcript: how the agent executed (tool calls, reasoning, timing)
    - Outcome:    what was produced (image, output data, blocked status)

    Graders declare their scope (outcome/transcript/both) and access
    the relevant parts of this context independently.
    """

    # === Input ===
    prompt: str = ""
    negative_prompt: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    # === Transcript: execution trace ===
    # The full execution record - tool calls, reasoning, timing, environment
    transcript: Transcript | None = None

    # === Outcome: final result ===
    # The end state - image, output data, blocked status
    outcome: Outcome | None = None

    # === Reference data (for comparison grading) ===
    reference_image: Image.Image | None = None
    reference_artifact: Artifact | None = None
    reference_images: dict[str, Image.Image] = field(default_factory=dict)

    # === Leak detection ===
    # Unique marker strings embedded in grader/solution files.
    # If any marker appears in the transcript, the trial is tainted.
    leak_markers: list[str] = field(default_factory=list)

    # === Extra metadata ===
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors (so graders don't need deep attribute chains)
    # ------------------------------------------------------------------

    @property
    def image(self) -> Image.Image | None:
        """Shortcut: the generated image from the outcome.

        Checks the legacy ``outcome.image`` field first, then falls back to
        looking for an ``ImageArtifact`` in the artifacts list.
        """
        if self.outcome is not None:
            if self.outcome.image is not None:
                return self.outcome.image
            img_art = self.outcome.get_artifact(ImageArtifact)
            if img_art is not None:
                return img_art.image
        return None

    @property
    def is_blocked(self) -> bool:
        """Shortcut: whether the agent was blocked (e.g., safety filter)."""
        if self.outcome is not None:
            return self.outcome.blocked
        return False

    @property
    def tool_calls(self) -> list:
        """Shortcut: tool calls from the transcript."""
        if self.transcript is not None:
            return self.transcript.tool_calls
        return []

    @property
    def reasoning_steps(self) -> list[str]:
        """Shortcut: reasoning steps from the transcript."""
        if self.transcript is not None:
            return self.transcript.reasoning_steps
        return []

    @property
    def total_duration_ms(self) -> float:
        """Shortcut: total execution duration from the transcript."""
        if self.transcript is not None:
            return self.transcript.total_duration_ms
        return 0.0

    @property
    def has_transcript(self) -> bool:
        """Whether transcript data is available."""
        return self.transcript is not None

    @property
    def has_outcome(self) -> bool:
        """Whether outcome data is available."""
        return self.outcome is not None

    def get_reference_image(self, name: str) -> Image.Image | None:
        """Get a named reference image from the reference_images dict."""
        return self.reference_images.get(name)

    @property
    def has_reference_images(self) -> bool:
        """Whether any reference images are available (single or dict)."""
        if self.reference_image is not None:
            return True
        return len(self.reference_images) > 0

    @property
    def has_output(self) -> bool:
        """Type-agnostic check: does the outcome contain any artifact with output?"""
        if self.outcome is not None:
            return self.outcome.has_output
        return False

    @property
    def code_artifact(self) -> CodeArtifact | None:
        """Shortcut: first CodeArtifact from the outcome, or None."""
        if self.outcome is not None:
            return self.outcome.get_artifact(CodeArtifact)
        return None

    @property
    def text_artifact(self) -> TextArtifact | None:
        """Shortcut: first TextArtifact from the outcome, or None."""
        if self.outcome is not None:
            return self.outcome.get_artifact(TextArtifact)
        return None

    def artifact(self, artifact_type: Type[Artifact]) -> Artifact | None:
        """Generic accessor: first artifact of any registered type."""
        if self.outcome is not None:
            return self.outcome.get_artifact(artifact_type)
        return None

    # ------------------------------------------------------------------
    # Leak detection (Stripe-style answer leakage check)
    # ------------------------------------------------------------------

    def check_leaks(
        self,
        extra_patterns: list[str] | None = None,
    ) -> "LeakCheckResult":
        """Scan transcript for leaked grader/solution markers.

        Searches tool call inputs, outputs, and reasoning steps for any
        of the configured ``leak_markers`` (plus optional extra patterns).
        If any marker is found, the trial should be considered tainted.

        Args:
            extra_patterns: Additional patterns to check beyond
                ``self.leak_markers``.

        Returns:
            LeakCheckResult with leaked markers and searchable text.
        """
        patterns = list(self.leak_markers)
        if extra_patterns:
            patterns.extend(extra_patterns)

        if not patterns or self.transcript is None:
            return LeakCheckResult(leaked=[], searched_patterns=patterns)

        text = self._build_transcript_text()
        leaked = [p for p in patterns if p in text]
        return LeakCheckResult(leaked=leaked, searched_patterns=patterns)

    def _build_transcript_text(self) -> str:
        """Build a searchable text blob from transcript data."""
        if self.transcript is None:
            return ""

        parts: list[str] = []
        for tc in self.transcript.tool_calls:
            if tc.output is not None:
                parts.append(str(tc.output))
            if tc.input:
                parts.append(str(tc.input))
        parts.extend(self.transcript.reasoning_steps)
        return "\n".join(parts)


@dataclass
class LeakCheckResult:
    """Result of a leak detection scan."""

    leaked: list[str]
    searched_patterns: list[str]

    @property
    def has_leaks(self) -> bool:
        return len(self.leaked) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_leaks": self.has_leaks,
            "leaked_markers": self.leaked,
            "total_patterns_checked": len(self.searched_patterns),
        }


def generate_leak_marker(prefix: str = "COMPASS_LEAK") -> str:
    """Generate a unique leak marker for embedding in grader/solution files.

    Example:
        marker = generate_leak_marker()
        # → "COMPASS_LEAK_a1b2c3d4e5f6"

        # Embed in a grader script:
        script_content = f"# {marker}\\nrun_tests()"

        # Configure detection in YAML:
        # leak_markers: ["{marker}"]
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class GradeResult:
    """Result from a grader."""

    name: str = ""
    grader_type: GraderType = GraderType.MODEL
    grader_scope: GraderScope = GraderScope.OUTCOME
    passed: bool = False
    score: float = 0.0
    weight: float = 1.0

    # Detailed results
    details: dict[str, Any] = field(default_factory=dict)
    failure_tags: list[str] = field(default_factory=list)  # Structured failure labels
    reasoning: str = ""

    # Error handling
    error: str | None = None

    @property
    def weighted_score(self) -> float:
        """Calculate weighted score."""
        return self.score * self.weight

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "grader_type": self.grader_type.value,
            "grader_scope": self.grader_scope.value,
            "passed": self.passed,
            "score": self.score,
            "weight": self.weight,
            "weighted_score": self.weighted_score,
            "details": self.details,
            "failure_tags": self.failure_tags,
            "reasoning": self.reasoning,
            "error": self.error,
        }


class Grader(ABC):
    """Abstract base class for all graders."""

    name: str = "base"
    grader_type: GraderType = GraderType.MODEL
    grader_scope: GraderScope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize grader with config."""
        self.config = config or {}

    @abstractmethod
    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade based on the provided context.

        The context provides independent access to:
        - context.transcript  → execution trace (tool calls, reasoning, timing)
        - context.outcome     → final result (image, output data, blocked)
        - context.image       → convenience shortcut for outcome.image
        - context.tool_calls  → convenience shortcut for transcript.tool_calls

        Graders should declare their `grader_scope` to indicate which
        parts of the context they require.

        Args:
            context: GradeContext with transcript, outcome, and input info.

        Returns:
            GradeResult with score and details.
        """
        pass

    def validate_context(self, context: GradeContext) -> str | None:
        """Validate that context has required data for this grader's scope.

        Returns:
            Error message if invalid, None if valid.
        """
        if self.grader_scope == GraderScope.OUTCOME and not context.has_outcome:
            return f"Grader '{self.name}' requires outcome data but none provided"
        if self.grader_scope == GraderScope.TRANSCRIPT and not context.has_transcript:
            return f"Grader '{self.name}' requires transcript data but none provided"
        if self.grader_scope == GraderScope.BOTH:
            if not context.has_outcome:
                return f"Grader '{self.name}' requires outcome data but none provided"
            if not context.has_transcript:
                return f"Grader '{self.name}' requires transcript data but none provided"
        return None

    def validate_config(self) -> bool:
        """Validate grader configuration."""
        return True


class CodeGrader(Grader):
    """Base class for deterministic, code-based graders.

    Code graders are:
    - Fast and deterministic
    - Objective and repeatable
    - Good for: size checks, format validation, state assertions, cost budgets
    - Limitations: brittle to valid variations
    """

    grader_type: GraderType = GraderType.CODE


class ModelGrader(Grader):
    """Base class for model-based (LLM/VLM) graders.

    Model graders are:
    - Flexible and can understand nuance
    - Good for: semantic matching, quality assessment, complex criteria
    - Limitations: non-deterministic, requires calibration
    """

    grader_type: GraderType = GraderType.MODEL


class HumanGrader(Grader):
    """Base class for human evaluation graders.

    Human graders are:
    - Gold standard for quality
    - Good for: subjective quality, complex judgments, calibration
    - Limitations: expensive, slow, doesn't scale
    """

    grader_type: GraderType = GraderType.HUMAN

    async def grade(self, context: GradeContext) -> GradeResult:
        """Human grading - returns pending result for human input."""
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=False,
            score=0.0,
            details={"status": "pending_human_review"},
            reasoning="Awaiting human evaluation",
        )

    async def submit_human_result(
        self,
        task_id: str,
        passed: bool,
        score: float,
        reasoning: str = "",
    ) -> GradeResult:
        """Submit human evaluation result."""
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=score,
            details={"status": "human_reviewed", "task_id": task_id},
            reasoning=reasoning,
        )
